import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import platform

# --- 한글 폰트 설정 (Windows/Mac 호환) ---
# 사용하는 OS에 맞춰 적절한 한글 폰트를 지정합니다.
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': # Mac OS
    plt.rc('font', family='AppleGothic')
else: # Linux
    # 리눅스의 경우 나눔고딕 등 별도 폰트 설치가 필요할 수 있습니다.
    plt.rc('font', family='NanumGothic')

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 데이터 불러오기 및 정제 ---

# CSV 파일에서 데이터를 불러옵니다.
try:
    data_df = pd.read_csv('전산업생산지수(계절조정지수).csv')
except FileNotFoundError:
    print("오류: '전산업생산지수(계절조정지수).csv' 파일을 찾을 수 없습니다.")
    print("스크립트와 동일한 폴더에 파일이 있는지 확인해주세요.")
    exit()


# '전산업생산지수' 행을 추출합니다.
industrial_production_series = data_df[data_df['산업별 지수'] == '전산업생산지수']

# 데이터를 'wide' 형식에서 'long' 형식으로 변환합니다.
industrial_production_long = industrial_production_series.melt(
    id_vars=['산업별 지수'],
    var_name='DateStr',
    value_name='IndexValue'
)

# 날짜 문자열을 정리하고 datetime 객체로 변환합니다.
industrial_production_long['DateStr'] = industrial_production_long['DateStr'].str.replace(r'\s*p\)', '', regex=True)
industrial_production_long['Date'] = pd.to_datetime(industrial_production_long['DateStr'], format='%Y.%m')

# 분석에 사용할 깔끔한 데이터프레임을 생성하고 날짜를 인덱스로 설정합니다.
time_series_df = industrial_production_long[['Date', 'IndexValue']].set_index('Date')

# 날짜순으로 정렬하고 결측치를 제거합니다.
time_series_df.sort_index(inplace=True)
time_series_df.dropna(inplace=True)


# --- 2. 데이터 전처리 ---

# 분석을 위해 지수 값에 자연로그를 적용합니다.
time_series_df['LogIndexValue'] = np.log(time_series_df['IndexValue'])


# --- 3. 밴드패스 필터 적용 ---

# 경기 순환 주기를 월 단위로 설정 (6분기~32분기 -> 18개월~96개월)
# 백스터-킹 필터를 사용하여 경기 순환 성분을 추출합니다.
cyclical_component = sm.tsa.filters.bkfilter(
    time_series_df['LogIndexValue'],
    low=18,
    high=96,
    K=12
)


# --- 4. 시각화 (한국어 적용) ---

plt.figure(figsize=(15, 7))
plt.plot(cyclical_component.index, cyclical_component, label='경기 순환 성분 (밴드패스 필터)')
plt.axhline(0, color='red', linestyle='--', linewidth=0.8, label='추세선 (0)')

# 제목과 축 레이블을 한국어로 변경
plt.title('전산업생산지수 경기변동 성분 (2000-2025)', fontsize=16)
plt.xlabel('연도')
plt.ylabel('경기 순환 성분 (로그 편차)')

plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()

# 시각화 결과물만 .png 파일로 저장 (CSV 저장 코드는 삭제)
output_filename = '전산업생산지수_경기변동시각화.png'
plt.savefig(output_filename)

print(f"경기 변동 시각화 그래프가 '{output_filename}' 파일로 저장되었습니다.")
