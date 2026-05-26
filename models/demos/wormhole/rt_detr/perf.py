import pandas as pd
df = pd.read_csv('ops_perf_results_rtdetr_2026_05_25_05_51_41.csv')
print(f"Host Time:   {df['HOST DURATION [ns]'].sum() / 1_000_000:.2f} ms")
print(f"Device Time: {df['DEVICE FW DURATION [ns]'].sum() / 1_000_000:.2f} ms")