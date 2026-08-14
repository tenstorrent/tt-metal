# Print DEVICE KERNEL DURATION [ns] from the newest profiler CSV, in call order.
import csv
import glob
import os
import sys

pat = "generated/profiler/reports/*/ops_perf_results_*.csv"
path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob(pat), key=os.path.getmtime)
rows = list(csv.DictReader(open(path)))
print(os.path.basename(path))
for i, r in enumerate(rows):
    print(f"  {i}  {r['DEVICE KERNEL DURATION [ns]']:>8}  cores={r['CORE COUNT']:>3}  {r['INPUT_0_MEMORY']}")
