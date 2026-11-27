"""Print each op's DEVICE KERNEL / NCRISC / BRISC duration (ns) from the newest ops_perf_results CSV, or the CSV given."""
import csv
import glob
import os
import sys

path = (
    sys.argv[1]
    if len(sys.argv) > 1
    else max(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"), key=os.path.getmtime)
)
print(path)
for i, row in enumerate(csv.DictReader(open(path))):
    print(
        i,
        row["DEVICE KERNEL DURATION [ns]"],
        row["DEVICE NCRISC KERNEL DURATION [ns]"],
        row["DEVICE BRISC KERNEL DURATION [ns]"],
    )
