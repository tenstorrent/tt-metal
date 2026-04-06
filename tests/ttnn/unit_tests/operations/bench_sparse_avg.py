"""Run the sparse benchmark 10 times via Tracy and print average device time."""
import subprocess
import re
import glob
import os
import sys

NUM_RUNS = 10
times = []

for run in range(NUM_RUNS):
    folder = f"/tmp/tracy_sparse_run_{run}"
    # Clean up previous
    subprocess.run(["rm", "-rf", folder], check=False)

    result = subprocess.run(
        ["python", "-m", "tracy", "-r", "-o", folder, "tests/ttnn/unit_tests/operations/bench_sparse_reduce.py"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    # Find the CSV
    csvs = glob.glob(f"{folder}/reports/*/ops_perf_results_*.csv")
    if not csvs:
        print(f"Run {run}: no CSV found")
        continue

    # Run tt-perf-report and extract device times
    perf = subprocess.run(["tt-perf-report", csvs[0]], capture_output=True, text=True, timeout=60)

    # Extract all device times for our op (skip warmup: first 3)
    op_times = []
    for line in perf.stdout.split("\n"):
        if "DeepseekMoEPostCombineReduce" in line:
            match = re.search(r"(\d[\d,]*)\s*μs", line)
            if match:
                t = int(match.group(1).replace(",", ""))
                op_times.append(t)

    # Skip first 3 (warmup), take the 10 benchmark iterations
    bench_times = op_times[3:]
    if bench_times:
        avg = sum(bench_times) / len(bench_times)
        times.append(avg)
        print(f"Run {run}: avg={avg:.1f} μs ({len(bench_times)} ops)")

if times:
    overall_avg = sum(times) / len(times)
    print(f"\nOverall average across {len(times)} runs: {overall_avg:.1f} μs")
