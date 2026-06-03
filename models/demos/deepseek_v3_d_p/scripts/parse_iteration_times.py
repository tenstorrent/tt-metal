#!/usr/bin/env python3
import argparse
import re
import statistics
import subprocess
from datetime import datetime


def parse_timestamp(ts_str):
    return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")


def filter_outliers(values, threshold=2.0):
    if len(values) < 2:
        return values, []
    median = statistics.median(values)
    cutoff = median * threshold
    filtered = [v for v in values if v <= cutoff]
    outliers = [v for v in values if v > cutoff]
    return filtered, outliers


def main():
    parser = argparse.ArgumentParser(description="Parse iteration timing from log files")
    parser.add_argument("-f", "--file", default="./260421-log5", help="Log file path")
    args = parser.parse_args()

    iter_pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*Starting iteration: (\d+)")
    sync_pattern = re.compile(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}).*Starting completion sync on iteration: (\d+)"
    )

    iterations = {}
    syncs = {}

    proc = subprocess.Popen(
        ["grep", "Starting", args.file],
        stdout=subprocess.PIPE,
        text=True,
    )

    for line in proc.stdout:
        m = iter_pattern.search(line)
        if m:
            ts, num = m.groups()
            iterations[int(num)] = parse_timestamp(ts)
            continue

        m = sync_pattern.search(line)
        if m:
            ts, num = m.groups()
            syncs[int(num)] = parse_timestamp(ts)

    proc.wait()

    iter_deltas = []
    sorted_iters = sorted(iterations.keys())
    for i in range(1, len(sorted_iters)):
        prev, curr = sorted_iters[i - 1], sorted_iters[i]
        if curr == prev + 1:
            delta = (iterations[curr] - iterations[prev]).total_seconds()
            iter_deltas.append(delta)

    sync_deltas = []
    for sync_num, sync_ts in syncs.items():
        next_iter = sync_num + 1
        if next_iter in iterations:
            delta = (iterations[next_iter] - sync_ts).total_seconds()
            sync_deltas.append(delta)

    print("=== Iteration Period ===")
    if iter_deltas:
        filtered, outliers = filter_outliers(iter_deltas)
        print(f"Samples: {len(filtered)} (filtered {len(outliers)} outliers)")
        print(f"Min: {min(filtered):.3f}s | Avg: {statistics.mean(filtered):.3f}s | Max: {max(filtered):.3f}s")
    else:
        print("No data")

    print()
    print("=== Sync to Next Iteration ===")
    if sync_deltas:
        filtered, outliers = filter_outliers(sync_deltas)
        print(f"Samples: {len(filtered)} (filtered {len(outliers)} outliers)")
        print(f"Min: {min(filtered):.3f}s | Avg: {statistics.mean(filtered):.3f}s | Max: {max(filtered):.3f}s")
    else:
        print("No sync data found")


if __name__ == "__main__":
    main()
