#!/usr/bin/env python3
"""Parse REDUCE-MATMUL-TILE / REDUCE-TILE zones from device profiler CSV."""
import csv
import sys
from collections import defaultdict


def parse_zones(csv_path, zone_name="REDUCE-MATMUL-TILE"):
    core_starts = defaultdict(list)
    core_ends = defaultdict(list)

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # header 1 (ARCH line)
        next(reader)  # header 2 (column names)
        for row in reader:
            if len(row) < 12:
                continue
            if zone_name in row[10]:
                core = (row[1], row[2])
                cycles = int(row[5])
                zone_type = row[11]
                if zone_type == "ZONE_START":
                    core_starts[core].append(cycles)
                elif zone_type == "ZONE_END":
                    core_ends[core].append(cycles)

    all_durations = []
    for core in sorted(core_starts.keys()):
        starts = core_starts[core]
        ends = core_ends[core]
        n = min(len(starts), len(ends))
        durations = [ends[i] - starts[i] for i in range(n)]
        all_durations.extend(durations)

    if not all_durations:
        print(f"  No {zone_name} zones found!")
        return

    # Filter out outliers (>2x median) for steady-state analysis
    sorted_d = sorted(all_durations)
    median = sorted_d[len(sorted_d) // 2]
    steady = [d for d in all_durations if d <= median * 3]
    outliers = [d for d in all_durations if d > median * 3]

    num_cores = len(core_starts)
    calls_per_core = len(all_durations) // num_cores if num_cores > 0 else 0

    print(f"  Zone: {zone_name}")
    print(f"  Cores: {num_cores}, Calls/core: {calls_per_core}, Total calls: {len(all_durations)}")
    print(
        f"  ALL     - min: {min(all_durations):>5}, max: {max(all_durations):>5}, avg: {sum(all_durations)/len(all_durations):>7.1f}"
    )
    if steady:
        print(
            f"  STEADY  - min: {min(steady):>5}, max: {max(steady):>5}, avg: {sum(steady)/len(steady):>7.1f} ({len(steady)} samples)"
        )
    if outliers:
        print(f"  OUTLIER - min: {min(outliers):>5}, max: {max(outliers):>5}, count: {len(outliers)}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not csv_path:
        print("Usage: python parse_zone_cycles.py <profile_log_device.csv>")
        sys.exit(1)
    parse_zones(csv_path, "REDUCE-MATMUL-TILE")
    parse_zones(csv_path, "REDUCE-TILE")
