#!/usr/bin/env python3
"""Parse device profiler log to find per-core kernel durations."""
import csv
import sys

logfile = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"

cores = {}
with open(logfile) as f:
    reader = csv.reader(f)
    next(reader)  # header 1
    next(reader)  # header 2
    for row in reader:
        if len(row) < 12:
            continue
        slot, cx, cy, risc = row[0], row[1], row[2], row[3]
        cycles_str = row[5]
        zone_name = row[10] if len(row) > 10 else ""
        zone_type = row[11] if len(row) > 11 else ""

        key = (slot, cx, cy, risc)
        if key not in cores:
            cores[key] = {}

        if "KERNEL" in zone_name:
            if "ZONE_START" in zone_type:
                cores[key]["kernel_start"] = int(cycles_str)
            elif "ZONE_END" in zone_type:
                cores[key]["kernel_end"] = int(cycles_str)

results = []
for key, data in cores.items():
    if "kernel_start" in data and "kernel_end" in data:
        dur_cycles = data["kernel_end"] - data["kernel_start"]
        dur_us = dur_cycles / 1000.0  # 1GHz clock
        results.append((dur_us, key))

results.sort(reverse=True)
print("%-25s %-10s %15s" % ("Core", "RISC", "Duration (us)"))
print("-" * 55)
for dur_us, (slot, cx, cy, risc) in results[:40]:
    print("  dev=%-3s (%s,%s)  %-10s %12.1f us" % (slot, cx, cy, risc, dur_us))
print("\nTotal cores with kernel data: %d" % len(results))
