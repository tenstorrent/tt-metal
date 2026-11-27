# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Summarise G6PAD_* device zones per op run from generated/profiler/.logs/profile_log_device.csv.

Usage: python3 zone_summary.py [profile_log_device.csv]
Prints, per run host ID (one per op call), per RISC and zone: core count, median and max duration
(ns), plus the kernel span (first kernel start to last kernel end) across cores."""

import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
with open(path) as f:
    header = f.readline()
    freq_mhz = float(header.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(f, skipinitialspace=True))

open_zones = {}
durations = defaultdict(list)  # (run, risc, zone) -> [ns]
kernel_span = defaultdict(lambda: [None, None])
for r in rows:
    zone = r["zone name"]
    if not (zone.startswith("G6PAD_") or zone.endswith("-KERNEL")):
        continue
    key = (r["run host ID"], r["core_x"], r["core_y"], r["RISC processor type"], zone)
    t = int(r["time[cycles since reset]"])
    if r["type"] == "ZONE_START":
        open_zones[key] = t
    elif r["type"] == "ZONE_END" and key in open_zones:
        start = open_zones.pop(key)
        run = r["run host ID"]
        if zone.endswith("-KERNEL"):
            span = kernel_span[run]
            span[0] = start if span[0] is None else min(span[0], start)
            span[1] = t if span[1] is None else max(span[1], t)
        else:
            durations[(run, r["RISC processor type"], zone)].append((t - start) * 1000.0 / freq_mhz)

for run in sorted({k[0] for k in durations} | set(kernel_span), key=int):
    span = kernel_span.get(run)
    span_ns = (span[1] - span[0]) * 1000.0 / freq_mhz if span and span[0] is not None else float("nan")
    print(f"run {run}: kernel span {span_ns:.0f} ns")
    for (r_run, risc, zone), ds in sorted(durations.items()):
        if r_run != run:
            continue
        print(f"   {risc:6s} {zone:18s} cores={len(ds):4d} median={statistics.median(ds):8.0f} max={max(ds):8.0f}")
