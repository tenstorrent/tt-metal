# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# CI EXPERIMENT BRANCH ONLY (not for merge): device time per op of a profiled run, from the newest
# ops_perf_results*.csv under a folder. Prints one table: op, calls and device-kernel time per device, share.
# Usage: python ci_op_profile_summary.py <profiler output folder> <label>

import csv
import sys
from collections import defaultdict
from pathlib import Path

folder, label = Path(sys.argv[1]), sys.argv[2]
reports = sorted(folder.rglob("ops_perf_results*.csv"), key=lambda p: p.stat().st_mtime)
if not reports:
    sys.exit(f"no ops_perf_results*.csv under {folder}")
report = reports[-1]
time_ns, calls, devices = defaultdict(float), defaultdict(int), set()
with report.open() as f:
    for row in csv.DictReader(f):
        ns = row.get("DEVICE KERNEL DURATION [ns]", "").strip()
        if not ns:
            continue
        devices.add(row.get("DEVICE ID"))
        time_ns[row["OP CODE"]] += float(ns)
        calls[row["OP CODE"]] += 1
n = max(len(devices), 1)
total = sum(time_ns.values())
print(f"OP_PROFILE {label} report={report} devices={n} total_ms_per_device={total / n / 1e6:.3f}")
for op, ns in sorted(time_ns.items(), key=lambda kv: -kv[1]):
    print(
        f"OP_PROFILE {label} {op:<48} calls/dev={calls[op] / n:6.1f} ms/dev={ns / n / 1e6:9.3f} share={100 * ns / total:5.1f}%"
    )
