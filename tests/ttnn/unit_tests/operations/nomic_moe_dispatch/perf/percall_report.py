# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Summarize test_nomic_moe_dispatch_percall.py's profiler CSV: per case, sorted per-call device us, median, mean."""

import csv
import os
import statistics
import sys

rows = [r for r in csv.DictReader(open(sys.argv[1])) if r.get("OP CODE") == "GenericOpDeviceOperation"]
durs = [float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in rows]
cases = [c for c in os.environ.get("NMD_PERCALL_CASES", "8416:").split(";") if c]
calls = int(os.environ.get("NMD_PERCALL_CALLS", "10"))
per = calls + 1
if len(durs) != per * len(cases):
    print(f"warning: {len(durs)} op rows for {len(cases)} cases x {per}")
for i, case in enumerate(cases):
    d = durs[i * per + 1 : (i + 1) * per]
    if not d:
        continue
    print(
        f"{case:45s} med {statistics.median(d):6.2f} mean {statistics.mean(d):6.2f} min {min(d):6.2f} max {max(d):6.2f} | "
        + " ".join(f"{x:.2f}" for x in sorted(d))
    )
