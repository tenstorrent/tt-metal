# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print DEVICE KERNEL DURATION [ns] per dispatch from the newest profiler CSV.

    python3 .../gamma_broadcast_rowsplit/read_ns.py "<GBR_PLAN>" [csv]

Rows come out in dispatch order, which is the pytest parametrize order, which is
GBR_PLAN order — so the plan string labels them.
"""
import csv
import glob
import sys

plan = [p.strip() for p in (sys.argv[1] if len(sys.argv) > 1 else "").split(",") if p.strip()]
path = (
    sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))[-1]
)
rows = list(csv.reader(open(path)))
hdr = rows[0]
ns_col = next(i for i, h in enumerate(hdr) if h.strip() == "DEVICE KERNEL DURATION [ns]")
core_col = next(i for i, h in enumerate(hdr) if h.strip() == "CORE COUNT")
code_col = next(i for i, h in enumerate(hdr) if h.strip() == "OP CODE")
body = [r for r in rows[1:] if r and r[ns_col].strip()]
print(f"# csv={path}")
print(f"# {len(body)} timed rows, {len(plan)} plan entries")
for i, r in enumerate(body):
    label = plan[i] if i < len(plan) else f"row{i}"
    print(f"{label:34s} {int(r[ns_col]):>10d} ns   cores={r[core_col]:>4}  op={r[code_col]}")
