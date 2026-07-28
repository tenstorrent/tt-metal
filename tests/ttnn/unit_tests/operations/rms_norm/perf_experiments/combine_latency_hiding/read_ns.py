# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print DEVICE KERNEL DURATION [ns] per bench row from the newest profiler CSV.

    python3 tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/combine_latency_hiding/read_ns.py <label> [csv] [names]

Rows come out in dispatch order, which is the pytest parametrize order. Pass
`names` (comma-separated) to label rows when the run used `-k` to select a
subset (row i maps to the i-th name instead of the default full list).
"""
import csv
import glob
import sys

NAMES = [
    "focus",
    "w32x1024",
    "w32x2304",
    "w32x5120",
    "w32x7168",
    "i32x5120",
    "i32x7168",
    "i8192x1024",
    "i8192x7168",
]

label = sys.argv[1] if len(sys.argv) > 1 else "?"
path = (
    sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))[-1]
)
if len(sys.argv) > 3 and sys.argv[3]:
    NAMES = sys.argv[3].split(",")
rows = list(csv.reader(open(path)))
hdr = rows[0]
ns_col = next(i for i, h in enumerate(hdr) if h.strip() == "DEVICE KERNEL DURATION [ns]")
core_col = next(i for i, h in enumerate(hdr) if h.strip() == "CORE COUNT")
body = [r for r in rows[1:] if r and r[ns_col].strip()]
print(f"# {label}   csv={path.split('/')[-1]}   ns_col={ns_col}")
for i, r in enumerate(body):
    name = NAMES[i] if i < len(NAMES) else f"row{i}"
    print(f"{label:20s} {name:16s} {int(r[ns_col]):>9d} ns   cores={r[core_col]}")
