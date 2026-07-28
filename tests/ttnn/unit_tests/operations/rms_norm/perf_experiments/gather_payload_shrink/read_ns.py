# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print DEVICE KERNEL DURATION [ns] per bench case from the newest profiler CSV.

    python3 tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/gather_payload_shrink/read_ns.py <label> [csv]

Rows come out in dispatch order, which is the pytest parametrize order, which is
`bench.CASES` order — the only way to tell `focus` from `focus_hb4` apart (same
shape, different HT_BLOCK).
"""
import csv
import glob
import sys

CASES = ["focus", "focus_hb4", "focus_hb2", "focus_hb1", "w32x1024", "w32x7168", "block8192x2304"]

label = sys.argv[1] if len(sys.argv) > 1 else "?"
path = (
    sys.argv[2] if len(sys.argv) > 2 else sorted(glob.glob("generated/profiler/reports/*/ops_perf_results_*.csv"))[-1]
)
rows = list(csv.reader(open(path)))
hdr = rows[0]
ns_col = next(i for i, h in enumerate(hdr) if h.strip() == "DEVICE KERNEL DURATION [ns]")
core_col = next(i for i, h in enumerate(hdr) if h.strip() == "CORE COUNT")
body = [r for r in rows[1:] if r and r[ns_col].strip()]
print(f"# {label}   csv={path.split('/')[-1]}   ns_col={ns_col}")
for i, r in enumerate(body):
    name = CASES[i] if i < len(CASES) else f"row{i}"
    print(f"{label:16s} {name:16s} {int(r[ns_col]):>9d} ns   cores={r[core_col]}")
