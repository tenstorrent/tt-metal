# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Print DEVICE KERNEL DURATION (+ per-RISC breakdown) from the newest profiler CSV.

Usage:  python3 tests/.../probes/read_perf_csv.py [csv_path]
One row per profiled op call, keyed by input shape + placement so a multi-case
`--profile` run can be read back without guessing at ordering.
"""
import csv
import glob
import sys

path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob("generated/profiler/reports/*/ops_perf_results*.csv"))[-1]
print(path)
rows = list(csv.DictReader(open(path)))
hdr = f"{'shape':>26} {'mem':>26} {'cores':>5} {'kernel_ns':>10} {'BR(wr)':>8} {'NC(rd)':>8} {'TR2':>8}"
print(hdr)
for r in rows:
    shp = "x".join(r.get(f"INPUT_0_{a}_PAD[LOGICAL]", "?").split("[")[0] for a in ("W", "Z", "Y", "X"))
    print(
        f"{shp:>26} {r.get('INPUT_0_MEMORY',''):>26} {r.get('CORE COUNT',''):>5} "
        f"{r.get('DEVICE KERNEL DURATION [ns]',''):>10} "
        f"{r.get('DEVICE BRISC KERNEL DURATION [ns]',''):>8} "
        f"{r.get('DEVICE NCRISC KERNEL DURATION [ns]',''):>8} "
        f"{r.get('DEVICE TRISC2 KERNEL DURATION [ns]',''):>8}"
    )
