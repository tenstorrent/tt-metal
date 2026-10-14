#!/usr/bin/env python3
"""Scratch, not for merge. From the job log's TAB OP lines: per case the median over runs of each run's median device
time per arm, B vs A and the A/A (B2 vs B) in percent, and the max per-run spread of each arm."""
import statistics as st
import sys
from collections import defaultdict

d = defaultdict(lambda: defaultdict(list))
for line in open(sys.argv[1], errors="replace"):
    p = line.split()
    if len(p) >= 8 and p[0] == "TAB" and p[1] == "OP":
        arm, case, vals = p[2], p[4], [float(x) for x in p[7].split(",") if x]
        if vals:
            d[case][arm].append(st.median(vals))
print("TAB HDR case nA nB nB2 A_us B_us B2_us BvsA% B2vsB% spreadA% spreadB%")
for case, arms in d.items():
    m = {a: st.median(v) for a, v in arms.items() if v}
    if not {"A", "B", "B2"} <= m.keys():
        print(f"TAB ROW {case} incomplete {sorted((a, len(v)) for a, v in arms.items())}")
        continue
    sp = {a: 100 * (max(arms[a]) - min(arms[a])) / m[a] for a in m}
    print(f"TAB ROW {case} {len(arms['A'])} {len(arms['B'])} {len(arms['B2'])} {m['A']/1e3:.2f} {m['B']/1e3:.2f} "
          f"{m['B2']/1e3:.2f} {100*(m['B']/m['A']-1):+.2f} {100*(m['B2']/m['B']-1):+.2f} {sp['A']:.2f} {sp['B']:.2f}")
    for a in sorted(set(m) - {"A", "B", "B2"}):
        print(f"TAB DROW {case} {a} {len(arms[a])} {m[a]/1e3:.2f} vsA {100*(m[a]/m['A']-1):+.2f} "
              f"vsB {100*(m[a]/m['B']-1):+.2f} spread {sp[a]:.2f}")
