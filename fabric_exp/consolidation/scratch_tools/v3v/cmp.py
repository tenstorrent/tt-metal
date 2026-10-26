#!/usr/bin/env python3
# cmp.py A_TABLE B_TABLE [...]: side-by-side instruction counts per config/level/path.
import re, sys
tabs = []
for f in sys.argv[1:]:
    t = {}
    for l in open(f):
        p = l.split()
        if len(p) < 3: continue
        c, lv = p[0], p[1]
        for m in re.finditer(r"(\w+) (\d+|path)/?(\d*)", " ".join(p[2:])):
            t[(c, lv, m.group(1))] = m.group(2)
        tabs.append
    tabs.append(t)
paths = ["write", "fwd", "atomic", "fused", "fusedfl", "scatter", "compu", "compm"]
print("cfg lvl " + " ".join(f"{p:>15s}" for p in paths))
for c in ["W", "NS", "1D", "VC1"]:
    for lv in ["O3", "O2", "Os"]:
        print(f"{c:3s} {lv:3s} " + " ".join(f"{'/'.join(t.get((c, lv, p), '-') for t in tabs):>15s}" for p in paths))
