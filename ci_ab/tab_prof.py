#!/usr/bin/env python3
"""tab_prof.py <PRO file(s)>: median over repetitions k>=1 per (case, op, kernel) of every key=value field of the PRO lines."""
import re, sys, statistics as st
from collections import defaultdict
acc = defaultdict(lambda: defaultdict(list)); order = []
for fn in sys.argv[1:]:
    for l in open(fn):
        p = l.split()
        if len(p) < 8 or p[0] != "PRO" or p[1] == "#": continue
        case, k, op, src = p[3], int(p[4][1:]), p[5], p[6]
        if k < 1: continue
        key = (case, op, src)
        if key not in acc: order.append(key)
        for kv in p[7:]:
            n, _, v = kv.partition("=")
            try: acc[key][n].append(float(v))
            except ValueError: pass
cols = ["cores", "dur", "t0", "t1", "t2", "br", "nc", "U", "F", "P", "S0", "S1", "S2", "T0", "T1", "T2"]
print("case op src | " + " ".join(cols))
for key in order:
    d = acc[key]
    print(" | ".join(key) + " | " + " ".join(f"{c}={st.median(d[c]):.1f}" if d.get(c) else f"{c}=na" for c in cols))
