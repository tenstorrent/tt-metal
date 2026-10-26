#!/usr/bin/env python3
"""mdtab.py CFG PATHS NAME=TXFILE ...: markdown table, rows = variant x level, cells instr+stall5."""
import sys
cfg = sys.argv[1]; paths = sys.argv[2].split(",")
vs = [a.split("=") for a in sys.argv[3:]]
def load(f):
    d = {}
    for l in open(f):
        p = l.split()
        if len(p) < 3: continue
        vals = {}
        for i in range(2, len(p) - 1, 2):
            v = p[i + 1].split("/")
            if len(v) == 4: vals[p[i]] = (int(v[0]), int(v[3]))
        d[(p[0], p[1])] = vals
    return d
T = {n: load(f) for n, f in vs}
print("| variant | level | " + " | ".join(paths) + " |")
print("| --- | --- | " + " | ".join("---:" for _ in paths) + " |")
for lvl in ("O3", "O2", "Os"):
    for n, _ in vs:
        r = T[n].get((cfg, lvl), {})
        cells = [f"{r[p][0]}+{r[p][1]}" if p in r else "-" for p in paths]
        print(f"| {n} | -{lvl} | " + " | ".join(cells) + " |")
