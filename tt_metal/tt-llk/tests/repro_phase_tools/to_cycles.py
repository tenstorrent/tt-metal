#!/usr/bin/env python3
"""Turn change lists from extract_signals.py into one row per rising clock edge (values just before the edge).

usage: to_cycles.py <out.cyc> <signals.txt> [more.txt ...]
"""
import sys

changes = []
for fn in sys.argv[2:]:
    for line in open(fn):
        if line[0] != "#":
            t, label, val = line.split()
            changes.append((int(t), label, val))
changes.sort(key=lambda c: c[0])
cur, rows, prev_clk, i = {}, [], "0", 0
while i < len(changes):
    t = changes[i][0]
    batch = {}
    while i < len(changes) and changes[i][0] == t:
        batch[changes[i][1]] = changes[i][2]
        i += 1
    clk = batch.pop("clk", None)
    if clk is not None:
        if prev_clk == "0" and clk == "1":
            rows.append(dict(cur))
        prev_clk = clk
    cur.update(batch)
keys = sorted({k for r in rows for k in r})
with open(sys.argv[1], "w") as w:
    w.write(" ".join(keys) + "\n")
    for r in rows:
        w.write(" ".join(r.get(k, "?") for k in keys) + "\n")
print(len(rows), "cycles")
