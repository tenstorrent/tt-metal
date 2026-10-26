#!/usr/bin/env python3
"""score.py BASE_TABLE VAR_TABLE: tables from stab5.sh (lines 'CFG LVL scen i/t/ld/st ...'). Prints per (cfg, lvl) the
deltas of instructions and of instructions + stall, per path, and sums (paths: write fwd atomic fused fusedfl scatter;
completions as (compu - write) / 4 and (compm - write) / 4 added to the write path's per-packet cost)."""
import sys, re
def load(f):
    d = {}
    for l in open(f):
        p = l.split()
        if len(p) < 3: continue
        key = (p[0], p[1])
        vals = {}
        for i in range(2, len(p) - 1, 2):
            sc = p[i]; v = p[i + 1].split("/")
            if len(v) == 4 and all(x.lstrip("-").isdigit() for x in v):
                vals[sc] = tuple(int(x) for x in v)
        d[key] = vals
    return d
A = load(sys.argv[1]); B = load(sys.argv[2])
scs = ["write", "fwd", "atomic", "fused", "fusedfl", "scatter", "compu", "compm"]
tot = {}
for key in sorted(A):
    if key not in B: continue
    a, b = A[key], B[key]
    row = []
    si = ss = 0
    for sc in scs:
        if sc in a and sc in b:
            di = b[sc][0] - a[sc][0]; ds = (b[sc][0] + b[sc][3]) - (a[sc][0] + a[sc][3]); dt = b[sc][1] - a[sc][1]
            row.append(f"{sc}{di:+d}/{ds:+d}/t{dt:+d}")
            if sc not in ("compu", "compm"): si += di; ss += ds
        else:
            row.append(f"{sc} ?")
    print(f"{key[0]:3s} {key[1]} sum {si:+3d}/{ss:+3d} | " + " ".join(row))
