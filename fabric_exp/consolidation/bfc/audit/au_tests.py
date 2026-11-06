#!/usr/bin/env python3
"""au_tests.py: placement-audit tables from the per-test CSVs (rows written before any abort), all runs of jobs A and
B on a09u32. Bases: A b0, A b1, B b0 (B's closing base runs failed). Delta = arm - mean of the bases present."""
import collections, csv, os
R = "/home/aagarwal/tt-metal-mainline/.scratch/val/results"
def t(label):
    p = f"{R}/{label}.tests.csv"
    acc = collections.defaultdict(list)
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            acc[(r["test_name"].split("_size_")[0], int(r["packet_size"]))].append(float(r["packets_per_second"]))
    return {k: 1.35e9 / (sum(v) / len(v)) for k, v in acc.items()}
for kind in ("two", "one"):
    bases = {"A0": t(f"pauA_0_base_{kind}"), "A4": t(f"pauA_4_base_{kind}"), "B0": t(f"pauB_0_base_{kind}")}
    for lab in sorted(l[:-10] for l in os.listdir(R) if l.startswith("pau") and l.endswith(f"_{kind}.tests.csv")):
        if "_base_" in lab:
            continue
        v = t(lab)
        if not v:
            print(f"-- {lab}: no rows"); continue
        print(f"-- {lab}")
        print(f"   {'row':18s} {'B':>5s} {'A b0':>7s} {'A b1':>7s} {'B b0':>7s} {'arm':>7s} {'d':>6s} {'spread':>6s}")
        for k in sorted(v):
            bs = [b[k] for b in bases.values() if k in b]
            if not bs:
                print(f"   {k[0]:18s} {k[1]:5d} no base"); continue
            d, sp = v[k] - sum(bs) / len(bs), max(bs) - min(bs)
            cols = " ".join(f"{bases[n][k]:7.1f}" if k in bases[n] else f"{'-':>7s}" for n in ("A0", "A4", "B0"))
            print(f"   {k[0]:18s} {k[1]:5d} {cols} {v[k]:7.1f} {d:+6.1f} {sp:6.1f}{' *' if abs(d) > max(sp, 0.5) else ''}")
