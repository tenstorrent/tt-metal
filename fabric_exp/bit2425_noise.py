#!/usr/bin/env python3
"""Bits 24/25 noise check: cycles/packet over the first run (b_*/x_*) and the repeats (r1_*, r2_*), untrimmed only.
Each cell: mean [min..max] over runs; delta = mean minus the base config's mean, '*' when the two ranges do not overlap."""
import csv, os

R = "/home/aagarwal/tt-metal/fabric_exp/results"
STOCK = [("13129088", "base"), ("29906304", "+24"), ("46683520", "+25"), ("63460736", "+24+25")]
BLIND = [("15226240w32", "blind"), ("32003456w32", "blind+24"), ("65557888w32", "blind+24+25")]


def load(label):
    f = os.path.join(R, label + ".csv")
    out = {}
    if os.path.exists(f):
        for r in csv.DictReader(open(f)):
            out[(r["test_name"], r["ntype"])] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def runs(cfg, kind):
    if kind == "4352":
        labels = [f"b_{cfg}_4352", f"r1_{cfg}_4352", f"r2_{cfg}_4352"]
    else:
        labels = [f"x_{cfg}_mix", f"r1_{cfg}_mixew", f"r2_{cfg}_mixew"]
    data = [load(l) for l in labels]
    return [d for d in data if d]


def table(cfgs, kind, keep):
    per = {cfg: runs(cfg, kind) for cfg, _ in cfgs}
    keys = [k for k in per[cfgs[0][0]][0] if keep(k)] if per[cfgs[0][0]] else []
    print(f"\n{kind}: " + ", ".join(f"{n} n={len(per[c])}" for c, n in cfgs) + "\n")
    print("| test | " + " | ".join(n for _, n in cfgs) + " |")
    print("| --- |" + " ---: |" * len(cfgs))
    for k in keys:
        cells = []
        base = None
        for cfg, _ in cfgs:
            vals = [d[k] for d in per[cfg] if k in d]
            if not vals:
                cells.append("-")
                continue
            m, lo, hi = sum(vals) / len(vals), min(vals), max(vals)
            s = f"{m:.1f} [{lo:.1f}..{hi:.1f}]"
            if base is None:
                base = (m, lo, hi)
            else:
                sep = "*" if lo > base[2] or hi < base[1] else ""
                s += f" {m - base[0]:+.1f}{sep}"
            cells.append(s)
        name = k[0] + ("" if "Mix" in k[0] or "Pure" in k[0] else (" scatter" if "SCATTER" in k[1] else " write"))
        print(f"| {name} | " + " | ".join(cells) + " |")


table(STOCK, "4352", lambda k: True)
table(BLIND, "4352", lambda k: k[0] not in ("Mesh3HopEW", "MeshMulticast"))
table(STOCK, "mix", lambda k: k[0].startswith("EW_"))
