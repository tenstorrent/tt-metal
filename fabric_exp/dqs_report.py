#!/usr/bin/env python3
"""§39 DRISC delay sweep (dqs<d>_15226240_*, d = TT_FABRIC_EXP_DRISC_DELAY): cycles/packet per row at each d, the
same-tree baseline (dqgb_13129088_*), and the DRISC's own loop estimated from the two largest delays
(loop = 2 * rate(64) - rate(96), valid once both are on the slope-1 line; 'slope' shows rate(96) - rate(64) / 32)."""
import csv, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
D = (0, 32, 64, 96)


def load(label):
    out = {}
    for r in csv.DictReader(open(os.path.join(R, label + ".csv"))):
        k = (r["test_name"], r["ntype"], r.get("packet_size", ""))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


for s in ("4352", "mix"):
    runs = {d: load(f"dqs{d}_15226240_{s}") for d in D}
    base = load(f"dqgb_13129088_{s}")
    print(
        f'\n{s}: {"row":38s} {"base":>6s}'
        + "".join(f' {"d=" + str(d):>7s}' for d in D)
        + f' {"slope":>6s} {"loop":>6s} {"slack":>6s}'
    )
    for k in runs[0]:
        plain = "Mix" in k[0] or "Pure" in k[0]
        name = k[0] + ("" if plain else " " + k[1].replace("NOC_UNICAST_", "").lower())
        r = [runs[d].get(k, float("nan")) for d in D]
        slope = (r[3] - r[2]) / 32
        loop = 2 * r[2] - r[3]
        print(
            f"  {name[:44]:44s} {base.get(k, 0):6.1f}"
            + "".join(f" {x:7.1f}" for x in r)
            + f" {slope:6.2f} {loop:6.1f} {r[0] - loop:6.1f}"
        )
