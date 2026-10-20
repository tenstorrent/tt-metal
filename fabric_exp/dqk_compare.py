#!/usr/bin/env python3
"""§47/§48: router kernel v2 (<prefix>*/<prefix>b*, default dqk) against v1 (dqi*/dqib*, reproduced today by dqv1*): per suite, mean change and the
rows that moved >= 1 cycle/packet (cycles/packet; MixCmd excluded, it is bimodal)."""
import csv, os, re, statistics, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def bw(l):
    f = os.path.join(R, l + ".csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r.get("packet_size", ""))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def lat(l):
    f = os.path.join(R, l + "_latency.csv")
    if not os.path.exists(f):
        return None
    g = {}
    for r in csv.DictReader(open(f)):
        g.setdefault(re.sub(r"_sample\d+$", "", r["test_name"]), []).append(float(r["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in g.items()}


P = sys.argv[1] if len(sys.argv) > 1 else "dqk"
for title, a, b in (
    ("ERISC-only", "dqib_13129088", f"{P}b_13129088"),
    ("offload", "dqi_15226240", f"{P}_15226240"),
    ("offload W32", "dqi_15226240w32", f"{P}_15226240w32"),
):
    for s in ("4352", "mix", "4352_trim", "mix_trim", "small"):
        x, y = bw(f"{a}_{s}"), bw(f"{b}_{s}")
        if not x or not y:
            continue
        d = [(y[k] - x[k], k) for k in x if k in y and "MixCmd" not in k[0]]
        moved = sorted([t for t in d if abs(t[0]) >= 1.0])
        print(f"{title} {s}: {len(d)} rows, mean {sum(v for v, _ in d) / len(d):+.2f}, moved >= 1: {len(moved)}")
        for v, k in moved:
            plain = "Mix" in k[0] or "Pure" in k[0]
            print(
                f'   {v:+6.1f}  {k[0]}{"" if plain else " " + k[1] + " " + k[2]}{" (replay)" if k[3] else ""}: {x[k]:.1f} -> {y[k]:.1f}'
            )
    la, lb = lat(f"{a}_lat"), lat(f"{b}_lat")
    if la and lb:
        print(
            f"{title} latency (ns): "
            + ", ".join(
                f'{k.replace("_ntype_unicast_write_size", "").replace("_ntype_fused_atomic_inc_size", "_fused")} {la[k]:.0f}->{lb[k]:.0f}'
                for k in la
                if k in lb
            )
        )
