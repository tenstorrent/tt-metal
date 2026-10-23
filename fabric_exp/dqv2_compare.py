#!/usr/bin/env python3
"""§49: router kernel versions side by side, cycles/packet: v1 (dqi/dqib), v2 second version (dql/dqlb) and a newer v2
(prefix given, default dqo/dqob). Per suite: the mean change of each v2 against v1, and every row where either v2 moves
>= 1 cycle against v1 (MixCmd excluded: bimodal). Latency: median net round trip in ns.
usage: dqv2_compare.py [prefix]"""
import csv, os, re, statistics, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P = sys.argv[1] if len(sys.argv) > 1 else "dqo"
BUILDS = (("ERISC-only", "13129088", "b"), ("offload", "15226240", ""), ("offload W32", "15226240w32", ""))


def bw(label):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r.get("packet_size", ""))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def lat(label):
    f = os.path.join(R, label + "_latency.csv")
    if not os.path.exists(f):
        return None
    g = {}
    for r in csv.DictReader(open(f)):
        g.setdefault(re.sub(r"_sample\d+$", "", r["test_name"]), []).append(float(r["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in g.items()}


def name(k):
    plain = "Mix" in k[0] or "Pure" in k[0]
    return k[0] + ("" if plain else f" {k[1]} {k[2]}") + (" (replay)" if k[3] else "")


for title, flags, b in BUILDS:
    for s in ("4352", "mix", "4352_trim", "mix_trim", "small"):
        v1, v2b, v2c = (bw(f"{p}{b}_{flags}_{s}") for p in ("dqi", "dql", P))
        if not (v1 and v2b and v2c):
            continue
        keys = [k for k in v1 if k in v2b and k in v2c and "MixCmd" not in k[0]]
        mean = lambda d: sum(d[k] - v1[k] for k in keys) / len(keys)
        print(f"\n{title} {s}: {len(keys)} rows; mean vs v1: dql {mean(v2b):+.2f}, {P} {mean(v2c):+.2f}")
        moved = [k for k in keys if abs(v2c[k] - v1[k]) >= 1 or abs(v2b[k] - v1[k]) >= 1]
        if moved:
            print(f"   {'row':52s} {'v1':>7s} {'dql':>7s} {P:>7s}")
        for k in sorted(moved, key=lambda k: v2c[k] - v1[k]):
            print(f"   {name(k)[:52]:52s} {v1[k]:7.1f} {v2b[k]:7.1f} {v2c[k]:7.1f}  ({v2c[k] - v1[k]:+5.1f})")
    if flags.endswith("w32"):
        continue
    L1, L2, L3 = (lat(f"{p}{b}_{flags}_lat") for p in ("dqi", "dql", P))
    if L1 and L2 and L3:
        print(f"\n{title} latency (ns), v1 / dql / {P}:")
        for k in L1:
            if k in L2 and k in L3:
                short = k.replace("_ntype_unicast_write_size", "").replace("_ntype_fused_atomic_inc_size", "_fused")
                print(f"   {short:28s} {L1[k]:7.0f} {L2[k]:7.0f} {L3[k]:7.0f}  ({L3[k] - L1[k]:+5.0f})")
