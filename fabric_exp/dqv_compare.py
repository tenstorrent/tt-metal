#!/usr/bin/env python3
"""§40/§41: DRISC offload versions side by side (cycles/packet; latency in ns): same-tree baseline of the newest run,
v2 (dqg), v3 (dqh), v3 without the forward inline (dqab_noinline), v4 (dqi). usage: dqv_compare.py [suite ...]"""
import csv, os, re, statistics, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COLS = [
    ("base", "dqib_13129088"),
    ("v2", "dqg_15226240"),
    ("v3", "dqh_15226240"),
    ("v3noinl", "dqab_noinline_15226240"),
    ("v4", "dqi_15226240"),
]


def bw(label):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return {}
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r.get("packet_size", ""))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


def lat(label):
    f = os.path.join(R, label + "_latency.csv")
    if not os.path.exists(f):
        return {}
    g = {}
    for r in csv.DictReader(open(f)):
        g.setdefault(re.sub(r"_sample\d+$", "", r["test_name"]), []).append(float(r["net_avg_ns"]))
    return {k: statistics.median(v) for k, v in g.items()}


for s in sys.argv[1:] or ["4352", "mix", "small", "lat"]:
    if s == "lat":
        d = {n: lat(f"{l}_lat") for n, l in COLS}
        print("\nlatency (ns)" + " " * 36 + "".join(f"{n:>9s}" for n, _ in COLS))
        for k in d["v4"]:
            print(f"  {k[:46]:46s}" + "".join(f"{d[n].get(k, float('nan')):9.1f}" for n, _ in COLS))
        continue
    d = {n: bw(f"{l}_{s}") for n, l in COLS}
    print(f"\n{s}" + " " * (44 - len(s)) + "".join(f"{n:>9s}" for n, _ in COLS))
    for k in d["v4"]:
        if s == "small" and not k[3]:
            continue
        plain = "Mix" in k[0] or "Pure" in k[0]
        name = k[0] + ("" if plain else f" {k[1]} {k[2]}") + (" r" if k[3] else "")
        print(f"  {name[:42]:42s}" + "".join(f"{d[n].get(k, float('nan')):9.1f}" for n, _ in COLS))
