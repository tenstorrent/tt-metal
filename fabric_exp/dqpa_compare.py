#!/usr/bin/env python3
"""§56: E1 local-delivery header-load variants (dqpa_<variant>_13129088_*) against the user's file (base), with v1
(dqib) and the E1 bit-27/28 flag build (dqx_415782272) for reference. Cycles/packet; every row that moves >= 1 against
base (MixCmd excluded: bimodal). Latency: median net round trip, ns."""
import csv, os, re, statistics, sys

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
V = [
    ("v1", "dqib_13129088"),
    ("flags", "dqx_415782272"),
    ("base", "dqpa_base_13129088"),
    ("A", "dqpa_A_13129088"),
    ("B", "dqpa_B_13129088"),
]


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


for s in ("4352", "mix"):
    d = {n: bw(f"{p}_{s}") for n, p in V}
    d = {n: v for n, v in d.items() if v}
    keys = [k for k in d["base"] if all(k in d[n] for n in d) and "MixCmd" not in k[0]]
    print(
        f"\n{s}: {len(keys)} rows; mean vs base: "
        + ", ".join(f"{n} {sum(d[n][k] - d['base'][k] for k in keys) / len(keys):+.2f}" for n in d if n != "base")
    )
    print(f"   {'row':36s}" + "".join(f"{n:>8s}" for n in d))
    for k in keys:
        if max(abs(d[n][k] - d["base"][k]) for n in d if n in ("A", "B")) >= 1:
            name = k[0] + ("" if ("Mix" in k[0] or "Pure" in k[0]) else f" {k[1][:7]} {k[2]}")
            print(f"   {name[:36]:36s}" + "".join(f"{d[n][k]:8.1f}" for n in d))
L = {n: lat(f"{p}_lat") for n, p in V}
L = {n: v for n, v in L.items() if v}
print("\nlatency ns" + " " * 28 + "".join(f"{n:>8s}" for n in L))
for k in L["base"]:
    print(
        f"   {k.replace('_ntype_unicast_write_size', '').replace('_ntype_fused_atomic_inc_size', '_fused'):36s}"
        + "".join(f"{L[n][k]:8.0f}" for n in L)
    )
