#!/usr/bin/env python3
"""§57: E1 bits 26 (variant B) and 28 (partial-batch hold) on the user's v2 with variant A folded in (dqhd_<flags>_*).
Reference columns: v1 (dqib) and the §56 file before variant A was folded in (dqpa_base). Cycles/packet; rows moving
>= 1 against the new base (13129088) are listed (MixCmd excluded). Latency: median net round trip (ns). Completion
frames: the receiving E1's TX-queue transfers (one per completion frame) for 48,000 packets, small-write runs with TT_FABRIC_TXQ_COUNTERS."""
import csv, os, re, statistics

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
V = [
    ("v1", "dqib_13129088"),
    ("§56base", "dqpa_base_13129088"),
    ("A", "dqhd_13129088"),
    ("B", "dqhd_80237952"),
    ("A+hold", "dqhd_281564544"),
    ("B+hold", "dqhd_348673408"),
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
    keys = [k for k in d["A"] if all(k in d[n] for n in d) and "MixCmd" not in k[0]]
    print(
        f"\n{s}: {len(keys)} rows; mean vs A (new base): "
        + ", ".join(f"{n} {sum(d[n][k] - d['A'][k] for k in keys) / len(keys):+.2f}" for n in d if n != "A")
    )
    print(f"   {'row':34s}" + "".join(f"{n:>9s}" for n in d))
    for k in keys:
        if max(abs(d[n][k] - d["A"][k]) for n in d if n in ("B", "A+hold", "B+hold", "§56base")) >= 1:
            name = k[0] + ("" if ("Mix" in k[0] or "Pure" in k[0]) else f" {k[1][:7]} {k[2]}")
            print(f"   {name[:34]:34s}" + "".join(f"{d[n][k]:9.1f}" for n in d))
L = {n: lat(f"{p}_lat") for n, p in V}
L = {n: v for n, v in L.items() if v}
print("\nlatency ns" + " " * 26 + "".join(f"{n:>9s}" for n in L))
for k in L["A"]:
    print(
        f"   {k.replace('_ntype_unicast_write_size', '').replace('_ntype_fused_atomic_inc_size', '_fused'):34s}"
        + "".join(f"{L[n][k]:9.0f}" for n in L)
    )
print("\nsmall writes with TX-queue counters: rate, and packets per completion frame at the busiest receiving E1")
for n, p in V[2:]:
    f = os.path.join(R, f"{p}_txq_phase.csv")
    if not os.path.exists(f):
        continue
    rates = {
        r["test_name"]: 1.35e9 / float(r["avg_packets_per_s"])
        for r in csv.DictReader(open(os.path.join(R, f"{p}_txq.csv")))
    }
    best = {}
    for r in csv.DictReader(open(f)):
        if r["erisc"] != "1" or not r["txq1_transfers"]:
            continue
        t = r["test"].split("_size")[0] if "_size" in r["test"] else r["test"]
        tr = int(r["txq1_transfers"])
        best[t] = max(best.get(t, 0), tr)
    print(
        f"   {n:8s} "
        + "  ".join(
            f"{t}: {rates.get(t, 0):6.1f} cyc, {48000 / best[t]:4.2f} pkt/frame" for t in sorted(best) if t in rates
        )
    )
