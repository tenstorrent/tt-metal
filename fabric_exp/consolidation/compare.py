#!/usr/bin/env python3
"""compare.py [--rows] NAME=label[,label...] ...: cycles/packet of builds against the first one (the base).

Labels are result prefixes (fabric_exp/results/<label>_13129088_<suite>.csv); several labels per build are averaged.
`s79m` expands to the seven -O3 runs of s79m/s79mc (§79 + Phase 0). Per suite (2D 4352, mix, flushed-signal mixes,
1D 4352; MixCmd excluded): the mean and largest change vs the base, and the base's run-to-run spread; then the key
rows of RESULTS §82, and every row with --rows. Cycles/packet = 1.35e9 / packets per second.
"""
import csv, os, statistics, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ALIAS = {"s79m": ["dq79m", "dq79mr2", "dq79m3", "p0o3", "p0o3b", "p0o3c", "p0inj"]}
KEY = [("mix", "EW_PureWrite16", None), ("mix", "NS_PureWrite16", None), ("mix", "FWD_PureWrite16", None),
       ("mix", "EW_MixSize4", None), ("4352", "Mesh1HopEW", "write"), ("4352", "Mesh3HopEW", "write"),
       ("4352", "MeshMulticast", "write"), ("mix", "EW_PureScatter4352", None), ("mix", "EW_PureFused4352", None),
       ("flushmix", "EW_Chunk8Fused_Flush", None), ("1d", "Linear3HopEW", "write"), ("1d", "NeighborExchange", "write")]


def bw(label, suite):
    f = os.path.join(R, f"{label}_13129088_{suite}.csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        k = (r["test_name"], r["ntype"].replace("NOC_UNICAST_", "").lower(), r.get("packet_size", ""))
        n = sum(1 for kk in out if kk[:3] == k)
        out[k + (n,)] = 1.35e9 / float(r["avg_packets_per_s"])
    return out


rows = "--rows" in sys.argv
B = {}
for a in sys.argv[1:]:
    if a.startswith("--"):
        continue
    name, labels = a.split("=", 1)
    B[name] = [x for l in labels.split(",") for x in ALIAS.get(l, [l])]
base = next(iter(B))
key_vals = {}
for suite in ("4352", "mix", "flushmix", "flushmix4", "1d"):
    d = {v: rs for v in B if (rs := [r for r in (bw(l, suite) for l in B[v]) if r])}
    if base not in d:
        continue
    keys = [k for k in d[base][0] if "MixCmd" not in k[0] and all(k in r for v in d for r in d[v])]
    avg = {v: {k: statistics.mean(r[k] for r in d[v]) for k in keys} for v in d}
    sp = {k: max(r[k] for r in d[base]) - min(r[k] for r in d[base]) for k in keys}
    note = f"; {base} spread ({len(d[base])} runs) median {statistics.median(sp.values()):.2f}" if len(d[base]) > 1 else ""
    print(f"{suite}: {len(keys)} rows{note}")
    for v in d:
        if v != base:
            dl = {k: avg[v][k] - avg[base][k] for k in keys}
            km = max(dl, key=lambda k: abs(dl[k]))
            print(f"   {v:14s} mean {statistics.mean(dl.values()):+6.2f}   largest {dl[km]:+6.2f} ({km[0]} {km[1][:7]})")
    for s, name, nt in KEY:
        if s == suite:
            k = next((k for k in keys if k[0] == name and (nt is None or k[1] == nt)), None)
            if k:
                key_vals[(name, nt)] = {v: avg[v][k] for v in d}
    if rows:
        print(f"   {'row':34s}" + "".join(f"{v[:9]:>10s}" for v in d))
        for k in keys:
            nm = k[0] + ("" if suite not in ("4352", "1d") else f" {k[1][:7]}")
            print(f"   {nm[:34]:34s}" + "".join(f"{avg[v][k]:10.1f}" for v in d))
print(f"\n{'key row':30s}" + "".join(f"{v[:9]:>10s}" for v in B))
for (name, nt), vals in key_vals.items():
    print(f"{(name + (' ' + nt if nt else ''))[:30]:30s}" + "".join(f"{vals.get(v, float('nan')):10.1f}" for v in B))
