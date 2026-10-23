#!/usr/bin/env python3
"""§61: staged atomic (dqst_13129088_*: the user's v2 with bits 26/28 hard-wired + staged atomic) against the same code
before the atomic change (dqhd_348673408_*: bits 26 + 28 as flags) and v1 (dqib). Fused write+atomic with flush off/on:
dqst_fl<off|on>_13129088 against §58's dqfl_<off|on>_348673408. Cycles/packet; rows moving >= 1 against before are
listed (MixCmd excluded). Latency: median net round trip (ns)."""
import csv, os, re, statistics

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
V = [("v1", "dqib_13129088"), ("before", "dqhd_348673408"), ("staged", "dqst_13129088")]


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
    if not all(d.values()):
        print(s, "missing")
        continue
    keys = [k for k in d["before"] if all(k in d[n] for n in d) and "MixCmd" not in k[0]]
    print(
        f"\n{s}: {len(keys)} rows; mean vs before: v1 {sum(d['v1'][k]-d['before'][k] for k in keys)/len(keys):+.2f}, staged {sum(d['staged'][k]-d['before'][k] for k in keys)/len(keys):+.2f}"
    )
    for k in keys:
        if abs(d["staged"][k] - d["before"][k]) >= 1:
            name = k[0] + ("" if ("Mix" in k[0] or "Pure" in k[0]) else f" {k[1][:7]} {k[2]}")
            print(
                f"   {name[:36]:36s}"
                + "".join(f"{d[n][k]:9.1f}" for n in d)
                + f"   ({d['staged'][k]-d['before'][k]:+.1f})"
            )
L = {n: lat(f"{p}_lat") for n, p in V}
if all(L.values()):
    print("\nlatency ns" + " " * 28 + "".join(f"{n:>9s}" for n in L))
    for k in L["before"]:
        print(
            f"   {k.replace('_ntype_unicast_write_size', '').replace('_ntype_fused_atomic_inc_size', '_fused'):36s}"
            + "".join(f"{L[n][k]:9.0f}" for n in L)
            + f"   ({L['staged'][k]-L['before'][k]:+.0f})"
        )
print("\nfused write+atomic, flush off / on (cycles/packet):")
for fl in ("off", "on"):
    a, b = bw(f"dqfl_{fl}_348673408"), bw(f"dqst_fl{fl}_13129088")
    if not (a and b):
        continue
    for k in a:
        if k in b:
            print(f"   flush {fl:3s} {k[0]:22s} before {a[k]:7.1f}  staged {b[k]:7.1f}  ({b[k]-a[k]:+.1f})")
