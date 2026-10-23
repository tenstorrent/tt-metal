#!/usr/bin/env python3
"""§62: flush-only staged atomic via noc_semaphore_inc_set_state / _with_state (dqsf_13129088_*) against the code before
any atomic change (dqhd_348673408_*), §61's always-staged version (dqst_13129088_*), its loads-first variant
(dqstlf_13129088_*) and v1 (dqib). Fused write+atomic with flush off/on: dq{fl,st_fl,stlf_fl,sf_fl}*. Cycles/packet;
rows moving >= 1 against before are listed (MixCmd excluded). Latency: median net round trip (ns)."""
import csv, os, re, statistics

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
V = [
    ("v1", "dqib_13129088"),
    ("before", "dqhd_348673408"),
    ("§61", "dqst_13129088"),
    ("§61 lf", "dqstlf_13129088"),
    ("§62", "dqsf_13129088"),
]
FL = [
    ("before", "dqfl_{}_348673408"),
    ("§61", "dqst_fl{}_13129088"),
    ("§61 lf", "dqstlf_fl{}_13129088"),
    ("§62", "dqsf_fl{}_13129088"),
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
    keys = [k for k in d["before"] if all(k in d[n] for n in d) and "MixCmd" not in k[0]]
    print(
        f"\n{s}: {len(keys)} rows; mean vs before: "
        + ", ".join(f"{n} {sum(d[n][k]-d['before'][k] for k in keys)/len(keys):+.2f}" for n in d if n != "before")
    )
    print(f"   {'row':30s}" + "".join(f"{n:>9s}" for n in d))
    for k in keys:
        if max(abs(d[n][k] - d["before"][k]) for n in d if n not in ("v1",)) >= 1:
            name = k[0] + ("" if ("Mix" in k[0] or "Pure" in k[0]) else f" {k[1][:7]} {k[2]}")
            print(f"   {name[:30]:30s}" + "".join(f"{d[n][k]:9.1f}" for n in d))
L = {n: lat(f"{p}_lat") for n, p in V}
L = {n: v for n, v in L.items() if v}
print("\nlatency ns" + " " * 20 + "".join(f"{n:>9s}" for n in L))
for k in L["before"]:
    print(
        f"   {k.replace('_ntype_unicast_write_size', '').replace('_ntype_fused_atomic_inc_size', '_fused'):28s}"
        + "".join(f"{L[n][k]:9.0f}" for n in L)
    )
print("\nfused write+atomic (cycles/packet)")
for fl in ("off", "on"):
    d = {n: bw(p.format(fl)) for n, p in FL}
    d = {n: v for n, v in d.items() if v}
    for k in d["before"]:
        if "Fused" in k[0]:
            print(f"   flush {fl:3s} {k[0]:20s}" + "".join(f"  {n} {d[n][k]:6.1f}" for n in d))
