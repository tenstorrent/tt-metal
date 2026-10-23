#!/usr/bin/env python3
"""§66 A/B on bh-lb-120-a08u24: §63 rebuilt from the transcript (a8s63_13129088_*) against §66 (dq66_*, repeated as
dq66r2_*), with a04u24's before (dqhd_348673408), §63 (dqpd) and §65 (dqps) for reference, on 4352 / mix / latency and the
flushed-signal yaml (flush cost per signal). Latency also gets the v1 host calibration (a8v1 vs dqib).
Cycles/packet; rows moving >= 1 against before are listed (MixCmd excluded)."""
import csv, os, re, statistics

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
V = [
    ("v1", "dqib_13129088"),
    ("v1 a08", "a8v1_13129088"),
    ("before", "dqhd_348673408"),
    ("§63 a04", "dqpd_13129088"),
    ("§65 a04", "dqps_13129088"),
    ("§63r", "a8s63_13129088"),
    ("§66", "dq66_13129088"),
    ("§66 r2", "dq66r2_13129088"),
]


def bw(label, keyed=True):
    f = os.path.join(R, label + ".csv")
    if not os.path.exists(f):
        return None
    out = {}
    for r in csv.DictReader(open(f)):
        if not keyed:
            out[r["test_name"]] = 1.35e9 / float(r["avg_packets_per_s"])
            continue
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


print("host calibration, v1 at 13129088 (a08u24 - a04u24), MixCmd excluded")
for s in ("4352", "mix"):
    here = bw(f"a8v1_13129088_{s}")
    for ref in ("dqib_13129088", "dqv1b_13129088"):
        old = bw(f"{ref}_{s}")
        if not (here and old):
            continue
        ks = [k for k in old if k in here and "MixCmd" not in k[0]]
        d = [here[k] - old[k] for k in ks]
        print(f"   {s:5s} vs {ref:15s} {len(ks):3d} rows  mean {sum(d)/len(d):+.2f}  max |d| {max(map(abs, d)):.2f}")
for s in ("4352", "mix"):
    d = {n: bw(f"{p}_{s}") for n, p in V}
    d = {n: v for n, v in d.items() if v}
    keys = [k for k in d["before"] if all(k in d[n] for n in d) and "MixCmd" not in k[0]]
    print(
        f"\n{s}: mean vs before: "
        + ", ".join(f"{n} {sum(d[n][k]-d['before'][k] for k in keys)/len(keys):+.2f}" for n in d if n != "before")
    )
    print(f"   {'row':30s}" + "".join(f"{n:>8s}" for n in d))
    for k in keys:
        if max(abs(d[n][k] - d["before"][k]) for n in d if not n.startswith("v1")) >= 1:
            name = k[0] + ("" if ("Mix" in k[0] or "Pure" in k[0]) else f" {k[1][:7]} {k[2]}")
            print(f"   {name[:30]:30s}" + "".join(f"{d[n][k]:8.1f}" for n in d))
    print(
        f"   §66 - §63r mean (both runs): "
        + ", ".join(
            f"{r} {sum(d[r][k]-d['§63r'][k] for k in keys)/len(keys):+.2f}"
            for r in ("§66", "§66 r2")
            if r in d and "§63r" in d
        )
    )
L = {n: lat(f"{p}_lat") for n, p in V}
L = {n: v for n, v in L.items() if v}
print("\nlatency ns" + " " * 20 + "".join(f"{n:>8s}" for n in L))
for k in L["before"]:
    print(
        f"   {k.replace('_ntype_unicast_write_size', '').replace('_ntype_fused_atomic_inc_size', '_fused'):28s}"
        + "".join(f"{L[n][k]:8.0f}" for n in L)
    )
F = [
    ("before", "dqfm_before"),
    ("§63 a04", "dqfm_s63"),
    ("§65 a04", "dqps_13129088_flushmix"),
    ("§63r", "a8s63_13129088_flushmix"),
    ("§66", "dq66_13129088_flushmix"),
    ("§66 r2", "dq66r2_13129088_flushmix"),
]
fm = {n: bw(p, keyed=False) for n, p in F}
fm = {n: v for n, v in fm.items() if v}
print("\nflushed-signal yaml (cycles/packet)")
print(f"   {'row':26s}" + "".join(f"{n:>9s}" for n in fm))
for t in fm["before"]:
    print(f"   {t:26s}" + "".join(f"{fm[n][t]:9.1f}" for n in fm))
CH = {"Fused": 1, "Chunk8Fused": 8, "Chunk16Fused": 16, "Chunk8Atomic": 8}
print("\ncost per flushed signal (flushed - unflushed) x chunk")
print(f"   {'':17s}" + "".join(f"{n:>9s}" for n in fm))
for dr in ("EW", "NS"):
    for sh, k in CH.items():
        print(
            f"   {dr}_{sh:14s}"
            + "".join(f"{(fm[n][f'{dr}_{sh}_Flush'] - fm[n][f'{dr}_{sh}_NoFlush']) * k:9.1f}" for n in fm)
        )
