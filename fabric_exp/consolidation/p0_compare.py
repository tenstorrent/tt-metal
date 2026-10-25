#!/usr/bin/env python3
"""Phase 0: how far s79m's numbers move when the router's work does not change (p0a_runs.sh, p0b_runs.sh).

Builds: s79m at -O3 (§79's three runs, job A's three and job C's first; s79mc compiles to the same code), -O2, -Os,
-O3 with a benign source edit (reorder, field), and -O3 plus one compiler flag (p0c_runs.sh, injected ELFs:
-fira-algorithm=priority, -fno-schedule-insns2, -fno-gcse, -fno-tree-loop-im). Per suite (2D 4352, mix,
flushed-signal mixes, 1D 4352; MixCmd excluded): the mean and the largest change vs -O3, and the -O3 run-to-run
spread as the noise floor; then every row.
Cycles/packet = 1.35e9 / packets per second."""
import csv, os, statistics

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
V = {
    "O3": ["dq79m", "dq79mr2", "dq79m3", "p0o3", "p0o3b", "p0o3c", "p0inj"],
    "O2": ["p0o2"],
    "Os": ["p0os"],
    "reorder": ["p0reorder"],
    "field": ["p0field"],
    "ira-prio": ["p0f_ira-algorithm_priority"],
    "no-sched2": ["p0f_no-schedule-insns2"],
    "no-gcse": ["p0f_no-gcse"],
    "no-loopim": ["p0f_no-tree-loop-im"],
}
BASE = "O3"


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


def runs(v, suite):
    return [d for d in (bw(f"{p}_13129088_{suite}") for p in V[v]) if d]


for suite in ("4352", "mix", "flushmix", "flushmix4", "1d"):
    d = {v: rs for v in V if (rs := runs(v, suite))}
    if BASE not in d:
        continue
    keys = [k for k in d[BASE][0] if "MixCmd" not in k[0] and all(k in r for v in d for r in d[v])]
    avg = {v: {k: statistics.mean(r[k] for r in d[v]) for k in keys} for v in d}
    noise = {k: max(r[k] for r in d[BASE]) - min(r[k] for r in d[BASE]) for k in keys}
    print(f"{suite}: {len(keys)} rows; -O3 run-to-run spread ({len(d[BASE])} runs) median "
          f"{statistics.median(noise.values()):.2f}, max {max(noise.values()):.2f}")
    for v in d:
        if v == BASE:
            continue
        dl = {k: avg[v][k] - avg[BASE][k] for k in keys}
        kmax = max(dl, key=lambda k: abs(dl[k]))
        print(f"   {v:8s} mean {statistics.mean(dl.values()):+6.2f}   largest {dl[kmax]:+6.2f} ({kmax[0]} {kmax[1][:7]})")
    print(f"   {'row':34s}" + "".join(f"{v[:8]:>9s}" for v in d) + "   O3 spread")
    for k in keys:
        name = k[0] + ("" if suite not in ("4352", "1d") else f" {k[1][:7]}")
        print(f"   {name[:34]:34s}" + "".join(f"{avg[v][k]:9.1f}" for v in d) + f"   {noise[k]:5.1f}")
    print()
