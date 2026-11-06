# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage device-zone report for nomic_sdpa (Perf 1). Reads generated/profiler/.logs/profile_log_device.csv
(the LAST program run per core) and prints, per (RISC, zone): executions per core, per-core summed duration (median /
max across cores, us) and the median first-start / last-end offset from the kernel start of that RISC (us).
Also: per-RISC kernel span p50 / max (core balance), marker counts (budget: 250 / RISC), zone coverage of the span.

usage: python zone_report.py [profile_log_device.csv] [--mhz 1350] [--run N (-1 = last)]"""
import csv, sys, statistics as st
from collections import defaultdict

path = "generated/profiler/.logs/profile_log_device.csv"
mhz = None
run_sel = -1
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--mhz":
        mhz = float(args[i + 1])
        i += 2
    elif args[i] == "--run":
        run_sel = int(args[i + 1])
        i += 2
    else:
        path = args[i]
        i += 1
with open(path) as f:
    first = f.readline()
    if mhz is None:
        mhz = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.DictReader(f, skipinitialspace=True))
runs = sorted({r["run host ID"] for r in rows}, key=int)
run = runs[run_sel]
rows = [r for r in rows if r["run host ID"] == run]
us = lambda c: c / mhz
stack = defaultdict(list)
dur = defaultdict(lambda: defaultdict(list))  # (risc, zone) -> core -> [(start, end)]
markers = defaultdict(int)
for r in rows:
    key = (r["core_x"], r["core_y"], r["RISC processor type"])
    z = r["zone name"]
    t = int(r["time[cycles since reset]"])
    markers[key] += 1
    if r["type"] == "ZONE_START":
        stack[(key, z)].append(t)
    elif r["type"] == "ZONE_END" and stack[(key, z)]:
        s = stack[(key, z)].pop()
        dur[(key[2], z)][(key[0], key[1])].append((s, t))
kstart = {}
for (risc, z), per in dur.items():
    if z.endswith("-KERNEL"):
        for core, iv in per.items():
            kstart[(core, risc)] = (iv[0][0], iv[0][1])
print(f"run host ID {run}, {mhz:.0f} MHz, {len({k[:2] for k in markers})} cores")
print(
    f"{'risc':7s} {'zone':22s} {'exec':>5s} {'sum p50':>8s} {'sum max':>8s} {'start p50':>9s} {'end p50':>8s} {'end max':>8s}"
)
for risc, z in sorted(
    dur, key=lambda k: (k[0], min(min(s for s, _ in v) - kstart.get((c, k[0]), (0, 0))[0] for c, v in dur[k].items()))
):
    per = dur[(risc, z)]
    if z.endswith("-FW"):
        continue
    sums = [us(sum(e - s for s, e in iv)) for iv in per.values()]
    execs = st.median(len(iv) for iv in per.values())
    starts, ends = [], []
    for core, iv in per.items():
        k0 = kstart.get((core, risc), (iv[0][0], 0))[0]
        starts.append(us(iv[0][0] - k0))
        ends.append(us(iv[-1][1] - k0))
    print(
        f"{risc:7s} {z:22s} {execs:5.0f} {st.median(sums):8.2f} {max(sums):8.2f} {st.median(starts):9.2f} {st.median(ends):8.2f} {max(ends):8.2f}"
    )
mx = defaultdict(int)
for (core, risc), n in ((k[:2], v) for k, v in markers.items()):
    pass
for k, n in markers.items():
    mx[k[2]] = max(mx[k[2]], n)
print("max markers per RISC (cap ~250 user):", dict(mx))
# coverage: last user-zone end vs kernel end
for risc in sorted({k[2] for k in markers}):
    gaps = []
    for (rk, z), per in dur.items():
        pass
    cov = []
    for core in {c for (c, r) in kstart if r == risc}:
        ke = kstart[(core, risc)][1]
        ks = kstart[(core, risc)][0]
        ends = [
            iv[-1][1]
            for (rk, z), per in dur.items()
            if rk == risc and not z.endswith(("-KERNEL", "-FW"))
            for c, iv in per.items()
            if c == core
        ]
        if ends:
            cov.append((max(ends) - ks) / max(1, ke - ks))
    if cov:
        print(f"{risc}: user zones cover min {min(cov)*100:.0f}% / p50 {st.median(cov)*100:.0f}% of the kernel span")
