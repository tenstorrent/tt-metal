#!/usr/bin/env python3
"""Table of a run_s19_ab.sh log: per case and op code (the pool, upsample and grid sample ops; others with --all) the median
over the measured repetitions (k 1 and up) of each run, the median of the per-run medians per arm (M main, S #60646 head,
S2 A/A copy of S), S against M, S2 against S, the run spreads and whether every S run is below or above every M run.
usage: pool_table.py <log> [--all]"""
import re, statistics, sys
from collections import defaultdict
allops = "--all" in sys.argv
runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"RAB OPC (\S+) (\d+) (\S+) k(\d+) (\S+) (\d+) (\d+)", l)
    if m and int(m.group(4)) > 0:
        arm, r, case, k, op, ns, n = m.groups()
        runs[(case, op)][arm][r].append(float(ns))
sp = lambda x: 100 * (max(x) - min(x)) / statistics.median(x)
print("| case | op | M ns | S ns | S2 ns | S vs M | S2 vs S | spread M / S / S2 | every run |")
print("|---|---|---|---|---|---|---|---|---|")
for (case, op), d in sorted(runs.items()):
    if not allops and not re.search(r"pool|upsample|bilinear|grid", op, re.I):
        continue
    pm = {a: [statistics.median(v) for v in d[a].values()] for a in d}
    if not all(pm.get(a) for a in ("M", "S", "S2")):
        continue
    md = {a: statistics.median(pm[a]) for a in pm}
    ev = "below" if max(pm["S"]) < min(pm["M"]) else ("above" if min(pm["S"]) > max(pm["M"]) else "overlap")
    print(f"| {case} | {op} | {md['M']:.0f} | {md['S']:.0f} | {md['S2']:.0f} | {100 * (md['S'] / md['M'] - 1):+.2f} % | "
          f"{100 * (md['S2'] / md['S'] - 1):+.2f} % | {sp(pm['M']):.2f} / {sp(pm['S']):.2f} / {sp(pm['S2']):.2f} % | {ev} |")
