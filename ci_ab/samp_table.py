#!/usr/bin/env python3
"""Table of a run_s163_ab.sh log: per case the median over the measured calls (k 1 and up) of each run, the median of the
per-run medians per arm (M main, S #60163 head, S2 A/A copy of S), S against M, S2 against S and the run spreads.
usage: samp_table.py <log>"""
import re, statistics, sys
from collections import defaultdict
runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"RAB OP (\S+) (\d+) (\S+) k(\d+) (\d+) (\d+)", l)
    if m and int(m.group(4)) > 0:
        arm, r, case, k, ns, n = m.groups()
        runs[case][arm][r].append(float(ns))
print("| case | M ns | S ns | S2 ns | S vs M | S2 vs S | spread M / S / S2 |")
print("|---|---|---|---|---|---|---|")
sp = lambda x: 100 * (max(x) - min(x)) / statistics.median(x)
for case, d in sorted(runs.items()):
    pm = {a: [statistics.median(v) for v in d[a].values()] for a in d}
    if not all(pm.get(a) for a in ("M", "S", "S2")):
        continue
    md = {a: statistics.median(pm[a]) for a in pm}
    print(f"| {case} | {md['M']:.0f} | {md['S']:.0f} | {md['S2']:.0f} | {100 * (md['S'] / md['M'] - 1):+.2f} % | "
          f"{100 * (md['S2'] / md['S'] - 1):+.2f} % | {sp(pm['M']):.2f} / {sp(pm['S']):.2f} / {sp(pm['S2']):.2f} % |")
