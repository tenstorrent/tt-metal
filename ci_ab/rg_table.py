#!/usr/bin/env python3
"""Table of a run_rg_ab.sh log: per case and op code (the pool, upsample, grid sample and untilize ops; every op with --all) the
median over the measured repetitions (k 1 and up) of each run, the median of the per-run medians per arm (M main, S #60646
head, X the MOP word fix, X2 A/A copy of X), X against M and S, S against M, X2 against X, the run spreads, and whether every
X run is below or above every M run.
usage: rg_table.py <log> [--all]"""
import re
import statistics
import sys
from collections import defaultdict

allops = "--all" in sys.argv
runs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"RAB OPC (\S+) (\d+) (\S+) k(\d+) (\S+) (\d+) (\d+)", l)
    if m and int(m.group(4)) > 0:
        arm, r, case, k, op, ns, n = m.groups()
        runs[(case, op)][arm][r].append(float(ns))
sp = lambda x: 100 * (max(x) - min(x)) / statistics.median(x)
pc = lambda a, b: f"{100 * (a / b - 1):+.2f} %"
print("| case | op | M ns | S ns | X ns | X2 ns | X vs M | X vs S | S vs M | X2 vs X | spread M / S / X / X2 | X runs vs M runs |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for (case, op), d in sorted(runs.items()):
    if not allops and not re.search(r"pool|upsample|bilinear|grid|untilize", op, re.I):
        continue
    pm = {a: [statistics.median(v) for v in d[a].values()] for a in d}
    if not all(pm.get(a) for a in ("M", "S", "X", "X2")):
        continue
    md = {a: statistics.median(pm[a]) for a in pm}
    ev = "below" if max(pm["X"]) < min(pm["M"]) else ("above" if min(pm["X"]) > max(pm["M"]) else "overlap")
    print(f"| {case} | {op} | {md['M']:.0f} | {md['S']:.0f} | {md['X']:.0f} | {md['X2']:.0f} | {pc(md['X'], md['M'])} | "
          f"{pc(md['X'], md['S'])} | {pc(md['S'], md['M'])} | {pc(md['X2'], md['X'])} | "
          f"{sp(pm['M']):.2f} / {sp(pm['S']):.2f} / {sp(pm['X']):.2f} / {sp(pm['X2']):.2f} % | {ev} |")
