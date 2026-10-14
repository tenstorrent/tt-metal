#!/usr/bin/env python3
"""Scratch, not for merge. x2p_table.py <ops lines file> <X arm>: per case the median of per-run medians of A, A2, X, X2;
X vs A from the mean of each pair; A/A = A2 vs A, X/X = X2 vs X; a row is slower or faster only beyond the larger A/A."""
import re, statistics, sys
from collections import defaultdict
path, X = sys.argv[1], sys.argv[2]
runs, order = defaultdict(lambda: defaultdict(list)), []
for line in open(path, errors="replace"):
    m = re.search(r"OPS OP (\S+) (\d+) (\S+) (\S*) ([\d,]+)\s*$", line)
    if not m:
        continue
    arm, _, case, _, vals = m.groups()
    runs[case][arm].append(statistics.median(float(v) for v in vals.split(",") if v))
    if case not in order:
        order.append(case)
print(f"| row | A | A2 | {X} | {X}2 | {X} vs A % | A/A % | {X}/{X} % | {X} runs below every A run | verdict |")
print("|---|---|---|---|---|---|---|---|---|---|")
slower, missing = [], []
for case in order:
    d = runs[case]
    if not all(d.get(a) for a in ("A", "A2", X, X + "2")):
        missing.append(case); print(f"| {case} | missing arms: {' '.join(sorted(d))} |"); continue
    a, a2, x, x2 = (statistics.median(d[k]) for k in ("A", "A2", X, X + "2"))
    xa = 100 * ((x + x2) / 2 - (a + a2) / 2) / ((a + a2) / 2)
    aa, xx = 100 * (a2 - a) / a, 100 * (x2 - x) / x
    spread = max(abs(aa), abs(xx))
    verdict = "faster" if xa < -spread else "SLOWER" if xa > spread else "equal"
    if verdict == "SLOWER":
        slower.append(case)
    arm_a = d["A"] + d["A2"]; below = sum(1 for v in d[X] + d[X + "2"] if v < min(arm_a))
    print(f"| {case} | {a:.0f} | {a2:.0f} | {x:.0f} | {x2:.0f} | {xa:+.2f} | {aa:+.2f} | {xx:+.2f} | {below} of {len(d[X]) + len(d[X + '2'])} | {verdict} |")
print(f"X2P PERF rows {len(order)} slower {len(slower)} {' '.join(slower)} missing {len(missing)} {' '.join(missing)}")
