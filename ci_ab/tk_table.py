#!/usr/bin/env python3
"""Table of a run_tk_ab.sh log: per case the median of each run's calls per arm (A main, B #60287 head, B2 A/A copy of B),
the medians of the per-run medians, B against A, B2 against B, and the B runs below every A run. usage: tk_table.py <log>"""
import re, statistics, sys
from collections import defaultdict
runs = defaultdict(lambda: defaultdict(list)); cores = {}; order = []
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"OPS OP (\S+) (\d+) (\S+) (\S*) ([\d,]+)\s*$", line)
    if not m:
        continue
    arm, r, case, c, vals = m.groups()
    runs[case][arm].append(statistics.median(float(x) for x in vals.split(",") if x)); cores[case] = c
    if case not in order:
        order.append(case)
print("| case | cores | A us | B us | B vs A | B2 vs B | spread A / B / B2 | B runs below all A / above all A |")
print("|---|---|---|---|---|---|---|---|")
sp = lambda x: 100 * (max(x) - min(x)) / statistics.median(x)
for case in order:
    d = runs[case]
    if not all(d.get(a) for a in ("A", "B", "B2")):
        continue
    med = {a: statistics.median(d[a]) for a in d}
    below = sum(1 for x in d["B"] if all(x < y for y in d["A"])); above = sum(1 for x in d["B"] if all(x > y for y in d["A"]))
    print(f"| {case} | {cores[case]} | {med['A'] / 1e3:.2f} | {med['B'] / 1e3:.2f} | {100 * (med['B'] - med['A']) / med['A']:+.2f} % | "
          f"{100 * (med['B2'] - med['B']) / med['B']:+.2f} % | {sp(d['A']):.2f} / {sp(d['B']):.2f} / {sp(d['B2']):.2f} % | {below} / {above} of {len(d['B'])} |")
