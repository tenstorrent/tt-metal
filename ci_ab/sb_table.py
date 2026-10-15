#!/usr/bin/env python3
"""Table of a run_sb_ab.sh log: per case and op, the per-run medians of each arm (M main, H PR head, H2 A/A copy of H),
the arm means, H against M and H2 against H, and the run spread. usage: sb_table.py <log>"""
import collections, re, statistics, sys
runs = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"RAB OP (\S+) (\d+) (\S+) (\S+) (\d+) ([\d.]+)", l)
    if m:
        arm, r, case, op, n, med = m.groups()
        runs[(case, op)][arm].append(float(med))
print("| case | op | M us | H us | H2 us | H vs M | H2 vs H | spread M / H / H2 |")
print("|---|---|---|---|---|---|---|---|")
sp = lambda x: 100 * (max(x) - min(x)) / statistics.mean(x)
for (case, op), d in sorted(runs.items()):
    if not all(d.get(a) for a in ("M", "H", "H2")):
        continue
    m, h, h2 = (statistics.mean(d[a]) for a in ("M", "H", "H2"))
    print(f"| {case} | {op} | {m / 1e3:.3f} | {h / 1e3:.3f} | {h2 / 1e3:.3f} | {100 * (h / m - 1):+.2f} % | {100 * (h2 / h - 1):+.2f} % | "
          f"{sp(d['M']):.2f} / {sp(d['H']):.2f} / {sp(d['H2']):.2f} % |")
