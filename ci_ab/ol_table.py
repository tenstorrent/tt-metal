#!/usr/bin/env python3
"""Table of a run_s16.sh log: per case and op the median over rounds of the per-run medians for each arm (M main, S the
#58819 restore head merged with main, S2 A/A copy of S, N S with conv3d's reduction tail per tile), S against M, S2
against S, N against S, the run spread of each arm and whether every S run is below every M run (below) or above (above).
usage: ol_table.py <log>"""
import re, statistics, sys
from collections import defaultdict
runs = defaultdict(lambda: defaultdict(list))
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"RAB OL (\S+) (\d+) (\S+) (\S+) (\d+) (\d+) ", l)
    if m:
        arm, r, case, op, n, med = m.groups()[:6]
        runs[(case, op)][arm].append(float(med))
sp = lambda x: 100 * (max(x) - min(x)) / statistics.median(x)
pc = lambda a, b: f"{100 * (a / b - 1):+.2f} %"
print("| case | op | M ns | S ns | S vs M | S2 vs S | N vs S | spread M / S / S2 | every run |")
print("|---|---|---|---|---|---|---|---|---|")
for (case, op), d in sorted(runs.items()):
    md = {a: statistics.median(v) for a, v in d.items()}
    if "S" not in md:
        continue
    cmpm = pc(md["S"], md["M"]) if "M" in md else "-"
    s2 = pc(md["S2"], md["S"]) if "S2" in md else "-"
    nn = pc(md["N"], md["S"]) if "N" in md else "-"
    spr = " / ".join(f"{sp(d[a]):.2f}" if a in d and len(d[a]) > 1 else "-" for a in ("M", "S", "S2"))
    ev = "-"
    if "M" in d:
        ev = "below" if max(d["S"]) < min(d["M"]) else ("above" if min(d["S"]) > max(d["M"]) else "overlap")
    print(f"| {case} | {op} | {md.get('M', float('nan')):.0f} | {md['S']:.0f} | {cmpm} | {s2} | {nn} | {spr} % | {ev} |")
