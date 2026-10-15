#!/usr/bin/env python3
"""Table of a run_s16t.sh log. Arms: M main, S the #58819 head merged with main, S2 A/A copy of S, MX main and SX the head with
the route prep TRISC images NOP padded to equal packed sizes (layout only). Per case and op: the median over runs of the per
run medians, the comparisons, run spreads, every-run relation of S to M and of SX to MX, and the medians of the odd and the
even measured calls (1st, 3rd, ... against 2nd, 4th, ...) pooled over runs. usage: ol_table2.py <log>"""
import re
import statistics
import sys
from collections import defaultdict

runs = defaultdict(lambda: defaultdict(list))
calls = defaultdict(lambda: defaultdict(lambda: ([], [])))
for l in open(sys.argv[1], errors="ignore"):
    m = re.search(r"RAB OL (\S+) (\d+) (\S+) (\S+) (\d+) (\d+) (.*)$", l)
    if m:
        arm, r, case, op, n, med, vals = m.groups()
        runs[(case, op)][arm].append(float(med))
        v = [float(x) for x in vals.split()]
        calls[(case, op)][arm][0].extend(v[0::2])
        calls[(case, op)][arm][1].extend(v[1::2])
sp = lambda x: 100 * (max(x) - min(x)) / statistics.median(x) if len(x) > 1 else float("nan")
pc = lambda d, a, b: f"{100 * (d[a] / d[b] - 1):+.2f} %" if a in d and b in d else "-"


def rel(d, a, b):
    if a not in d or b not in d:
        return "-"
    return "below" if max(d[a]) < min(d[b]) else ("above" if min(d[a]) > max(d[b]) else "overlap")


arms = ("M", "S", "S2", "MX", "SX")
print("| case | op | " + " | ".join(f"{a} ns" for a in arms) + " | S vs M | S2 vs S | MX vs M | SX vs MX | SX vs M | "
      "spread M / S / S2 / MX / SX | S to M | SX to MX |")
print("|---" * 15 + "|")
for (case, op), d in sorted(runs.items()):
    md = {a: statistics.median(v) for a, v in d.items()}
    cols = " | ".join(f"{md[a]:.0f}" if a in md else "-" for a in arms)
    spr = " / ".join(f"{sp(d[a]):.2f}" if a in d else "-" for a in arms)
    print(f"| {case} | {op} | {cols} | {pc(md, 'S', 'M')} | {pc(md, 'S2', 'S')} | {pc(md, 'MX', 'M')} | "
          f"{pc(md, 'SX', 'MX')} | {pc(md, 'SX', 'M')} | {spr} % | {rel(d, 'S', 'M')} | {rel(d, 'SX', 'MX')} |")
print()
print("| case | op | " + " | ".join(f"{a} odd / even" for a in arms) + " |")
print("|---" * 7 + "|")
for (case, op), c in sorted(calls.items()):
    cols = " | ".join(f"{statistics.median(c[a][0]):.0f} / {statistics.median(c[a][1]):.0f}" if a in c and c[a][1] else "-"
                      for a in arms)
    print(f"| {case} | {op} | {cols} |")
