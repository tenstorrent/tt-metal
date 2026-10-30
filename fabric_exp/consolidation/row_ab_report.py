#!/usr/bin/env python3
"""row_ab_report.py TAG: cycles/packet (1.35 GHz) per row and router from row_ab_runs.sh, each repetition and the mean."""
import collections, csv, glob, os, re, sys
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
tag = sys.argv[1]
data = collections.defaultdict(lambda: collections.defaultdict(list))
routers = []
for f in sorted(glob.glob(f"{R}/rab{tag}_*.csv")):
    m = re.match(rf"rab{tag}_(v[^_]+)_(.+)_(\d+)\.csv$", os.path.basename(f))
    if not m:
        continue
    router, row, rep = m.groups()
    if router not in routers:
        routers.append(router)
    for r in csv.DictReader(open(f)):
        data[row][router].append(1.35e9 / float(r["avg_packets_per_s"]))
print(f"{'row':12}" + "".join(f"{r:>22}" for r in routers))
for row, per in data.items():
    print(f"{row:12}" + "".join(f"{sum(per[r]) / len(per[r]):9.1f} ({' / '.join(f'{v:.1f}' for v in per[r])})".rjust(22) if per[r] else f"{'-':>22}" for r in routers))
