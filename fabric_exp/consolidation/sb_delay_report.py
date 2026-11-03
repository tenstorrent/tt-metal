#!/usr/bin/env python3
"""sb_delay_report.py TAG: sb_delay_runs.sh results (labels sbd<TAG>_<router>_<row>_<arm>): per row and router,
cycles/packet at each arm and its change against base (a spin of N cycles costs about N + 12 where its side binds)."""
import collections, csv, glob, os, re, sys

R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
tag = sys.argv[1]
data = collections.defaultdict(dict)
arms = []
for f in sorted(glob.glob(f"{R}/sbd{tag}_*.csv")):
    b = os.path.basename(f)
    if b.endswith("_detail.csv") or b.endswith("_latency.csv"):
        continue
    m = re.match(rf"sbd{tag}_(v[^_]+)_(.+)_(base|[a-z]+\d+)\.csv$", b)
    if not m:
        continue
    router, row, arm = m.groups()
    if arm not in arms:
        arms.append(arm)
    for r in csv.DictReader(open(f)):
        data[(row, router)][arm] = 1.35e9 / float(r["avg_packets_per_s"])
order = lambda a: (a != "base", re.sub(r"\d+", "", a), int(re.sub(r"\D", "", a) or 0))
arms.sort(key=order)
print(f"{'row':16} {'router':10}" + "".join(f"{a:>9}" for a in arms))
for (row, router) in sorted(data):
    d = data[(row, router)]
    base = d.get("base")
    cells = []
    for a in arms:
        v = d.get(a)
        cells.append(f"{'-':>9}" if v is None else f"{v:9.1f}" if a == "base" or base is None else f"{v - base:+9.1f}")
    print(f"{row:16} {router:10}" + "".join(cells))
