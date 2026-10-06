"""python prof2_total.py <layers.json from prof2_sum.py --json> -- 40-layer per-chunk totals (ms, mean over devices and max over devices) by category, from the profiled layers
0 (window), 1 (window+Engram), 2 (ratio-2 index source), 3 (ratio-2 reader), 14 (Engram), 20 / 24 (ratio-1 index sources), 21 (ratio-1 reader)."""
import json
import sys
from collections import defaultdict

d = json.load(open(sys.argv[1]))
# layer -> (profiled layer used, Engram layer (its Engram categories are added once, the base uses the non-Engram twin))
plan = {0: "0", 1: "0", 20: "20"}
for i in (2, 8, 14):
    plan[i] = "2"
for i in list(range(3, 8)) + list(range(9, 14)) + list(range(15, 20)):
    plan[i] = "3"
for i in (24, 28, 32, 36):
    plan[i] = "24"
for i in range(21, 40):
    if i not in plan:
        plan[i] = "21"
tot = defaultdict(lambda: [0.0, 0.0])
eng = defaultdict(lambda: [0.0, 0.0])
for lay, src in plan.items():
    for k, v in d[src].items():
        tot[k][0] += v[0]
        tot[k][1] += v[1]
for k, v in d[
    "1"
].items():  # Engram layer 1 minus its window twin (layer 0) = the Engram share, twice (layers 1 and 14)
    base = d["0"].get(k, [0, 0, 0])
    dv = [v[0] - base[0], v[1] - base[1]]
    if abs(dv[0]) > 0.01:
        tot["ENGRAM: " + k][0] += 2 * dv[0]
        tot["ENGRAM: " + k][1] += 2 * dv[1]
T0 = sum(v[0] for v in tot.values())
T1 = sum(v[1] for v in tot.values())
print(f"TOTAL 40 layers per chunk: mean {T0:.0f} ms, max-over-devices {T1:.0f} ms")
for k, v in sorted(tot.items(), key=lambda x: -x[1][0])[:30]:
    print(f"  {v[0]:8.1f} ms ({100 * v[0] / T0:4.1f}%) max {v[1]:8.1f}  {k}")
