#!/usr/bin/env python3
"""Aggregate zone stats across multiple profiler CSVs (TRISC_2 only)."""
import csv
import sys
from collections import Counter, defaultdict
from statistics import mean, pstdev

zones_of_interest = {"reduce_max", "reduce_sum"}
all_durations = defaultdict(list)

for path in sys.argv[1:]:
    with open(path) as f:
        f.readline()
        reader = csv.DictReader(f, skipinitialspace=True)
        opens = {}
        for row in reader:
            zname = row["zone name"].strip()
            if zname not in zones_of_interest:
                continue
            if row["RISC processor type"].strip() != "TRISC_2":
                continue
            ztype = row["type"].strip()
            key = (row["core_x"].strip(), row["core_y"].strip(), zname, path)
            t = int(row["time[cycles since reset]"].strip())
            if ztype == "ZONE_START":
                opens.setdefault(key, []).append(t)
            elif ztype == "ZONE_END":
                s = opens.get(key, [])
                if s:
                    all_durations[zname].append(t - s.pop(0))

freq = 1350.0
ns = lambda c: c / freq * 1000.0
buckets = [(150, 170), (170, 200), (200, 250), (250, 300), (300, 400), (400, 500), (500, 800)]

for z in ("reduce_max", "reduce_sum"):
    vs = sorted(all_durations[z])
    n = len(vs)
    if n == 0:
        print(f"{z}: no data")
        continue
    mn, mx = vs[0], vs[-1]
    avg = mean(vs)
    sd = pstdev(vs)
    med = vs[n // 2]
    p90 = vs[int(0.90 * (n - 1))]
    p99 = vs[int(0.99 * (n - 1))]
    print(f"\n{z}  TRISC_2  n={n}")
    print(f"  min={mn} ({ns(mn):.1f}ns)  med={med} ({ns(med):.1f}ns)  mean={avg:.1f} ({ns(avg):.1f}ns)  std={sd:.1f}")
    print(f"  p90={p90} ({ns(p90):.1f}ns)  p99={p99} ({ns(p99):.1f}ns)  max={mx} ({ns(mx):.1f}ns)")
    counts = Counter()
    for v in vs:
        for lo, hi in buckets:
            if lo <= v < hi:
                counts[(lo, hi)] += 1
                break
        else:
            counts[("other", "")] = counts.get(("other", ""), 0) + 1
    maxc = max(counts.values()) if counts else 1
    for lo, hi in buckets:
        c = counts.get((lo, hi), 0)
        bar = "#" * int(40 * c / maxc) if maxc else ""
        pct = 100.0 * c / n
        print(f"  [{lo:4d}..{hi:4d})  {c:5d} ({pct:5.1f}%)  {bar}")
