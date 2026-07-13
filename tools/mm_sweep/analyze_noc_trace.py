#!/usr/bin/env python3
# Direct analysis of a tt-metal noc_trace JSON: DRAM-endpoint balance + NOC_0 link-load
# hotspots (approx XY-torus routing; treat link numbers as indicative, not exact).
import json, sys
from collections import Counter

d = json.load(open(sys.argv[1]))
reads = [e for e in d if e.get("type") == "READ"]
print(f"READ events {len(reads)}  bytes/read {set(e['num_bytes'] for e in reads)}  noc {set(e['noc'] for e in reads)}")
dst = Counter((e["dx"], e["dy"]) for e in reads)
print(f"DRAM endpoints {len(dst)}  reads/endpoint min {min(dst.values())} max {max(dst.values())} (balanced if equal)")
src = Counter((e["sx"], e["sy"]) for e in reads)
print(f"src cores {len(src)}  reads/core min {min(src.values())} max {max(src.values())}")
X = max(max(e["sx"], e["dx"]) for e in reads) + 1
Y = max(max(e["sy"], e["dy"]) for e in reads) + 1
link = Counter()
for e in reads:
    x, y, b = e["sx"], e["sy"], e["num_bytes"]
    while x != e["dx"]:
        link[("H", x, y)] += b
        x = (x + 1) % X
    while y != e["dy"]:
        link[("V", x, y)] += b
        y = (y + 1) % Y
tot = sum(link.values())
mean = tot / len(link)
mx = max(link.values())
print(f"links used {len(link)}  max/mean {mx/mean:.2f}")
for k, v in sorted(link.items(), key=lambda kv: -kv[1])[:8]:
    print(f"   hot {k} {v/1e6:.2f} MB")
