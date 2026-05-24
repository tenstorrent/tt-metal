#!/usr/bin/env python3
"""Side-by-side histogram comparison of zone durations on TRISC_2 (PACK)."""
import argparse
import csv
from collections import Counter, defaultdict


def load_trisc2(path, zones):
    durations = defaultdict(list)
    with open(path) as f:
        f.readline()
        reader = csv.DictReader(f, skipinitialspace=True)
        opens = {}
        for row in reader:
            zname = row["zone name"].strip()
            if zname not in zones:
                continue
            if row["RISC processor type"].strip() != "TRISC_2":
                continue
            ztype = row["type"].strip()
            key = (row["core_x"].strip(), row["core_y"].strip(), zname)
            t = int(row["time[cycles since reset]"].strip())
            if ztype == "ZONE_START":
                opens.setdefault(key, []).append(t)
            elif ztype == "ZONE_END":
                s = opens.get(key, [])
                if s:
                    durations[zname].append(t - s.pop(0))
    return durations


def histogram(vals, buckets):
    if not vals:
        return ""
    counts = Counter()
    for v in vals:
        for lo, hi in buckets:
            if lo <= v < hi:
                counts[(lo, hi)] += 1
                break
    maxc = max(counts.values()) if counts else 1
    lines = []
    for lo, hi in buckets:
        c = counts.get((lo, hi), 0)
        bar = "#" * int(40 * c / maxc) if maxc else ""
        lines.append(f"  [{lo:4d}..{hi:4d})  {c:5d}  {bar}")
    return "\n".join(lines)


ap = argparse.ArgumentParser()
ap.add_argument("baseline")
ap.add_argument("fix")
ap.add_argument("--freq-mhz", type=float, default=1350.0)
args = ap.parse_args()
zones = {"reduce_max", "reduce_sum"}

b = load_trisc2(args.baseline, zones)
f_ = load_trisc2(args.fix, zones)

buckets = [(0, 100), (100, 150), (150, 200), (200, 300), (300, 400), (400, 500), (500, 600), (600, 800)]

ns = lambda c: c / args.freq_mhz * 1000.0
for z in ("reduce_max", "reduce_sum"):
    print(f"\n=== {z}  (TRISC_2 / PACK)  cycles  [ns ≈ cycles × {1000/args.freq_mhz:.3f}]\n")
    for label, data in (("BASELINE", b[z]), ("FIX     ", f_[z])):
        vs = sorted(data)
        if not vs:
            print(f"{label}: (no data)")
            continue
        n = len(vs)
        mn, mx = vs[0], vs[-1]
        med = vs[n // 2]
        p90 = vs[int(0.90 * (n - 1))]
        p99 = vs[int(0.99 * (n - 1))]
        avg = sum(vs) / n
        print(
            f"{label}  n={n:5d}  min={mn:3d} ({ns(mn):5.1f}ns)  med={med:4d} ({ns(med):6.1f}ns)  mean={avg:6.1f} ({ns(avg):6.1f}ns)  p90={p90:4d} ({ns(p90):6.1f}ns)  p99={p99:4d} ({ns(p99):6.1f}ns)  max={mx:4d} ({ns(mx):6.1f}ns)"
        )
        print(histogram(data, buckets))
        print()
