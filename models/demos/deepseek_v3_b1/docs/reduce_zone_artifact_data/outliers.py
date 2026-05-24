#!/usr/bin/env python3
"""Show which (core, run) had reduce_sum > 200c after the fix."""
import csv
import sys

for path in sys.argv[1:]:
    label = path.split("/tracy_run/")[1].split("/")[0]
    opens = {}
    outliers = []
    with open(path) as f:
        f.readline()
        reader = csv.DictReader(f, skipinitialspace=True)
        for row in reader:
            if row["zone name"].strip() != "reduce_sum":
                continue
            if row["RISC processor type"].strip() != "TRISC_2":
                continue
            ztype = row["type"].strip()
            key = (row["core_x"].strip(), row["core_y"].strip())
            t = int(row["time[cycles since reset]"].strip())
            if ztype == "ZONE_START":
                opens.setdefault(key, []).append(t)
            elif ztype == "ZONE_END":
                s = opens.get(key, [])
                if s:
                    d = t - s.pop(0)
                    if d > 200:
                        outliers.append((d, key))
    if outliers:
        print(f"\n{label}: {len(outliers)} outliers (>200c)")
        for d, (cx, cy) in sorted(outliers):
            print(f"  ({cx:>2s},{cy:>2s})  {d:4d} cycles ({d/1.350:.1f} ns)")
    else:
        print(f"\n{label}: no outliers")
