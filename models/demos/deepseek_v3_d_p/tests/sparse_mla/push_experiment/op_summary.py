import collections
import csv
import statistics
import sys

for p in sys.argv[1:]:
    rows = [r for r in csv.DictReader(open(p)) if r["DEVICE KERNEL DURATION [ns]"]]
    d = collections.defaultdict(list)
    for r in rows:
        d[r["OP CODE"]].append(float(r["DEVICE KERNEL DURATION [ns]"]) / 1000)
    print(p.split("/")[-1])
    for op in ("IndexerScoreDeviceOperation", "TopkLargeIndicesDeviceOperation", "SparseSDPAOperation"):
        v = d[op]
        print(f"  {op:35s} n={len(v):4d} med={statistics.median(v):8.1f} min={min(v):8.1f} max={max(v):8.1f}")
    g = []
    for i, r in enumerate(rows):
        if r["OP CODE"].startswith("TopkLarge"):
            for r2 in rows[i + 1 : i + 40]:
                if r2["OP CODE"].startswith("HighBwAllGather") and r2["DEVICE ID"] == r["DEVICE ID"]:
                    g.append(float(r2["DEVICE KERNEL DURATION [ns]"]) / 1000)
                    break
    print("  overlapped gather med", round(statistics.median(g), 1))
