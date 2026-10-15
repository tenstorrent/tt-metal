#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run of prof_ops12.py: per op code the median of the last N device kernel durations
of the report. usage: ol_sum.py <out dir> <arm> <round> <case> <N>; prints "RAB OL <arm> <round> <case> <op> <calls> <median> <values>"."""
import csv, glob, os, statistics, sys
from collections import defaultdict
d, arm, rnd, case, n = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
fs = sorted(glob.glob(d + "/reports/*/ops_perf_results_*.csv") + glob.glob(d + "/ops_perf_results_*.csv"))
if not fs:
    print("RAB NOREPORT", arm, rnd, case)
    sys.exit(0)
agg = defaultdict(list)
for row in csv.DictReader(open(fs[-1])):
    if row.get("OP TYPE") == "signpost":
        continue
    try:
        agg[row["OP CODE"]].append(float(row["DEVICE KERNEL DURATION [ns]"]))
    except (KeyError, ValueError):
        pass
for op, vals in sorted(agg.items()):
    cut = vals[-n:]
    print("RAB OL", arm, rnd, case, op, len(vals), round(statistics.median(cut)), " ".join(str(round(x)) for x in cut))
