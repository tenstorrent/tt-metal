#!/usr/bin/env python3
"""Round 3 eltwise binary: device kernel time of one op code summed over every launch (all devices) in each run's tracy ops
report, for runs made without the repetition plugin (one test per process); then main against the opt-in over all runs.
usage: prof_sum.py <prof dir> <op code regex>   (reads <prof dir>/out_<test idx>_<run>_<variant>/reports/*/ops_perf_results_*.csv)"""
import collections
import csv
import glob
import os
import re
import statistics
import sys

d, opre = sys.argv[1], re.compile(sys.argv[2])
COL = "DEVICE KERNEL DURATION [ns]"
data = collections.defaultdict(lambda: collections.defaultdict(list))
names = {}
for rd in sorted(glob.glob(os.path.join(d, "out_*_*_*"))):
    _, t, run, v = os.path.basename(rd).split("_", 3)
    fs = glob.glob(os.path.join(rd, "reports", "*", "ops_perf_results_*.csv"))
    if not fs:
        print(f"PROFSUM {t} run {run} {v}: no report")
        continue
    rows = list(csv.DictReader(open(fs[0])))
    if not rows or "OP CODE" not in rows[0]:
        print(f"PROFSUM {t} run {run} {v}: report without host op data")
        continue
    sel = [r for r in rows if opre.search(r["OP CODE"]) and r.get(COL, "").replace(".", "", 1).isdigit()]
    tot = sum(float(r[COL]) for r in sel)
    print(f"PROFSUM {t} run {run} {v}: {len(sel)} launches, {tot:.0f} ns")
    data[t][v].append(tot)
for t, vv in sorted(data.items()):
    m, o = vv.get("main", []), vv.get("optin", [])
    if not m or not o:
        continue
    mm, om = statistics.median(m), statistics.median(o)
    rng = max(max(m) - min(m), max(o) - min(o))
    verdict = "equal" if abs(om - mm) <= rng else ("faster" if om < mm else "slower")
    print(f"PROFSUM test {t}: main n={len(m)} median {mm:.0f} ({', '.join(f'{x:.0f}' for x in m)}) optin n={len(o)} median {om:.0f} ({', '.join(f'{x:.0f}' for x in o)}) change {om - mm:+.0f} ns ({100 * (om - mm) / mm:+.2f} %) range {rng:.0f} {verdict}")
