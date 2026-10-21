#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run: per signposted case, the dominant op's device kernel durations [ns], printed as
"RAB <arm> <run> <case> <op> <n> <median> <comma-separated values>"."""
import csv, glob, os, statistics, sys
from collections import defaultdict
out, arm, run = sys.argv[1:4]
fs = sorted(glob.glob(os.path.join(out, "reports", "*", "ops_perf_results_*.csv")))
if not fs:
    print(f"RAB NOREPORT {arm} {run}"); sys.exit(0)
case, per = None, defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(fs[-1])):
    if r.get("OP TYPE") == "signpost":
        case = r["OP CODE"]; continue
    v = r.get("DEVICE KERNEL DURATION [ns]")
    if case and v:
        per[case][r["OP CODE"]].append(float(v))
for c, ops in per.items():
    op = max(ops, key=lambda o: sum(ops[o])); xs = ops[op]
    print(f"RAB {arm} {run} {c} {op} {len(xs)} {statistics.median(xs):.0f} {','.join(f'{x:.0f}' for x in xs)}")
