#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run of one case: after the last signpost, the device kernel durations per op code,
printed as "RAB OP <arm> <run> <case> <op> <n> <median ns> <values>"."""
import csv, glob, os, statistics, sys
out, arm, run, case = sys.argv[1:5]
fs = sorted(glob.glob(os.path.join(out, "reports", "*", "ops_perf_results_*.csv")))
if not fs:
    print(f"RAB OP {arm} {run} {case} NOREPORT"); sys.exit(0)
rows = list(csv.DictReader(open(fs[-1])))
idx = max([i for i, r in enumerate(rows) if r.get("OP TYPE") == "signpost"] or [-1])
ops = {}
for r in rows[idx + 1:]:
    if r.get("OP TYPE") != "signpost" and r.get("DEVICE KERNEL DURATION [ns]"):
        ops.setdefault(r["OP CODE"].replace(" ", "_"), []).append(float(r["DEVICE KERNEL DURATION [ns]"]))
for op, xs in ops.items():
    print(f"RAB OP {arm} {run} {case} {op} {len(xs)} {statistics.median(xs):.0f} " + " ".join("%.0f" % x for x in xs))
