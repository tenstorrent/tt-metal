#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run of prof_ops.py: per case, the device ops between its signpost and the next "end" signpost, split
evenly into N_MEAS calls; per call the sum of DEVICE KERNEL DURATION [ns]: "OPS OP <arm> <run> <case> <cores> <ns,...>"."""
import csv, glob, os, sys
out, arm, run = sys.argv[1:4]
N = int(os.environ.get("OPS_NMEAS", "8"))
COL = "DEVICE KERNEL DURATION [ns]"
fs = sorted(glob.glob(os.path.join(out, "reports", "*", "ops_perf_results_*.csv")))
if not fs:
    print(f"OPS NOREPORT {arm} {run}"); sys.exit(0)
rows = list(csv.DictReader(open(fs[-1])))
if not rows or "OP TYPE" not in rows[0]:
    print(f"OPS DEVONLY {arm} {run}"); sys.exit(0)
cur, vals, cores = None, [], ""
for r in rows:
    if r["OP TYPE"] == "signpost":
        if cur is not None and r["OP CODE"] == "end":
            per = len(vals) // N if vals and len(vals) % N == 0 else 0
            calls = [sum(vals[i * per:(i + 1) * per]) for i in range(N)] if per else []
            print(f"OPS OP {arm} {run} {cur} {cores} " + (",".join(f"{v:.0f}" for v in calls) if calls else f"BADCOUNT{len(vals)}"))
            cur = None
        elif r["OP CODE"] != "end":
            cur, vals, cores = r["OP CODE"], [], ""
        continue
    if cur is None:
        continue
    try:
        vals.append(float(r[COL])); cores = max(cores, r.get("CORE COUNT", ""), key=lambda c: int(c) if c.isdigit() else -1)
    except (ValueError, KeyError):
        pass
