#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run: per case (signpost), the first N device ops after it (the measured calls):
DEVICE KERNEL DURATION [ns], core count and op code, printed as "TAB OP <arm> <run> <case> <cores> <op> <ns,...>"."""
import csv
import glob
import os
import sys

out, arm, run = sys.argv[1:4]
N = int(os.environ.get("SDPA_NMEAS", "5"))
COL = "DEVICE KERNEL DURATION [ns]"
fs = sorted(glob.glob(os.path.join(out, "reports", "*", "ops_perf_results_*.csv")))
if not fs:
    print(f"TAB NOREPORT {arm} {run}")
    sys.exit(0)
rows = list(csv.DictReader(open(fs[-1])))
cur, got = None, {}
for r in rows:
    if r.get("OP TYPE") == "signpost":
        cur = r["OP CODE"]
        got[cur] = []
        continue
    if cur is None or len(got[cur]) >= N:
        continue
    try:
        got[cur].append((float(r[COL]), r.get("CORE COUNT", ""), r.get("OP CODE", "")))
    except (ValueError, KeyError):
        pass
for c, v in got.items():
    if v:
        print(f"TAB OP {arm} {run} {c} {v[0][1]} {v[0][2]} " + ",".join(f"{x[0]:.0f}" for x in v))
