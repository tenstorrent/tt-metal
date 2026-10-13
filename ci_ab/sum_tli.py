#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run: per case (signpost), the first N_MEAS topk_large_indices entries' DEVICE KERNEL
DURATION [ns] and core count, printed as "TAB OP <arm> <run> <case> <cores> <ns,ns,...>"."""
import csv
import glob
import os
import sys
from collections import defaultdict

out, arm, run = sys.argv[1:4]
N = int(os.environ.get("TLI_NMEAS", "8"))
COL = "DEVICE KERNEL DURATION [ns]"
fs = sorted(glob.glob(os.path.join(out, "reports", "*", "ops_perf_results_*.csv")))
if not fs:
    print(f"TAB NOREPORT {arm} {run}")
    sys.exit(0)
rows = list(csv.DictReader(open(fs[-1])))
if not rows or "OP TYPE" not in rows[0]:
    print(f"TAB DEVONLY {arm} {run} " + " ".join(r.get(COL, "") for r in rows))
    sys.exit(0)
cur, vals, cores, order = None, defaultdict(list), {}, []
for r in rows:
    if r["OP TYPE"] == "signpost":
        cur = r["OP CODE"]
        order.append(cur)
        continue
    if cur is None or "TopkLargeIndices" not in r["OP CODE"]:
        continue
    try:
        if len(vals[cur]) < N:
            vals[cur].append(float(r[COL]))
            cores[cur] = r.get("CORE COUNT", "")
    except (ValueError, KeyError):
        pass
for c in order:
    print(f"TAB OP {arm} {run} {c} {cores.get(c, '')} " + ",".join(f"{v:.0f}" for v in vals[c]))
