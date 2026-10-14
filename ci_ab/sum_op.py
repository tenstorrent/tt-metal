#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run: per node and repetition (signpost "<node>#<k>"), the sum of DEVICE KERNEL
DURATION [ns] over its ops, printed as "RAB OP <arm> <run> <case> k<k> <ns> <ops>". A report without signposts prints
its device durations in call order (RAB DEVONLY) so the reps can be split offline."""
import csv, glob, os, sys
from collections import defaultdict

out, arm, run = sys.argv[1:4]
COL = "DEVICE KERNEL DURATION [ns]"
fs = sorted(glob.glob(os.path.join(out, "reports", "*", "ops_perf_results_*.csv")))
if not fs:
    print(f"RAB NOREPORT {arm} {run}")
    sys.exit(0)
rows = list(csv.DictReader(open(fs[-1])))
if not rows or "OP TYPE" not in rows[0]:
    print(f"RAB DEVONLY {arm} {run} " + " ".join(r.get(COL, "") for r in rows))
    sys.exit(0)
cur, acc, order = None, defaultdict(lambda: [0.0, 0]), []
for r in rows:
    if r["OP TYPE"] == "signpost":
        node, _, k = r["OP CODE"].rpartition("#")
        cur = (node.split("::")[-1], int(k)) if k.isdigit() else None
        if cur:
            order.append(cur)
        continue
    if cur is None:
        continue
    try:
        acc[cur][0] += float(r[COL]); acc[cur][1] += 1
    except (ValueError, KeyError):
        pass
for key in order:
    print(f"RAB OP {arm} {run} {key[0]} k{key[1]} {acc[key][0]:.0f} {acc[key][1]}")
