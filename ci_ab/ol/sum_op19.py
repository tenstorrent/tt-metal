#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run: per node and repetition (signpost "<node>#<k>") and op code, the sum of DEVICE KERNEL
DURATION [ns] and the number of launches, printed as "RAB OPC <arm> <run> <case> k<k> <op> <ns> <launches>" (spaces of a case
dropped)."""
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
    print(f"RAB DEVONLY {arm} {run}")
    sys.exit(0)
cur, acc, order = None, defaultdict(lambda: [0.0, 0]), []
import re
names = [re.sub(r".*case=([A-Za-z0-9_]+).*", r"\1", l.strip()) for l in open(os.environ["AB_NODES_FILE"]) if l.strip()]
idx = -1
for r in rows:
    if r["OP TYPE"] == "signpost":
        node, _, k = r["OP CODE"].rpartition("#")
        if k.isdigit() and int(k) == 0:
            idx += 1
        cur = (names[idx] if 0 <= idx < len(names) else node, int(k)) if k.isdigit() else None
        continue
    if cur is None:
        continue
    try:
        v = float(r[COL])
    except (ValueError, KeyError):
        continue
    a = acc[(cur, r["OP CODE"])]
    a[0] += v; a[1] += 1
for (cur, op), (ns, n) in acc.items():
    print(f"RAB OPC {arm} {run} {cur[0]} k{cur[1]} {op} {ns:.0f} {n}")
