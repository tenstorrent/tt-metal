#!/usr/bin/env python3
"""Scratch, not for merge. One tracy run of test_w3_sdpa.py: the SDPA device kernel durations in call order, split per
case by the run's W3_LOG, printed as "RAB ROW <arm> <run> <case> <ns per call...>" (the first call compiles)."""
import csv, os, re, sys

out, log, arm, run = sys.argv[1:5]
COL = "DEVICE KERNEL DURATION [ns]"
OP = re.compile(r'TT_DNN_DEVICE_OP: "([^"]+)", \d+, \d+, \w+, (\d+)')
p = os.path.join(out, ".logs", "cpp_device_perf_report.csv")
if not os.path.exists(p):
    print(f"RAB NOREPORT {arm} {run}")
    sys.exit(0)
rows = list(csv.DictReader(open(p)))
q = os.path.join(out, ".logs", "tracy_ops_data.csv")
names = {int(n): op for op, n in OP.findall(open(q).read())} if os.path.exists(q) else {}
if names:
    rows = [r for r in rows if "sdpa" in names.get(int(r["GLOBAL CALL COUNT"]), "").lower()]
rows.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
cases = [l.split() for l in open(log) if l.strip()]
n = sum(int(c) for _, c in cases)
if n != len(rows):
    print(f"RAB MISMATCH {arm} {run} rows {len(rows)} calls {n}")
i = 0
for case, c in cases:
    v = [r[COL] for r in rows[i : i + int(c)]]
    i += int(c)
    print(f"RAB ROW {arm} {run} {case} {' '.join(v)}")
