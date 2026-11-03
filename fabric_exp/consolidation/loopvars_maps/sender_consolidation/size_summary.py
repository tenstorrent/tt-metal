#!/usr/bin/env python3
"""size_summary.py SIZE_REPORT_OUTPUT: from size_report.py's per-case table, per build over every case it has: largest
one-ERISC binary (E1 code of *_1erisc cases), least free stack E0 / E1, and VC1 (bh_vc1) E0+E1. V4_SIZE.md's log
columns."""
import re, sys
lines = open(sys.argv[1]).read().splitlines()
names = None
rows = {}
for l in lines:
    if "|" in l and names is None and l.startswith("case"):
        names = [x.split()[0] for x in l.split("|")[1:]]
        continue
    if "|" in l and not l.startswith("case") and not l.startswith("-"):
        parts = [x.strip() for x in l.split("|")]
        rows[parts[0]] = [p.split() for p in parts[1:]]
if names is None:
    names = [f"build{i}" for i in range(len(next(iter(rows.values()))))]
for i, n in enumerate(names):
    one = max((int(r[i][1]), c) for c, r in rows.items() if c.endswith("_1erisc") and r[i][1] != "-")
    f0 = min((int(r[i][3]), c) for c, r in rows.items() if r[i][3] != "-")
    f1 = min((int(r[i][4]), c) for c, r in rows.items() if r[i][4] != "-")
    vc1 = rows.get("bh_vc1", [None] * (i + 1))[i]
    print(f"{n:12s} VC1 E0+E1 {vc1[2] if vc1 else '-':>6}  largest one-ERISC {one[0]} ({one[1]})  least free E0 {f0[0]} "
          f"({f0[1]}) / E1 {f1[0]} ({f1[1]})")
