"""timeline.py <tag> [chunk=-1] [chip=0] [min_us=5]: programs of one chunk on one chip in start order."""
import os
import json
import sys
from collections import defaultdict

S = os.environ.get("PUSH_EXP_DIR", "/tmp/push_exp") + "/runs"
tag = sys.argv[1]
chunk = int(sys.argv[2]) if len(sys.argv) > 2 else -1
chip = int(sys.argv[3]) if len(sys.argv) > 3 else 0
min_us = float(sys.argv[4]) if len(sys.argv) > 4 else 5
d = json.load(open(f"{S}/{tag}/rt/rank0.json"))
f = d["meta"]["freq"]
per = defaultdict(list)
for c, rid, s, e in d["recs"]:
    if c == chip:
        per[rid].append((s, e))
n = max(len(v) for v in per.values())
rows = []
for rid, v in per.items():
    v.sort()
    if len(v) != n:
        continue
    s, e = v[chunk]
    k = d["names"][str(rid)].split(";")
    name = k[0].split("operations/")[-1].split("kernels/")[-1]
    rows.append((s, e, name, len(k)))
rows.sort()
t0 = rows[0][0]
prev_end = t0
for s, e, name, nk in rows:
    dur = (e - s) / f / 1e3
    gap = (s - prev_end) / f / 1e3
    if dur >= min_us:
        print(f"{(s - t0) / f / 1e3:9.1f} +{dur:8.1f} us  gap {gap:7.1f}  {name[:90]}")
    prev_end = max(prev_end, e)
print(f"chunk span {(prev_end - t0) / f / 1e3:.1f} us, {len(rows)} programs ({n} occurrences each)")
