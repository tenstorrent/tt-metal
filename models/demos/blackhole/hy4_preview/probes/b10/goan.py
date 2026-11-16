#!/usr/bin/env python3
"""goan.py <raw.json (rawan)> <disp.csv> : per op: A = go-signal send time (dispatch subordinate NCRISC, CQ_DISPATCH_CMD_SEND_GO_SIGNAL start) - prev op last kernel end; B = K0 - go; per chip then mean."""
import bisect
import collections
import json
import statistics as st
import sys

d = json.load(open(sys.argv[1]))
go = collections.defaultdict(list)
for ln in open(sys.argv[2]):
    f = ln.rstrip("\n").split(",")
    if "SEND_GO_SIGNAL" in f[8] and f[9] == "ZONE_START":
        go[int(f[0])].append(int(f[5]) / 1350.0)
for v in go.values():
    v.sort()
ops = sorted(int(k) for k in d)
res = {}
for op in ops[1:]:
    cur = {r["dev"]: r for r in d[str(op)]}
    prv = {r["dev"]: r for r in d.get(str(op - 1), [])}
    A = []
    B = []
    G = []
    for dev, r in cur.items():
        if dev not in prv or dev not in go:
            continue
        g = go[dev]
        i = bisect.bisect_right(g, r["K0"] + 0.05) - 1  # last go at or before kernel start
        if i < 0:
            continue
        tg = g[i]
        if r["K0"] - tg > 3:
            continue  # unmatched
        A.append(tg - prv[dev]["K1"])
        B.append(r["K0"] - tg)
        G.append(r["K0"] - prv[dev]["K1"])
    if A:
        res[op] = (st.mean(A), min(A), max(A), st.mean(B), st.mean(G), len(A))
json.dump(res, open(sys.argv[3], "w"))
print("op | A=go-prevK1 (mean min max) | B=K0-go | gap | nchips")
for op, v in res.items():
    print(op, "| %.2f %.2f %.2f | %.2f | %.2f | %d" % v)
