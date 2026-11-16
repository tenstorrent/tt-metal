#!/usr/bin/env python3
"""rawan.py <raw_filtered.csv> <out.json> : per op x chip decomposition from per-core zone timestamps (last replay session = max time per core/zone).
Per (chip, op): FWs=min BRISC-FW start, FWl=max BRISC-FW start (go-arrival spread over the op's cores), K0=min kernel start (any RISC), K1=max kernel end, FWe=max BRISC-FW end.
startup = K0-FWs (first core: FW entry -> kernel start), teardown = FWe-K1 (last kernel end -> FW end), gap(i->i+1, chip) = FWs(i+1) - FWe(i). us at 1350 MHz."""
import collections
import json
import statistics as st
import sys

best = {}
for ln in open(sys.argv[1]):
    c, x, y, risc, t, hid, zone, typ = ln.rstrip("\n").split(",")
    hid = int(hid)
    k = (hid & 1023, hid >> 10, int(x), int(y), risc, zone, typ)
    t = int(t)
    if best.get(k, -1) < t:
        best[k] = t
ops = collections.defaultdict(lambda: collections.defaultdict(list))
for (dev, op, x, y, risc, zone, typ), t in best.items():
    us = t / 1350.0
    d = ops[(dev, op)]
    if zone == "BRISC-FW":
        d["fw" + typ].append(us)
    elif zone.endswith("KERNEL"):
        d["k" + typ].append(us)
res = {}
for (dev, op), d in ops.items():
    if not d["fwZONE_START"] or not d["kZONE_END"]:
        continue
    res[(dev, op)] = dict(
        FWs=min(d["fwZONE_START"]),
        FWl=max(d["fwZONE_START"]),
        K0=min(d["kZONE_START"]),
        K0l=max(d["kZONE_START"]),
        K1=max(d["kZONE_END"]),
        K1f=min(d["kZONE_END"]),
        FWe=max(d["fwZONE_END"]),
        n=len(d["fwZONE_START"]),
    )
out = collections.defaultdict(list)
for dev in sorted({k[0] for k in res}):
    prev = None
    for op in sorted(o for (dv, o) in res if dv == dev):
        r = res[(dev, op)]
        r.update(
            startup=r["K0"] - r["FWs"],
            teardown=r["FWe"] - r["K1"],
            spread=r["FWl"] - r["FWs"],
            kdur=r["K1"] - r["K0"],
            kspr=r["K1"] - r["K1f"],
            gap=(r["FWs"] - prev["FWe"]) if prev else None,
        )
        out[op].append(dict(dev=dev, **r))
        prev = r
json.dump({str(k): v for k, v in out.items()}, open(sys.argv[2], "w"))
print("op | cores kdur startup teardown go_spread kend_spread | gap")
for op in sorted(out):
    L = out[op]
    m = lambda k: st.mean(r[k] for r in L if r[k] is not None) if any(r[k] is not None for r in L) else float("nan")
    print(
        op,
        "|",
        round(m("n")),
        "%.1f %.2f %.2f %.2f %.1f | %.2f" % (m("kdur"), m("startup"), m("teardown"), m("spread"), m("kspr"), m("gap")),
    )
