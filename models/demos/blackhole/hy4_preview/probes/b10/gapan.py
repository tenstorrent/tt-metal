#!/usr/bin/env python3
"""gapan.py <cpp_device_perf_report.csv> <ops.json> [--session N] : per-op split of kernel time / FW overhead / launch spread / op-to-op gap, per chip stats.
cols: kernel = DEVICE KERNEL DURATION, fwoh = FW DURATION - KERNEL DURATION (FW start->kernel start + kernel end->FW end = startup+teardown),
l2l = KERNEL FIRST TO LAST START (spread of the kernel start over the cores of the op = go-signal multicast skew), gap = OP TO OP LATENCY."""
import argparse
import collections
import csv
import json
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("ops")
ap.add_argument("--session", type=int, default=None)
ap.add_argument("--json", default=None)
a = ap.parse_args()
rec = json.load(open(a.ops))["rows"]
rows = [r for r in csv.DictReader(open(a.csv)) if r["METAL TRACE ID"] != ""]
sess = sorted({int(r["METAL TRACE REPLAY SESSION ID"]) for r in rows})
S = a.session or sess[-1]
rows = [r for r in rows if int(r["METAL TRACE REPLAY SESSION ID"]) == S]
f = lambda x: float(x) if x not in ("", None) else 0.0
per = collections.defaultdict(dict)
for r in rows:
    g = int(r["GLOBAL CALL COUNT"])
    per[g >> 10][g & 1023] = dict(
        k=f(r["DEVICE KERNEL DURATION [ns]"]),
        fw=f(r["DEVICE FW DURATION [ns]"]),
        l2l=f(r["DEVICE KERNEL FIRST TO LAST START [ns]"]),
        gap=f(r["OP TO OP LATENCY [ns]"]),
        cores=int(f(r["CORE COUNT"])),
        s=int(f(r["DEVICE FW START CYCLE"])),
        e=int(f(r["DEVICE FW END CYCLE"])),
    )
ids = sorted(per)
out = []
for i, rw in enumerate(rec):
    for oid in range(rw["id0"], rw["id1"]):
        if oid not in per:
            continue
        d = per[oid]
        ch = sorted(d)
        m = lambda k: st.mean(d[c][k] for c in ch) / 1e3
        out.append(
            dict(
                row=i,
                oid=oid,
                op=rw["op"].split(".")[-1],
                caller=rw["caller"],
                cores=max(d[c]["cores"] for c in ch),
                k=m("k"),
                fwoh=m("fw") - m("k"),
                l2l=m("l2l"),
                gap=0.0 if oid == ids[0] else m("gap"),
                gmin=min(d[c]["gap"] for c in ch) / 1e3,
                gmax=max(d[c]["gap"] for c in ch) / 1e3,
                kmin=min(d[c]["k"] for c in ch) / 1e3,
                kmax=max(d[c]["k"] for c in ch) / 1e3,
            )
        )
if a.json:
    json.dump(out, open(a.json, "w"))
print("row oid op caller cores | kernel kmin kmax | fwoh l2l | gap gmin gmax")
for o in out:
    print(
        o["row"],
        o["oid"],
        o["op"][:22],
        o["caller"].split(":")[0][-20:],
        o["cores"],
        "| %.1f %.1f %.1f | %.2f %.2f | %.1f %.1f %.1f"
        % (o["k"], o["kmin"], o["kmax"], o["fwoh"], o["l2l"], o["gap"], o["gmin"], o["gmax"]),
    )
