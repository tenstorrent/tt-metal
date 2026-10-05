"""python opsum.py <profiler dir with .logs/cpp_device_perf_report.csv and rows.json> [layer] -- joins the recorded calls of the last chunk of each layer
(tests/test_prefill_sparse_device.py with DSV41_PS_PROF=1) with the device digest: kernel time (ms, mean over the 32 devices) per call, listed in order,
then summed per caller line."""

import csv
import json
import sys
from collections import defaultdict

d = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else None
rows = json.load(open(d + "/rows.json"))
K = "DEVICE KERNEL DURATION [ns]"
dur = defaultdict(list)
for r in csv.DictReader(open(d + "/.logs/cpp_device_perf_report.csv")):
    if r["METAL TRACE ID"]:
        continue
    dur[int(float(r["GLOBAL CALL COUNT"])) >> 10].append(float(r[K] or 0))
for L, calls in rows.items():
    if want and L != want:
        continue
    tot = 0.0
    byc = defaultdict(float)
    print(f"==== layer {L}: {len(calls)} calls")
    for c in calls:
        ms = sum(sum(dur.get(i, [0])) / max(1, len(dur.get(i, [0]))) for i in range(c["id0"], c["id1"])) / 1e6
        tot += ms
        byc[c["caller"] + " " + c["op"]] += ms
        if ms > 0.05:
            print(f"{ms:8.3f} ms {c['op']:45s} {c['caller']:40s} {c['ins'][:2]} -> {c['outs'][:1]}")
    print(f"---- total {tot:.2f} ms; by caller:")
    for k, v in sorted(byc.items(), key=lambda x: -x[1])[:25]:
        print(f"{v:8.3f} ms {k}")
