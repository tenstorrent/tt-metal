"""python opsum_decode.py <profiler dir with segments.json and .logs/cpp_device_perf_report.csv> [segment ...]
Per traced segment (replay session 2, the second replay): the ordered op list with core count, kernel duration (ns: mean and max over the 32 devices) and op-to-op latency (mean); totals per segment."""
import csv
import json
import sys
from collections import defaultdict

d = sys.argv[1]
want = sys.argv[2:]
segs = json.load(open(d + "/segments.json"))
K, OO = "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]"
by = defaultdict(lambda: defaultdict(list))  # (tid, session) -> device -> [(call, dur, cores, oo)]
for r in csv.DictReader(open(d + "/.logs/cpp_device_perf_report.csv")):
    if not r["METAL TRACE ID"]:
        continue
    by[(r["METAL TRACE ID"], r["METAL TRACE REPLAY SESSION ID"])][r["DEVICE ID"]].append(
        (int(r["GLOBAL CALL COUNT"]), float(r[K] or 0), int(r["CORE COUNT"] or 0), float(r[OO] or 0))
    )
out = {}
for name, s in segs.items():
    if want and name not in want:
        continue
    sess = by[(str(s["trace_id"]), "2")] or by[(str(s["trace_id"]), "1")]
    devs = {k: sorted(v) for k, v in sess.items()}
    n = min(len(v) for v in devs.values())
    ops = []
    for i in range(n):
        v = [devs[k][i] for k in devs]
        ops.append(
            dict(
                cores=v[0][2],
                mean=sum(x[1] for x in v) / len(v),
                max=max(x[1] for x in v),
                oo=sum(x[3] for x in v) / len(v),
            )
        )
    tot_k = sum(o["mean"] for o in ops) / 1e6
    tot_max = sum(o["max"] for o in ops) / 1e6
    tot_oo = sum(o["oo"] for o in ops) / 1e6
    out[name] = dict(n_ops=len(ops), kernel_ms_mean=tot_k, kernel_ms_sumofmax=tot_max, op2op_ms=tot_oo, ops=ops)
    print(
        f"==== {name} {s}: {len(ops)} ops, kernel sum (mean over devices) {tot_k:.3f} ms, sum of per-op max over devices {tot_max:.3f} ms, op-to-op gaps {tot_oo:.3f} ms"
    )
    for i, o in enumerate(ops):
        print(
            f"{i:3d} cores={o['cores']:4d} kernel mean {o['mean']/1e3:9.1f} us max {o['max']/1e3:9.1f} us  op2op {o['oo']/1e3:8.1f} us"
        )
json.dump(out, open(d + "/opsum.json", "w"))
