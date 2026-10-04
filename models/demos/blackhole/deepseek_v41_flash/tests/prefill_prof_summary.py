"""python prefill_prof_summary.py <dir with .logs/cpp_device_perf_report.csv and marks.json> [layers=2]
Device kernel time per PHASE of the prefill layer (phases = device-operation id ranges recorded by prefill_layer._mark), summed over the layer's
ops: mean and max over the 32 devices of the per-device sum, op counts, plus the biggest individual op instances (by core count class)."""
import bisect
import csv
import json
import sys
from collections import defaultdict

d = sys.argv[1]
nl = int(sys.argv[2]) if len(sys.argv) > 2 else 1
marks = json.load(open(d + "/marks.json"))
ids = [m[1] for m in marks]
K = "DEVICE KERNEL DURATION [ns]"
per = defaultdict(lambda: defaultdict(float))  # phase -> device -> ns
cnt = defaultdict(lambda: defaultdict(int))
top = []
for r in csv.DictReader(open(d + "/.logs/cpp_device_perf_report.csv")):
    if r["METAL TRACE ID"]:
        continue
    oid = int(float(r["GLOBAL CALL COUNT"])) >> 10
    i = bisect.bisect_right(ids, oid)  # first mark whose end id > oid
    if i >= len(marks) or oid < ids[0]:
        continue
    ph = marks[i][0]
    dev = r["DEVICE ID"]
    v = float(r[K] or 0)
    per[ph][dev] += v
    cnt[ph][dev] += 1
tot_mean = sum(sum(v.values()) / len(v) for v in per.values())
print(f"total kernel time per layer (mean over devices): {tot_mean / 1e6 / nl:.2f} ms")
for ph, v in sorted(per.items(), key=lambda kv: -sum(kv[1].values()) / len(kv[1])):
    mean = sum(v.values()) / len(v)
    print(
        f"{ph:22s} mean {mean / 1e6 / nl:8.3f} ms/layer  max-dev {max(v.values()) / 1e6 / nl:8.3f}  {100 * mean / tot_mean:5.1f}%  ops/dev/layer {max(cnt[ph].values()) / nl:7.1f}  avg op {mean / max(cnt[ph].values()) / 1e3:7.1f} us"
    )
