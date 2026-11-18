#!/usr/bin/env python3
"""Prefill op breakdown from the raw tracy logs when post-processing failed.
usage: analyze_prefill_raw.py <profiler .logs dir>
Joins cpp_device_perf_report.csv (device times) with tracy_ops_data.csv (op names) on GLOBAL CALL COUNT, takes device 0
from the last Embeddings op to the end (the final profiled prefill), and sums kernel time per op code."""
import csv
import re
import sys
from collections import defaultdict

import pandas as pd

d = sys.argv[1].rstrip("/")
dev = pd.read_csv(f"{d}/cpp_device_perf_report.csv", low_memory=False)

# tracy_ops_data.csv: lines like  `TT_DNN_DEVICE_OP: "<name>", <hash>, <global id>, ...`  (format varies); grab name + id
names = {}
with open(f"{d}/tracy_ops_data.csv", errors="ignore") as f:
    for line in f:
        m = re.search(r'TT_DNN_DEVICE_OP:\s*"([A-Za-z0-9_:]+)",\s*[0-9]+,\s*[0-9]+,\s*\w+,\s*([0-9]+)', line)
        if m:
            names[int(m.group(2))] = m.group(1)
            continue
        m = re.search(r'"op_code"\s*:\s*"([^"]+)".*?"global_call_count"\s*:\s*([0-9]+)', line)
        if m:
            names[int(m.group(2))] = m.group(1)
print(f"named ops: {len(names)}  device rows: {len(dev)}")
if not names:
    with open(f"{d}/tracy_ops_data.csv", errors="ignore") as f:
        print("tracy_ops_data.csv head:", [next(f)[:200] for _ in range(3)])
    sys.exit(1)

dev["OP"] = dev["GLOBAL CALL COUNT"].map(names)
d0 = (
    dev[dev["DEVICE ID"] == sorted(dev["DEVICE ID"].unique())[0]]
    .sort_values("GLOBAL CALL COUNT")
    .reset_index(drop=True)
)
emb = [i for i, n in enumerate(d0["OP"]) if isinstance(n, str) and "Embedding" in n]
start = emb[-1] if emb else 0
seg = d0.iloc[start:]
K = "DEVICE KERNEL DURATION [ns]"
agg = defaultdict(lambda: [0, 0.0])
for _, r in seg.iterrows():
    a = agg[r["OP"]]
    a[0] += 1
    a[1] += r[K]
tot = sum(v[1] for v in agg.values())
s, e = seg["DEVICE FW START CYCLE"].min(), seg["DEVICE FW END CYCLE"].max()
print(
    f"final prefill segment: {len(seg)} ops, kernel sum {tot/1e6:.2f} ms, wall {(e - s)/1.35e9*1e3:.2f} ms (at 1.35 GHz)"
)
print(f"{'op':52} {'n':>4} {'kernel ms':>10} {'%':>6}")
for op, (n, k) in sorted(agg.items(), key=lambda x: -x[1][1])[:30]:
    print(f"{str(op)[:52]:52} {n:>4} {k/1e6:>10.3f} {100*k/tot:>5.1f}%")
