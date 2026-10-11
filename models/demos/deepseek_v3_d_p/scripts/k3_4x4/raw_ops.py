"""Build an ops_perf-style CSV (OP CODE, DEVICE ID, DEVICE KERNEL DURATION [ns]) from the raw profiler
logs when tracy's report step fails. usage: raw_ops.py <.logs dir> <out.csv>"""
import bisect
import re
import sys

import pandas as pd

d, out = sys.argv[1], sys.argv[2]
txt = open(f"{d}/tracy_ops_data.csv", errors="replace").read()
gcc = [(m.start(), int(m.group(1))) for m in re.finditer(r'"global_call_count":\s*(\d+)', txt)]
ops = [(m.start(), m.group(1)) for m in re.finditer(r'"op_code":\s*"([^"]+)"', txt)]
gpos = [p for p, _ in gcc]
name = {}
for p, op in ops:  # pair each op_code with the nearest global_call_count
    i = bisect.bisect_left(gpos, p)
    cands = [gcc[j] for j in (i - 1, i) if 0 <= j < len(gcc)]
    _, g = min(cands, key=lambda c: abs(c[0] - p))
    name.setdefault(g, op)
dev = pd.read_csv(f"{d}/cpp_device_perf_report.csv", low_memory=False)
dev["OP CODE"] = dev["GLOBAL CALL COUNT"].map(name)
print(f"device rows {len(dev)}, named {dev['OP CODE'].notna().sum()}, host ops {len(name)}")
dev[["GLOBAL CALL COUNT", "OP CODE", "DEVICE ID", "CORE COUNT", "DEVICE KERNEL DURATION [ns]"]].to_csv(out, index=False)
