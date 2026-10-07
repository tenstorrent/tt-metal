#!/usr/bin/env python3
"""Fallback when `python -m tracy -r` post-processing asserts (it does when the device profiler DRAM buffer overflowed in the EAGER warm window: eager ops
recorded by the host have no device row; AssertionError 'Device data missing: Op N not present in cpp_device_perf_report.csv').
Builds an ops_perf-style CSV of the TRACE REPLAY rows only (the columns tracy_report.py reads) from the two small files the run already wrote:
  <dir>/.logs/cpp_device_perf_report.csv  (device rows: OP NAME, core count, kernel duration, op-to-op latency, trace id/session; per device in trace order)
  <dir>/.logs/tracy_ops_data.csv          (host op messages: JSON per op with shapes/dtypes/attributes, in dispatch order)
The ops captured into the trace are the LAST N host ops of each device (nothing but replays runs after the capture); they are matched 1:1 in order to the device
rows of every replay session, and the OP NAME must agree at every position (asserted).
  python tracy_join.py <profiler dir> <out.csv>"""
import csv
import json
import re
import sys
from collections import defaultdict

d, out = sys.argv[1], sys.argv[2]
rows = defaultdict(lambda: defaultdict(list))  # (trace, session) -> dev -> rows
for r in csv.DictReader(open(f"{d}/.logs/cpp_device_perf_report.csv")):
    if r["METAL TRACE ID"]:
        rows[(r["METAL TRACE ID"], r["METAL TRACE REPLAY SESSION ID"])][r["DEVICE ID"]].append(r)
head = re.compile(r'`TT_DNN_DEVICE_OP: "(\w+)", (\d+), (\d+), (\w+), (\d+)( ->)?')
host = defaultdict(list)  # dev -> [(gcc, hash, code)] in dispatch order
shape_of = (
    {}
)  # op hash -> parsed json of the first (program-cache miss) record: cached ops carry only a compact header, same hash = same shapes/attributes
buf = None
for line in open(f"{d}/.logs/tracy_ops_data.csv"):
    if buf is not None:
        buf.append(line)
        if line.startswith("}`;"):
            shape_of.setdefault(cur[1], json.loads("".join(buf)[: -len(line)] + "}"))
            buf = None
        continue
    m = head.match(line)
    if m:
        cur = (m.group(1), m.group(2))
        host[m.group(3)].append((int(m.group(5)), m.group(2), m.group(1)))
        if m.group(6):
            buf = []
print(
    "host ops per device",
    {k: len(v) for k, v in list(host.items())[:3]},
    "distinct hashes with json",
    len(shape_of),
    file=sys.stderr,
)


def shp(t):
    sh = t["shape"]
    return sh["Y"], sh["X"]


cols = ["OP CODE", "GLOBAL CALL COUNT", "DEVICE ID", "CORE COUNT", "DEVICE KERNEL DURATION [ns]", "OP TO OP LATENCY [ns]", "METAL TRACE ID", "METAL TRACE REPLAY SESSION ID", "ATTRIBUTES",
        "INPUT_0_Y_PAD[LOGICAL]", "INPUT_0_X_PAD[LOGICAL]", "INPUT_0_DATATYPE", "INPUT_1_Y_PAD[LOGICAL]", "INPUT_1_X_PAD[LOGICAL]", "INPUT_1_DATATYPE",
        "OUTPUT_0_Y_PAD[LOGICAL]", "OUTPUT_0_X_PAD[LOGICAL]", "OUTPUT_0_DATATYPE"]  # fmt: skip
w = csv.DictWriter(open(out, "w", newline=""), fieldnames=cols)
w.writeheader()
n_ok = 0
for key, devs in rows.items():
    for dev, rs in devs.items():
        rs.sort(key=lambda r: int(r["GLOBAL CALL COUNT"]))
        hs = host[dev][-len(rs) :]
        assert len(hs) == len(rs), (dev, len(hs), len(rs))
        for r, (gcc, h, code) in zip(rs, hs):
            j = shape_of[h]
            assert code == r["OP NAME"], (dev, gcc, code, r["OP NAME"])
            row = {"OP CODE": j["op_code"], "GLOBAL CALL COUNT": r["GLOBAL CALL COUNT"], "DEVICE ID": dev, "CORE COUNT": r["CORE COUNT"],
                   "DEVICE KERNEL DURATION [ns]": r["DEVICE KERNEL DURATION [ns]"], "OP TO OP LATENCY [ns]": r["OP TO OP LATENCY [ns]"],
                   "METAL TRACE ID": key[0], "METAL TRACE REPLAY SESSION ID": key[1],
                   "ATTRIBUTES": str(j.get("attributes", {}))}  # fmt: skip
            for k, t in enumerate(j.get("input_tensors", [])[:2]):
                row[f"INPUT_{k}_Y_PAD[LOGICAL]"], row[f"INPUT_{k}_X_PAD[LOGICAL]"] = shp(t)
                row[f"INPUT_{k}_DATATYPE"] = t["dtype"]
            if j.get("output_tensors"):
                t = j["output_tensors"][0]
                row["OUTPUT_0_Y_PAD[LOGICAL]"], row["OUTPUT_0_X_PAD[LOGICAL]"] = shp(t)
                row["OUTPUT_0_DATATYPE"] = t["dtype"]
            w.writerow(row)
            n_ok += 1
print("rows written", n_ok, file=sys.stderr)
