# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per traced replay: slowest chip's first kernel start -> last kernel end (us), from profile_log_device.csv."""
import csv, sys, statistics
from collections import defaultdict

path = sys.argv[1]
with open(path) as fh:
    head = fh.readline()
    freq = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rd = csv.reader(fh)
    cols = [c.strip() for c in next(rd)]
    ix = {c: i for i, c in enumerate(cols)}
    span = defaultdict(lambda: [float("inf"), 0])  # (trace, counter, device) -> [start, end]
    for r in rd:
        z = r[ix["zone name"]].strip()
        if not z.endswith("-KERNEL"):
            continue
        tid, cnt = r[ix["trace id"]].strip(), r[ix["trace id counter"]].strip()
        if not tid:
            continue
        key = (int(tid), int(cnt), int(r[ix["PCIe slot"]]))
        t = int(r[ix["time[cycles since reset]"]])
        s = span[key]
        if r[ix["type"]].strip() == "ZONE_START":
            s[0] = min(s[0], t)
        else:
            s[1] = max(s[1], t)
per = defaultdict(dict)
for (tid, cnt, dev), (a, b) in span.items():
    per[(tid, cnt)][dev] = (b - a) / freq
by_trace = defaultdict(list)
for (tid, cnt), devs in sorted(per.items()):
    by_trace[tid].append(max(devs.values()))
for tid, v in sorted(by_trace.items()):
    print(f"trace {tid}: n={len(v)} median={statistics.median(v):.1f} us min={min(v):.1f} max={max(v):.1f}")
