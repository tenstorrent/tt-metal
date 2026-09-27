#!/usr/bin/env python3
# usage: b2b_launch_gap.py generated/profiler/.logs/profile_log_device.csv [N]  (after a MIMO_FL_B2B / MIMO_KREF_B2B run)
# last N launches in the raw device log: kernel-start-to-kernel-start period, kernel span, gap (last kernel end -> next first kernel start), us
import collections
import csv
import statistics as st
import sys

f = open(sys.argv[1])
next(f)
rd = csv.DictReader(f, skipinitialspace=True)
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
ks = collections.defaultdict(list)
ke = collections.defaultdict(list)
for r in rd:
    if r["zone name"].strip().endswith("-KERNEL"):
        (ks if r["type"].strip() == "ZONE_START" else ke)[int(r["run host ID"])].append(
            int(r["time[cycles since reset]"])
        )
ids = sorted(ks)[-N:]
per = [(min(ks[b]) - min(ks[a])) / 1350 for a, b in zip(ids, ids[1:])]
gap = [(min(ks[b]) - max(ke[a])) / 1350 for a, b in zip(ids, ids[1:])]
ker = [(max(ke[a]) - min(ks[a])) / 1350 for a in ids[:-1]]
print(f"period {st.mean(per):8.1f}  kernel {st.mean(ker):8.1f}  gap mean {st.mean(gap):5.2f} max {max(gap):5.2f}")
