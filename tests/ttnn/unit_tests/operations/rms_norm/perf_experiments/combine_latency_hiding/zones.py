# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage zone aggregation from an EXPLICIT profile_log_device.csv path.

`perf_experiments/zone_report.py` reads the shared
`generated/profiler/.logs/profile_log_device.csv`, which parallel sibling
experiments overwrite. This takes a snapshot path so a measurement stays
attributable.

    python3 .../zones.py <snapshot.csv> [run_host_id] [label]
"""
import collections
import csv
import sys

FREQ = 1350.0
path = sys.argv[1]
want = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
label = sys.argv[3] if len(sys.argv) > 3 else ""

lines = open(path).readlines()
rd = list(csv.reader(lines[2:]))
runids = sorted({int(r[7]) for r in rd if len(r) > 7 and r[7].strip().isdigit()})
if want is None:
    want = runids[0]
open_stack = collections.defaultdict(list)
dur = collections.defaultdict(list)
for r in rd:
    if len(r) < 12 or not r[7].strip().isdigit() or int(r[7]) != want:
        continue
    core = (r[1].strip(), r[2].strip())
    risc = r[3].strip()
    t = int(r[5])
    zone = r[10].strip()
    typ = r[11].strip()
    k = (core, risc, zone)
    if typ == "ZONE_START":
        open_stack[k].append(t)
    elif typ == "ZONE_END" and open_stack[k]:
        dur[k].append(t - open_stack[k].pop())
per = collections.defaultdict(lambda: collections.defaultdict(float))
cnt = collections.defaultdict(lambda: collections.defaultdict(int))
for (core, risc, zone), ds in dur.items():
    per[(risc, zone)][core] += sum(ds)
    cnt[(risc, zone)][core] += len(ds)
print(f"# {label}  run={want}  (ids present: {runids})")
print(f"{'RISC':8s} {'zone':22s} {'ncores':>6s} {'inst':>6s} {'ns avg':>10s} {'ns max':>10s}")
out = []
for (risc, zone), m in per.items():
    n = len(m)
    tot = list(m.values())
    inst = sum(cnt[(risc, zone)].values()) / n
    out.append((risc, zone, n, inst, sum(tot) / n / FREQ * 1000, max(tot) / FREQ * 1000))
for risc, zone, n, inst, a, mx in sorted(out, key=lambda z: (z[0], -z[5])):
    print(f"{risc:8s} {zone:22s} {n:6d} {inst:6.1f} {a:10.0f} {mx:10.0f}")
