# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-stage zone breakdown from generated/profiler/.logs/profile_log_device.csv.

Usage:  python3 .../probes/zone_breakdown.py [run_host_id ...]

Reads the MaybeDeviceZoneScope markers (perf_instrumentation.hpp) and prints, per
`run host ID` (== one op launch, in launch order), the summed ZONE duration per
zone name, split by RISC.  Two columns matter:

  sum_ns/core   the zone's total cycles on the BUSIEST core, in ns -- the
                critical-path contribution (a stage on ONE root core is what
                every member waits behind).
  max_ns/core   the largest single-core total for that zone.

Zone budget is 125 entries per RISC per launch, so a shape with many row-blocks
truncates the tail; `n` (entry count) tells you when that happened.
"""
import csv
import sys
from collections import defaultdict

PATH = (
    sys.argv.pop(1)
    if len(sys.argv) > 1 and sys.argv[1].endswith(".csv")
    else "generated/profiler/.logs/profile_log_device.csv"
)

rows = []
with open(PATH) as f:
    lines = f.readlines()
freq = float(lines[0].split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rdr = csv.reader(lines[1:])
hdr = next(rdr)
for r in rdr:
    if len(r) < 13:
        continue
    rows.append(r)

# (run_id, risc, core, zone) -> list of (type, cycles)
ev = defaultdict(list)
order = []
for r in rows:
    core = (r[1].strip(), r[2].strip())
    risc = r[3].strip()
    cyc = int(r[5])
    run = r[7].strip()
    zone = r[10].strip()
    typ = r[11].strip()
    if run not in order:
        order.append(run)
    ev[(run, risc, core, zone)].append((typ, cyc))

want = sys.argv[1:] or order
for run in want:
    if run not in order:
        continue
    print(f"\n=== run host ID {run} (launch #{order.index(run)}) ===")
    # zone -> core -> total cycles, count
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (r_, risc, core, zone), evs in ev.items():
        if r_ != run:
            continue
        start = None
        for typ, cyc in sorted(evs, key=lambda t: t[1]):
            if typ == "ZONE_START":
                start = cyc
            elif typ == "ZONE_END" and start is not None:
                agg[(risc, zone)][core][0] += cyc - start
                agg[(risc, zone)][core][1] += 1
                start = None
    out = []
    for (risc, zone), percore in agg.items():
        tot = max(v[0] for v in percore.values())
        n = max(v[1] for v in percore.values())
        out.append((tot / freq * 1000.0, n, risc, zone, len(percore)))
    out.sort(reverse=True)
    print(f"{'max_ns/core':>12} {'n':>5} {'risc':>7} {'cores':>6}  zone")
    for ns, n, risc, zone, ncores in out:
        print(f"{ns:12.0f} {n:5d} {risc:>7} {ncores:6d}  {zone}")
