# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-CORE zone closure from generated/profiler/.logs/profile_log_device.csv.

    python3 .../probes/zone_percore.py [csv] [--risc TRISC_0] [--top N]

zone_breakdown.py reports max-over-cores per zone, which cannot close the
arithmetic (different zones peak on different cores).  This one picks the core
with the largest KERNEL zone on the chosen RISC and prints THAT core's zone
totals, so `sum(stages) vs KERNEL` is a real closure check on one critical-path
core.  Then it does the same for the second-largest, so a root core and a member
core can be compared side by side.
"""
import csv
import sys
from collections import defaultdict

args = [a for a in sys.argv[1:]]
PATH = "generated/profiler/.logs/profile_log_device.csv"
RISC = None
TOPN = 3
HAS = None
rest = []
i = 0
while i < len(args):
    if args[i] == "--risc":
        RISC = args[i + 1]
        i += 2
    elif args[i] == "--top":
        TOPN = int(args[i + 1])
        i += 2
    elif args[i] == "--has":
        # Keep only cores that recorded this zone.  Ranking by KERNEL wall alone
        # surfaces the MEMBER cores of a cross-core combine (their wall is one
        # long cb_wait_front on the root's stat), which hides the ROOT -- the
        # core the critical path actually runs on.  `--has compute_root_fused`
        # is how you get the root's own closure.
        HAS = args[i + 1]
        i += 2
    elif args[i].endswith(".csv"):
        PATH = args[i]
        i += 1
    else:
        rest.append(args[i])
        i += 1

with open(PATH) as f:
    lines = f.readlines()
freq = float(lines[0].split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rdr = csv.reader(lines[1:])
next(rdr)

ev = defaultdict(list)
order = []
for r in rdr:
    if len(r) < 13:
        continue
    key = (r[7].strip(), r[3].strip(), (r[1].strip(), r[2].strip()), r[10].strip())
    if key[0] not in order:
        order.append(key[0])
    ev[key].append((r[11].strip(), int(r[5])))

want = rest or order
for run in want:
    if run not in order:
        continue
    # (risc, core) -> zone -> [cycles, n]
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for (r_, risc, core, zone), evs in ev.items():
        if r_ != run or (RISC and risc != RISC):
            continue
        start = None
        for typ, cyc in sorted(evs, key=lambda t: t[1]):
            if typ == "ZONE_START":
                start = cyc
            elif typ == "ZONE_END" and start is not None:
                per[(risc, core)][zone][0] += cyc - start
                per[(risc, core)][zone][1] += 1
                start = None
    if HAS:
        per = {k: v for k, v in per.items() if any(HAS in z for z in v)}
    ranked = sorted(per.items(), key=lambda kv: -max((v[0] for z, v in kv[1].items() if "KERNEL" in z), default=0))
    print(f"\n=== run host ID {run}: top {TOPN} cores by KERNEL zone ===")
    for (risc, core), zones in ranked[:TOPN]:
        kern = max((v[0] for z, v in zones.items() if "KERNEL" in z), default=0)
        stages = {z: v for z, v in zones.items() if "KERNEL" not in z and "-FW" not in z}
        tot = sum(v[0] for v in stages.values())
        print(
            f"\n-- core {core} {risc}: KERNEL {kern/freq*1000:.0f} ns, stages sum {tot/freq*1000:.0f} ns, gap {(kern-tot)/freq*1000:.0f} ns"
        )
        for z, v in sorted(stages.items(), key=lambda kv: -kv[1][0]):
            print(f"     {v[0]/freq*1000:10.0f} ns  n={v[1]:<3d} {z}")
