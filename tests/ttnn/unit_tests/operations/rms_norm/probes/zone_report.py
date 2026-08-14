# Per-stage device-zone report for rms_norm (or any op instrumented with
# MaybeDeviceZoneScope).  Reads generated/profiler/.logs/profile_log_device.csv
# and prints, per (RISC, zone):
#
#   execs   total device-ns summed over cores/execs  (sum / ncores = ns per core)
#   percore ns per core (the number to rank stages by)
#   max/p50 across cores -> a concentrated stage (root serialization) shows up here
#
# Plus the two coverage checks the marker budget makes mandatory:
#   * markers per (core, RISC) vs the 250 cap  (exhaustion is SILENT)
#   * last user-zone end vs the *-KERNEL span  (a partial profile invents a
#     dominant stage and looks complete doing it)
import csv
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "generated/profiler/.logs/profile_log_device.csv"
NAME_FILTER = sys.argv[2] if len(sys.argv) > 2 else None

rows = []
with open(path) as f:
    hdr = f.readline()
    freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    r = csv.reader(f)
    cols = next(r)
    for row in r:
        if len(row) < 12:
            continue
        rows.append(row)

# columns: 0 slot,1 x,2 y,3 risc,4 timer_id,5 cycles,6 data,7 runid,...,10 zone,11 type
open_stack = defaultdict(list)
dur = defaultdict(list)  # (risc, zone) -> [ns]
percore = defaultdict(lambda: defaultdict(float))  # (risc,zone) -> core -> ns
markers = defaultdict(int)  # (core,risc) -> count
span = {}  # (core,risc) -> [start,end] of *-FW/*-KERNEL
last_user_end = defaultdict(float)
kern_span = {}

for row in rows:
    x, y, risc, cyc, zone, typ = row[1], row[2], row[3], int(row[5]), row[10], row[11]
    core = (x, y)
    key = (core, risc)
    markers[key] += 1
    if typ == "ZONE_START":
        open_stack[(key, zone)].append(cyc)
    elif typ == "ZONE_END":
        st = open_stack[(key, zone)]
        if not st:
            continue
        s = st.pop()
        ns = (cyc - s) * 1000.0 / freq
        if zone.endswith("-KERNEL"):
            kern_span.setdefault(key, [1 << 62, 0])
            kern_span[key][0] = min(kern_span[key][0], s)
            kern_span[key][1] = max(kern_span[key][1], cyc)
        elif zone.endswith("-FW"):
            pass
        else:
            dur[(risc, zone)].append(ns)
            percore[(risc, zone)][core] += ns
            last_user_end[key] = max(last_user_end[key], cyc)

print(f"clock {freq} MHz;  {len(percore)} user zones")
print(f"{'RISC':<8}{'zone':<26}{'execs':>6}{'cores':>6}{'ns/core':>10}{'p50':>9}{'max':>9}{'max/p50':>8}")
items = []
for (risc, zone), pc in percore.items():
    if NAME_FILTER and NAME_FILTER not in zone:
        continue
    vals = sorted(pc.values())
    p50 = vals[len(vals) // 2]
    items.append((sum(vals) / len(vals), risc, zone, len(dur[(risc, zone)]), len(vals), p50, vals[-1]))
for mean, risc, zone, execs, ncores, p50, mx in sorted(items, reverse=True):
    print(f"{risc:<8}{zone:<26}{execs:>6}{ncores:>6}{mean:>10.0f}{p50:>9.0f}{mx:>9.0f}{mx/max(p50,1):>8.2f}")

print("\n-- coverage --")
cap = max(markers.values())
n_at_cap = sum(1 for v in markers.values() if v >= 240)
print(f"max markers per (core,RISC) = {cap}  (cap 250);  {n_at_cap} of {len(markers)} at/near cap")
worst = []
for key, sp in kern_span.items():
    if key in last_user_end:
        frac = (last_user_end[key] - sp[0]) / max(sp[1] - sp[0], 1)
        worst.append((frac, key, (sp[1] - sp[0]) * 1000.0 / freq))
worst.sort()
if worst:
    print("last user zone end as fraction of KERNEL span (min 5):")
    for frac, key, span_ns in worst[:5]:
        print(f"   {key} frac={frac:.2f} kernel_span={span_ns:.0f} ns")
    print(f"   median frac = {worst[len(worst)//2][0]:.2f}")
    print("\nKERNEL span per RISC (ns): max vs p50 across cores")
    byrisc = defaultdict(list)
    for key, sp in kern_span.items():
        byrisc[key[1]].append((sp[1] - sp[0]) * 1000.0 / freq)
    for risc, v in sorted(byrisc.items()):
        v.sort()
        print(
            f"   {risc:<8} p50={v[len(v)//2]:>9.0f}  max={v[-1]:>9.0f}  max/p50={v[-1]/max(v[len(v)//2],1):.2f}  cores={len(v)}"
        )
