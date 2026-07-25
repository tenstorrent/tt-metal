#!/usr/bin/env python3
"""Attribute Dispatch writer-kernel time from device-profiler zones.

Pairs ZONE_START/ZONE_END per (device, core, risc, zone, invocation) in profile_log_device.csv and
reports per-invocation duration for each zone: median and max across the 32 devices.

The four DISPATCH_* zones sit inside the writer kernel, so they attribute DEVICE KERNEL time -- they
cannot explain the op2op gap, which is time outside any kernel.

init_barrier_wait + exit_barrier_wait = cross-device stall (waiting for the slowest of 32 peers).
Everything else in the kernel = residual (real drain/compute).

Usage: zone_split.py <profile_log_device.csv> [zone_prefix]
"""

import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else "DISPATCH_"

with open(path) as fh:
    first = fh.readline()
    freq_mhz = 1350.0
    for tok in first.split(","):
        if "CHIP_FREQ" in tok:
            try:
                freq_mhz = float(tok.split(":")[1].strip())
            except Exception:
                pass
    rows = list(csv.DictReader(fh, skipinitialspace=True))

print(f"# {path}")
print(f"# chip freq {freq_mhz} MHz, {len(rows)} zone rows\n")

# (device, core, risc, zone, invocation) -> {start, end}
pend = defaultdict(dict)
durs = defaultdict(list)      # zone -> [us]
per_inv = defaultdict(list)   # (zone, invocation) -> [us across devices]
kernel_tot = defaultdict(list)

for r in rows:
    zone = (r.get("zone name") or "").strip()
    typ = (r.get("type") or "").strip()
    try:
        t = int((r.get("time[cycles since reset]") or "0").strip())
    except ValueError:
        continue
    key = (
        (r.get("PCIe slot") or "").strip(),
        (r.get("core_x") or "").strip() + "," + (r.get("core_y") or "").strip(),
        (r.get("RISC processor type") or "").strip(),
        zone,
        (r.get("run host ID") or "").strip(),
    )
    if typ == "ZONE_START":
        pend[key]["s"] = t
    elif typ == "ZONE_END":
        pend[key]["e"] = t

for key, se in pend.items():
    if "s" not in se or "e" not in se:
        continue
    us = (se["e"] - se["s"]) / freq_mhz
    dev, core, risc, zone, inv = key
    if zone.startswith(prefix):
        durs[zone].append(us)
        per_inv[(zone, inv)].append(us)
    if zone in ("BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"):
        kernel_tot[zone].append(us)

if not durs:
    print(f"NO ZONES matching prefix {prefix!r} found.")
    zs = sorted({(r.get('zone name') or '').strip() for r in rows})
    print(f"zone names present ({len(zs)}): {zs[:25]}")
    sys.exit(1)

print(f"{'zone':<34} {'n(dev*core*inv)':>15} {'median us':>11} {'max us':>11} {'sum us':>11}")
for z in sorted(durs, key=lambda z: -statistics.median(durs[z])):
    v = durs[z]
    print(f"{z:<34} {len(v):>15} {statistics.median(v):>11.1f} {max(v):>11.1f} {sum(v):>11.1f}")

# Per-invocation view: for each invocation, the MAX across devices is the critical path.
print(f"\n{'zone':<34} {'invocations':>12} {'median-of-max us':>18} {'max-of-max us':>15}")
by_zone = defaultdict(list)
for (z, inv), v in per_inv.items():
    by_zone[z].append(max(v))
for z in sorted(by_zone, key=lambda z: -statistics.median(by_zone[z])):
    v = by_zone[z]
    print(f"{z:<34} {len(v):>12} {statistics.median(v):>18.1f} {max(v):>15.1f}")

stall = sum(sum(durs.get(z, [])) for z in ("DISPATCH_init_barrier_wait", "DISPATCH_exit_barrier_wait"))
fabric = sum(durs.get("DISPATCH_fabric_open", []))
final = sum(durs.get("DISPATCH_final_full_barrier", []))
tracked = stall + fabric + final
print("\n---- attribution (sum over all device*core*invocation samples) ----")
print(f"init+exit barrier stall : {stall:12.1f} us")
print(f"fabric_open             : {fabric:12.1f} us")
print(f"final_full_barrier      : {final:12.1f} us")
print(f"total tracked in zones  : {tracked:12.1f} us")
for z, v in kernel_tot.items():
    print(f"{z:<24}: {sum(v):12.1f} us  (median {statistics.median(v):.1f})")
    if sum(v):
        print(f"  -> barrier stall is {100*stall/sum(v):.1f}% of {z} total")
