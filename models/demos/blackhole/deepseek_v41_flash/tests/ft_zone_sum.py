"""ft_zone_sum.py <profile_log_device.csv> [slot ...] [--last N] : fuse-tail all-reduce ops (zone C_ALL): per op and chip, for every zone: earliest start, latest end (us from the chip's first event of the op),
median / max duration over its cores. Prints the last N ops (default 2) of each requested chip slot (default: all chips, brief)."""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
args = sys.argv[2:]
last = 2
if "--last" in args:
    i = args.index("--last")
    last = int(args[i + 1])
    del args[i : i + 2]
slots = set(args)
freq = 1350.0
ev = defaultdict(list)  # (slot, run) -> events
open_ = {}
rows = csv.reader(open(path))
next(rows)
next(rows)
for r in rows:
    if len(r) < 12:
        continue
    slot, cx, cy, risc, _tid, cyc, _d, run, _t, _tc, zone, typ = r[:12]
    if slots and slot not in slots:
        continue
    if zone.endswith("-FW") or zone.startswith("PROFILER"):
        continue
    key = (slot, run, cx, cy, risc, zone)
    if typ == "ZONE_START":
        open_.setdefault(key, []).append(int(cyc))
    elif typ == "ZONE_END" and open_.get(key):
        s = open_[key].pop(0)
        ev[(slot, int(run))].append(((cx, cy), risc, zone, s, int(cyc)))
by_slot = defaultdict(list)
for (slot, run), es in ev.items():
    if any(e[2] == "C_ALL" for e in es):
        by_slot[slot].append(run)
for slot in sorted(by_slot, key=int):
    for run in sorted(by_slot[slot])[-last:]:
        es = ev[(slot, run)]
        t0 = min(e[3] for e in es)
        print(f"== chip {slot} op {run}: span {(max(e[4] for e in es) - t0) / freq:.1f} us")
        if not slots:
            continue
        byz = defaultdict(list)
        for e in es:
            byz[(e[1], e[2])].append(e)
        for (risc, zone), l in sorted(byz.items(), key=lambda kv: min(e[3] for e in kv[1])):
            d = [(e[4] - e[3]) / freq for e in l]
            print(
                f"  {risc:7s} {zone:14s} n={len(l):3d} start {(min(e[3] for e in l) - t0) / freq:6.2f} end {(max(e[4] for e in l) - t0) / freq:6.2f} med {statistics.median(d):6.2f} max {max(d):6.2f}"
            )
