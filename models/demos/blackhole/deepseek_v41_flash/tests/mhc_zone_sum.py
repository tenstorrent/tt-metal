"""mhc_zone_sum.py <profile_log_device.csv> [device] : per op (run host id) of one device: for every zone, over its cores: earliest start, latest end (us from the op's first event)
and the median / max zone duration. Ops are printed in launch order; kernel-level zones (*-KERNEL) are included, firmware zones are not."""
import csv
import statistics
import sys
from collections import defaultdict

path = sys.argv[1]
BRIEF = "--brief" in sys.argv
args = [a for a in sys.argv[1:] if a != "--brief"]
dev = args[1] if len(args) > 1 else None
freq = 1350.0
ev = defaultdict(list)  # op id -> [(core, risc, zone, t_start, t_end)]
open_ = {}
rows = csv.reader(open(path))
next(rows)
hdr = next(rows)
for r in rows:
    if len(r) < 12:
        continue
    slot, cx, cy, risc, _tid, cyc, _d, run, _t, _tc, zone, typ = r[:12]
    if dev is None:
        dev = slot
    if slot != dev:
        continue
    if zone.endswith("-FW") or zone.startswith("PROFILER") or zone.endswith("-KERNEL") and False:
        continue
    key = (run, cx, cy, risc, zone)
    if typ == "ZONE_START":
        open_.setdefault(key, []).append(int(cyc))
    elif typ == "ZONE_END" and open_.get(key):
        s = open_[key].pop(0)
        ev[int(run)].append(((cx, cy), risc, zone, s, int(cyc)))
for run in sorted(ev):
    es = ev[run]
    names = {e[2] for e in es}
    t0 = min(e[3] for e in es)
    tot = (max(e[4] for e in es) - t0) / freq
    kind = (
        "probe"
        if "PB_ALL" in names
        else "mixes"
        if "PW_ALL" in names
        else "expand"
        if "EW_ALL" in names or "ER_ALL" in names
        else "other"
    )
    if kind == "other":
        continue
    print(f"== op {run} {kind}: span {tot:.1f} us, cores {len({e[0] for e in es})}")
    if BRIEF:
        continue
    byz = defaultdict(list)
    for e in es:
        byz[(e[1], e[2])].append(e)
    for (risc, zone), l in sorted(byz.items(), key=lambda kv: min(e[3] for e in kv[1])):
        st = min(e[3] for e in l)
        en = max(e[4] for e in l)
        d = [(e[4] - e[3]) / freq for e in l]
        print(
            f"   {risc:8s} {zone:22s} n={len(l):3d} start {(st - t0) / freq:6.1f}  end {(en - t0) / freq:6.1f}  dur med {statistics.median(d):6.1f} max {max(d):6.1f}"
        )
