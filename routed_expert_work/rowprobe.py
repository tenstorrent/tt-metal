"""Aggregate a named NCRISC zone per physical grid row, from a device-profiler CSV.

Usage: python3 rowprobe.py [--zone NAME] <csv> [<csv> ...]

Default zone is `pf_publish`.  NOTE: `pf_publish` and `pf_x_reserve` both contain a
`cb_reserve_back`, so they measure CONSUMER progress; `pf_x_barrier` and `pf_publish`
at GU_CHUNKS=1 are the barrier-only (true read-return) measurements.  See WORKLOG 8.8.
"""
import csv, sys
from collections import defaultdict

FREQ_MHZ = 1350.0


def rows(path, zone="pf_publish"):
    with open(path) as f:
        lines = f.read().splitlines()
    hdr = [h.strip() for h in lines[1].split(",")]
    open_ = {}
    tot = defaultdict(float)
    cnt = defaultdict(int)
    for rec in csv.DictReader(lines[2:], fieldnames=hdr):
        if (rec.get("zone name") or "").strip() != zone:
            continue
        key = (
            int(rec["core_x"]),
            int(rec["core_y"]),
            rec[" RISC processor type"].strip()
            if " RISC processor type" in rec
            else rec["RISC processor type"].strip(),
        )
        t = int(rec["time[cycles since reset]"])
        typ = rec["type"].strip()
        if typ == "ZONE_START":
            open_[key] = t
        elif typ == "ZONE_END" and key in open_:
            tot[key] += (t - open_.pop(key)) / FREQ_MHZ
            cnt[key] += 1
    per_row = defaultdict(list)
    for (x, y, _), v in tot.items():
        per_row[y].append(v)
    return {y: sum(v) / len(v) for y, v in sorted(per_row.items())}, {y: len(v) for y, v in sorted(per_row.items())}


args = sys.argv[1:]
ZONE = "pf_publish"
if args and args[0] == "--zone":
    ZONE = args[1]
    args = args[2:]

for path in args:
    prof, ncores = rows(path, ZONE)
    print(f"\n=== {path}  zone={ZONE}")
    print(f" phys y   logical row   mean {ZONE} (us)   cores")
    for i, (y, v) in enumerate(prof.items()):
        print(f"   {y:4d}   {i:11d}   {v:21.1f}   {ncores[y]:5d}")
    if prof:
        vals = list(prof.values())
        print(f"  spread: min {min(vals):.1f}  max {max(vals):.1f}  ratio {max(vals)/min(vals):.2f}x")
        order = sorted(prof, key=prof.get, reverse=True)
        print(f"  slowest phys rows: {order[:3]}   fastest: {order[-3:]}")
