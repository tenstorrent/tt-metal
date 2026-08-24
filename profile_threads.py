"""Per-RISC kernel-zone spans from the device profiler CSV.

Each of the five baby RISCs emits a *-KERNEL zone per program invocation, so pairing
ZONE_START with ZONE_END gives how long that thread was inside the kernel. That is the
question 'which thread is the critical path' -- unlike a sum zone, which measures a region's
duration on whichever thread reads it and so ranks without attributing.
"""
import csv, sys, statistics
from collections import defaultdict


def spans(path, want_run=None):
    per = defaultdict(list)  # (risc) -> [durations in us]
    open_at = {}
    freq = None
    with open(path) as f:
        head = f.readline()
        freq = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) < 12:
                continue
            cx, cy, risc, tid, t, _d, run = row[1], row[2], row[3], row[4], row[5], row[6], row[7]
            zone, kind = row[10], row[11]
            if not zone.endswith("-KERNEL"):
                continue
            if want_run is not None and run.strip() != str(want_run):
                continue
            key = (cx, cy, risc, run)
            if kind == "ZONE_START":
                open_at[key] = int(t)
            elif kind == "ZONE_END" and key in open_at:
                per[risc].append((int(t) - open_at.pop(key)) / freq)
    return per


if __name__ == "__main__":
    path = sys.argv[1]
    run = sys.argv[2] if len(sys.argv) > 2 else None
    per = spans(path, run)
    if not per:
        print("  no KERNEL zones for that run")
        sys.exit(1)
    peak = max(max(v) for v in per.values())
    print(f"  {'risc':8s} {'n':>5s} {'median':>9s} {'max':>9s}  share of the slowest thread")
    for risc in ("BRISC", "NCRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
        v = per.get(risc)
        if not v:
            continue
        bar = "#" * int(40 * max(v) / peak)
        print(f"  {risc:8s} {len(v):5d} {statistics.median(v):8.1f}us {max(v):8.1f}us  {bar}")
