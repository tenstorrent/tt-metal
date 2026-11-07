# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-core kernel windows of one op from a device profile (profile_log_device.csv): when each core's
RISCs start and end, so an op's kernel duration splits into start skew, per-core run time and end skew.

    python core_spread.py <profile_log_device.csv> [run_index (default: last run with >= 100 cores)]
"""
import csv
import sys
from collections import defaultdict

path = sys.argv[1]
with open(path) as f:
    next(f)  # arch / freq header line
    rows = list(csv.DictReader(f, skipinitialspace=True))
runs = defaultdict(lambda: defaultdict(dict))  # run -> (x, y, risc) -> {begin, end}
for r in rows:
    z = r["zone name"]
    if not z.endswith("-KERNEL"):
        continue
    key = (int(r["core_x"]), int(r["core_y"]), r["RISC processor type"])
    runs[int(r["run host ID"])][key][r["type"] if "type" in r else r["zone phase"]] = int(r["time[cycles since reset]"])
ids = [k for k, v in runs.items() if len({(x, y) for x, y, _ in v}) >= 100]
rid = ids[int(sys.argv[2])] if len(sys.argv) > 2 else ids[-1]
w = runs[rid]
by_risc = defaultdict(list)
for (x, y, risc), ph in w.items():
    b, e = ph.get("ZONE_START", ph.get("begin")), ph.get("ZONE_END", ph.get("end"))
    if b is not None and e is not None:
        by_risc[risc].append((b, e, x, y))
t0 = min(b for v in by_risc.values() for b, *_ in v)
t1 = max(e for v in by_risc.values() for _, e, *_ in v)
print(f"run {rid}: op window {(t1 - t0) / 1e3:.1f} k cycles over {len({(x, y) for x, y, _ in w})} cores")
for risc, v in sorted(by_risc.items()):
    d = sorted(e - b for b, e, *_ in v)
    starts = sorted(b - t0 for b, *_ in v)
    ends = sorted(t1 - e for _, e, *_ in v)
    print(
        f"  {risc:7s} run k cycles min {d[0]/1e3:8.1f} med {d[len(d)//2]/1e3:8.1f} max {d[-1]/1e3:8.1f} | "
        f"start after op begin: min {starts[0]/1e3:.1f} max {starts[-1]/1e3:.1f} | end before op end: min {ends[0]/1e3:.1f} max {ends[-1]/1e3:.1f}"
    )
