"""Per-stage zone breakdown of one GenericOp in a `--profile` report dir (perf tournament, Perf 1).

usage: python3 zone_report.py <report_dir> [op_index ...]      (default: every GenericOp row)

For each op: DEVICE KERNEL DURATION, then per RISC the *-KERNEL span (p50 / max across cores -> the
`/perf-measure` balance check), and per zone name: executions per core, p50 / max across cores of the per-core
SUM of that zone (us), and the number of cores that recorded it. Ends with the two silent-truncation checks of
.claude/references/device-zone-scope-attribution.md §7: markers per (core, RISC) against the 250 cap, and the
last user zone's end vs the *-KERNEL end (coverage) on the busiest core of each RISC.
"""
import csv
import os
import statistics
import sys
from collections import defaultdict

CLK = 1350.0  # MHz; only scales the printed us, ratios are clock-free
MARKER_CAP = 250

rep = sys.argv[1]
ops_csv = [f for f in os.listdir(rep) if f.startswith("ops_perf")][0]
ops = [r for r in csv.DictReader(open(f"{rep}/{ops_csv}")) if r["OP CODE"] == "GenericOpDeviceOperation"]
want = [int(a) for a in sys.argv[2:]] or list(range(len(ops)))

lines = open(f"{rep}/profile_log_device.csv").read().splitlines()[1:]
hdr = [h.strip() for h in lines[0].split(",")]
ix = {h: i for i, h in enumerate(hdr)}
rows_by_run = defaultdict(list)
for line in lines[1:]:
    f = [x.strip() for x in line.split(",")]
    if len(f) < len(hdr) - 1:
        continue
    rows_by_run[f[ix["run host ID"]]].append(f)


def p50(v):
    return statistics.median(v) if v else 0.0


for k in want:
    op = ops[k]
    run_id = op["GLOBAL CALL COUNT"]
    print(f"\n=== op {k}: {float(op['DEVICE KERNEL DURATION [ns]'])/1000:.2f} us, cores {op['CORE COUNT']} ===")
    starts = {}
    zsum = defaultdict(float)  # (core, risc, zone) -> summed us
    zcnt = defaultdict(int)
    zend = {}  # (core, risc, zone) -> last end cycle
    kspan = {}  # (core, risc) -> (start, end)
    markers = defaultdict(int)
    for f in rows_by_run[run_id]:
        core = (int(f[ix["core_x"]]), int(f[ix["core_y"]]))
        risc = f[ix["RISC processor type"]]
        z = f[ix["zone name"]]
        t = int(f[ix["time[cycles since reset]"]])
        typ = f[ix["type"]]
        key = (core, risc, z)
        if z.endswith("-KERNEL"):
            if typ == "ZONE_START":
                kspan[(core, risc)] = (t, kspan.get((core, risc), (0, 0))[1])
            elif typ == "ZONE_END":
                kspan[(core, risc)] = (kspan.get((core, risc), (t, 0))[0], t)
            continue
        if z.endswith("-FW") or z.startswith("PROFILER"):
            continue
        markers[(core, risc)] += 1
        if typ == "ZONE_START":
            starts[key] = t
        elif typ == "ZONE_END" and key in starts:
            zsum[key] += (t - starts.pop(key)) / CLK
            zcnt[key] += 1
            zend[key] = t
    riscs = sorted({r for (_, r) in kspan})
    for risc in riscs:
        spans = {c: (e - s) / CLK for (c, r), (s, e) in kspan.items() if r == risc and e > s}
        if not spans:
            continue
        smax = max(spans.values())
        busiest = max(spans, key=spans.get)
        print(
            f"-- {risc}: KERNEL span p50 {p50(list(spans.values())):.2f} us, max {smax:.2f} us "
            f"(max/p50 {smax/max(p50(list(spans.values())),1e-9):.2f}), cores {len(spans)}, busiest {busiest}"
        )
        zones = sorted({z for (c, r, z) in zsum if r == risc})
        rows = []
        for z in zones:
            per_core = [zsum[(c, r, zz)] for (c, r, zz) in zsum if r == risc and zz == z]
            execs = [zcnt[(c, r, zz)] for (c, r, zz) in zcnt if r == risc and zz == z]
            on_busiest = zsum.get((busiest, risc, z), 0.0)
            rows.append((z, p50(execs), p50(per_core), max(per_core), on_busiest, len(per_core)))
        rows.sort(key=lambda r: -r[2])
        print(f"   {'zone':18s} {'exec':>5s} {'p50 us':>8s} {'max us':>8s} {'busiest':>8s} {'cores':>5s}")
        for z, ex, med, mx, ob, n in rows:
            print(f"   {z:18s} {ex:5.0f} {med:8.2f} {mx:8.2f} {ob:8.2f} {n:5d}")
        # truncation / coverage checks on the busiest core
        m = markers.get((busiest, risc), 0)
        ks, ke = kspan[(busiest, risc)]
        last_end = max([e for (c, r, z), e in zend.items() if c == busiest and r == risc], default=ks)
        cov = (last_end - ks) / max(ke - ks, 1)
        flag = " <-- AT CAP: zones truncated" if m >= MARKER_CAP - 2 else ""
        print(f"   markers on busiest core: {m}/{MARKER_CAP}{flag}; last zone end at {cov*100:.0f}% of KERNEL span")
        over = [c for (c, r), m in markers.items() if r == risc and m >= MARKER_CAP - 2]
        if over:
            print(f"   {len(over)} core(s) at the marker cap on {risc}")
