"""Per-stage zone breakdown of the latest mhc_pre program in a device profile log (perf tournament, Perf 1).

usage: python3 zone_report.py [profile_log_device.csv] [--core X,Y ...]

Picks the newest profile_log_device.csv under generated/ (or the given one) and the newest run host ID that
recorded user zones. Prints: the wall (latest *-KERNEL end), per RISC the KERNEL span p50 / max across cores, per
zone name executions per core and p50 / max of the per-core SUM (us); the timeline (start, dur) of every zone on
the critical core (the one whose KERNEL ends last) and any --core given; and the silent-truncation checks of
.claude/references/device-zone-scope-attribution.md §7 (markers per (core, RISC) vs the 250 cap, last user zone end
vs the KERNEL end).
"""
import collections
import glob
import os
import statistics
import sys

CLK = 1350.0  # MHz (BH); us = cycles / CLK
CAP = 250

args = sys.argv[1:]
extra = []
if "--core" in args:
    i = args.index("--core")
    extra = [tuple(int(v) for v in c.split(",")) for c in args[i + 1 :]]
    args = args[:i]
f = (
    args[0]
    if args
    else max(
        glob.glob("generated/profiler/.logs/profile_log_device.csv")
        + glob.glob("generated/dev*/profiler/.logs/profile_log_device.csv"),
        key=os.path.getmtime,
    )
)
lines = open(f).read().splitlines()
hdr = [h.strip() for h in lines[1].split(",")]
ix = {h: i for i, h in enumerate(hdr)}
rows = []
for line in lines[2:]:
    v = [x.strip() for x in line.split(",")]
    if len(v) >= len(hdr) - 1 and v[ix["run host ID"]]:
        rows.append(v)
user = lambda z: not (z.endswith("-KERNEL") or z.endswith("-FW") or z.startswith("PROFILER") or z == "")
rid = max(int(r[ix["run host ID"]]) for r in rows if user(r[ix["zone name"]]))
rows = [r for r in rows if int(r[ix["run host ID"]]) == rid]
ev = collections.defaultdict(list)
for r in rows:
    key = (int(r[ix["core_x"]]), int(r[ix["core_y"]]), r[ix["RISC processor type"]], r[ix["zone name"]])
    ev[key].append((int(r[ix["time[cycles since reset]"]]), r[ix["type"]]))
t0 = min(t for k, v in ev.items() if k[3].endswith("-KERNEL") for t, ty in v if ty == "ZONE_START")


def spans(v):
    v = sorted(v)
    s = [t for t, ty in v if ty == "ZONE_START"]
    e = [t for t, ty in v if ty == "ZONE_END"]
    return [((a - t0) / CLK, (b - a) / CLK) for a, b in zip(s, e)]


kspan, zsum, zcnt, zlast, markers = {}, collections.defaultdict(float), {}, {}, collections.defaultdict(int)
for k, v in ev.items():
    c, risc, z = (k[0], k[1]), k[2], k[3]
    sp = spans(v)
    if z.endswith("-KERNEL"):
        kspan[(c, risc)] = (sp[0][0], sp[-1][0] + sp[-1][1])
    elif user(z):
        zsum[(c, risc, z)] = sum(d for _, d in sp)
        zcnt[(c, risc, z)] = len(sp)
        zlast[(c, risc, z)] = max(s + d for s, d in sp)
        markers[(c, risc)] += len(v)
wall = max(e for _, e in kspan.values())
crit = max(kspan, key=lambda k: kspan[k][1])[0]
print(f"{f} run {rid}: wall {wall:.1f} us, critical core {crit}")
p50 = lambda v: statistics.median(v) if v else 0.0
for risc in sorted({r for _, r in kspan}):
    durs = {c: e - s for (c, r), (s, e) in kspan.items() if r == risc}
    print(f"-- {risc}: KERNEL p50 {p50(list(durs.values())):.1f} max {max(durs.values()):.1f} us ({len(durs)} cores)")
    for z in sorted(
        {z for (_, r, z) in zsum if r == risc},
        key=lambda z: -p50([v for (c, r, zz), v in zsum.items() if r == risc and zz == z]),
    ):
        vals = [v for (c, r, zz), v in zsum.items() if r == risc and zz == z]
        ex = [v for (c, r, zz), v in zcnt.items() if r == risc and zz == z]
        print(
            f"   {z:16s} exec {p50(ex):4.0f}  p50 {p50(vals):7.2f}  max {max(vals):7.2f}  crit {zsum.get((crit, risc, z), 0):7.2f}  cores {len(vals)}"
        )
    worst = (
        max((m, c) for (c, r), m in markers.items() if r == risc) if any(r == risc for (_, r) in markers) else (0, None)
    )
    if worst[1] is not None:
        c = worst[1]
        last = max(v for (cc, r, z), v in zlast.items() if cc == c and r == risc)
        ks, ke = kspan[(c, risc)]
        flag = "  <-- AT CAP" if worst[0] >= CAP - 2 else ""
        print(
            f"   max markers {worst[0]}/{CAP} on {c}{flag}; last zone end at {100 * (last - ks) / max(ke - ks, 1e-9):.0f}% of its KERNEL span"
        )
byrow = collections.defaultdict(list)
for (c, r), (s, e) in kspan.items():
    byrow[c[1]].append(e)
print("per core-row latest KERNEL end:", " ".join(f"y{y}:{max(v):.0f}" for y, v in sorted(byrow.items())))
for c in [crit] + extra:
    print(f"== timeline core {c} (start/dur us) ==")
    for risc in ("NCRISC", "BRISC", "TRISC_0", "TRISC_1", "TRISC_2"):
        if (c, risc) not in kspan:
            continue
        items = []
        for (cx, cy, r, z), v in ev.items():
            if (cx, cy) == c and r == risc and user(z):
                items += [(s, d, z) for s, d in spans(v)]
        items.sort()
        ks, ke = kspan[(c, risc)]
        print(f" {risc} [{ks:.1f}..{ke:.1f}] " + " ".join(f"{z}@{s:.1f}+{d:.1f}" for s, d, z in items if d >= 0.3))
