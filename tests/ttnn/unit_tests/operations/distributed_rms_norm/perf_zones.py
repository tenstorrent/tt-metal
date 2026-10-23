"""Summarize distributed_rms_norm device zones from profile_log_device.csv (run after a --profile run).

Per (chip, run): op span = first *-KERNEL start .. last *-KERNEL end on that chip. Per zone: first start / last end
(us from the span start), and per-core mean duration. Prints the shortest chip-program per run (the last-launched
chip, which waits on nobody at the fence) unless --all.
usage: python3 perf_zones.py [csv] [--all]
"""
import csv
import statistics
import sys
from collections import defaultdict

path = next((a for a in sys.argv[1:] if not a.startswith("--")), "generated/profiler/.logs/profile_log_device.csv")
show_all = "--all" in sys.argv
with open(path) as f:
    first = f.readline()
    freq = float(first.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
    rows = list(csv.reader(f))
hdr = [h.strip() for h in rows[0]]
ix = {h: i for i, h in enumerate(hdr)}
progs = defaultdict(lambda: defaultdict(list))  # (chip, run) -> zone -> list of (core, risc, start/end, t)
for r in rows[1:]:
    if len(r) < len(hdr):
        continue
    key = (r[ix["PCIe slot"]], r[ix["run host ID"]])
    progs[key][r[ix["zone name"]]].append(
        (
            (r[ix["core_x"]], r[ix["core_y"]], r[ix["RISC processor type"]]),
            r[ix["type"]],
            int(r[ix["time[cycles since reset]"]]),
        )
    )


def spans(evs):
    open_ = {}
    out = []
    for core, typ, t in sorted(evs, key=lambda e: e[2]):
        if typ == "ZONE_START":
            open_.setdefault(core, []).append(t)
        elif typ == "ZONE_END" and open_.get(core):
            out.append((core, open_[core].pop(), t))
    return out


# one (chip, run host ID) may hold several calls of the same cached program: split them at time gaps
GAP = 40 * freq  # cycles
split = {}
for (chip, run), zones in progs.items():
    if "R_xload" not in zones and "W_store" not in zones:
        continue
    ts = sorted({e[2] for evs in zones.values() for e in evs})
    cuts = [ts[0]] + [b for a, b in zip(ts, ts[1:]) if b - a > GAP]
    for ci, c0 in enumerate(cuts):
        c1 = cuts[ci + 1] if ci + 1 < len(cuts) else float("inf")
        sub = {z: [e for e in evs if c0 <= e[2] < c1] for z, evs in zones.items()}
        split[(chip, f"{run}.{ci}")] = {z: v for z, v in sub.items() if v}
runs = defaultdict(list)
for (chip, run), zones in split.items():
    k = [s for z in ("BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL") for s in spans(zones.get(z, []))]
    if not k:
        continue
    t0, t1 = min(s[1] for s in k), max(s[2] for s in k)
    runs[run].append((t1 - t0, chip, t0, zones))
for run in sorted(runs, key=lambda r: tuple(map(int, r.split(".")))):
    progs_ = sorted(runs[run])
    print(f"run {run}: chip spans us {[round(p[0] / freq, 2) for p in progs_]}")
    for dur, chip, t0, zones in progs_ if show_all else progs_[:1]:
        print(f"  chip {chip}: {dur / freq:.2f} us")
        rows_out = []
        for z in zones:
            sp_all = spans(zones[z])
            # split by occurrence index per core (round r of a per-round zone = occurrence r)
            occ = defaultdict(list)
            seen = defaultdict(int)
            for c, a0, a1 in sorted(sp_all, key=lambda x: x[1]):
                occ[seen[c]].append((c, a0, a1))
                seen[c] += 1
            for o, sp in occ.items():
                rows_out.append((min(x[1] for x in sp), f"{z}#{o}" if len(occ) > 1 else z, sp))
        for _, z, sp in sorted(rows_out):
            st = [(s[1] - t0) / freq for s in sp]
            en = [(s[2] - t0) / freq for s in sp]
            d = [(s[2] - s[1]) / freq for s in sp]
            ncores = len({s[0] for s in sp})
            print(
                f"    {z:18s} n={len(sp):4d} cores={ncores:4d} start[{min(st):6.2f},{max(st):6.2f}] "
                f"end[{min(en):6.2f},{max(en):6.2f}] dur mean {statistics.mean(d):6.2f} max {max(d):6.2f}"
            )
