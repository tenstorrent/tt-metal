#!/usr/bin/env python3
"""Per op: device span, per-RISC kernel durations and sum-zone totals per core, from profile_log_device.csv."""
import csv, glob, os, statistics, sys
from collections import defaultdict
out, arm, rnd = sys.argv[1:4]
detail = len(sys.argv) > 4
p = os.path.join(out, ".logs", "profile_log_device.csv")
if not os.path.exists(p):
    c = glob.glob(os.path.join(out, "**", "profile_log_device.csv"), recursive=True)
    if not c:
        print(f"NOLOG {arm} r{rnd}"); sys.exit(0)
    p = c[0]
f = open(p)
hdr = f.readline()
freq = float(hdr.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
rd = csv.reader(f)
cols = [c.strip() for c in next(rd)]
ix = {c: i for i, c in enumerate(cols)}
start = defaultdict(dict); end = defaultdict(dict); sums = defaultdict(lambda: defaultdict(int))
for r in rd:
    if len(r) < len(cols):
        continue
    run = r[ix["run host ID"]].strip(); risc = r[ix["RISC processor type"]].strip()
    core = (int(r[ix["core_x"]]), int(r[ix["core_y"]])); zone = r[ix["zone name"]].strip(); typ = r[ix["type"]].strip()
    t = int(r[ix["time[cycles since reset]"]])
    if typ == "ZONE_TOTAL":
        sums[run][(core, risc, zone)] += int(r[ix["data"]]); continue
    if not zone.endswith("-KERNEL"):
        continue
    k = (core, risc)
    if typ == "ZONE_START":
        start[run][k] = min(t, start[run].get(k, t))
    elif typ == "ZONE_END":
        end[run][k] = max(t, end[run].get(k, t))
i = 0
for run in sorted(start, key=lambda x: int(x) if x.isdigit() else 0):
    ks = [k for k in start[run] if k in end[run]]
    if not ks:
        continue
    t0 = min(start[run][k] for k in ks); t1 = max(end[run][k] for k in ks)
    ns = (t1 - t0) * 1000.0 / freq
    if ns < 3000:
        continue
    cores = sorted({k[0] for k in ks})
    line = f"RES {arm} r{rnd} op{i} run{run} span {ns:.0f} ns {t1 - t0} cyc cores {len(cores)}"
    per = defaultdict(list)
    for k in ks:
        per[k[1]].append(end[run][k] - start[run][k])
    for risc in sorted(per):
        v = per[risc]
        line += f" | {risc} med {statistics.median(v):.0f} max {max(v)}"
    print(line)
    if sums[run]:
        agg = defaultdict(list)
        for (core, risc, zone), v in sums[run].items():
            agg[(risc, zone)].append(v)
        print("SUM " + f"{arm} r{rnd} op{i} " + " | ".join(
            f"{risc}:{zone} med {statistics.median(v):.0f} min {min(v)} max {max(v)} n {len(v)}" for (risc, zone), v in sorted(agg.items())))
        if detail:
            for core in cores:
                row = [f"{risc}:{zone}={v}" for (c, risc, zone), v in sorted(sums[run].items()) if c == core]
                kd = [f"{k[1]}={end[run][k]-start[run][k]}@{start[run][k]-t0}" for k in ks if k[0] == core]
                print(f"CORE {arm} r{rnd} op{i} {core[0]},{core[1]} " + " ".join(sorted(kd) + row))
    i += 1
