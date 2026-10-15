#!/usr/bin/env python3
"""Scratch, not for merge. dev_parse.py <tracy out dir> <arm> <case> [n_warm=2]: per gate op (warm-up ops dropped) and
core, the RISC kernel zones, the GMG-* probe zones and the perf counter records of the device log; prints medians over
ops x cores in cycles:
  "ZONE <arm> <case> <n> <key>=<median> ..." (times relative to the core's first kernel zone start)
  "CNT <arm> <case> <n> <counter>=<median value> ... ref=<median ref cnt>"."""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

out, arm, case = sys.argv[1:4]
n_warm = int(sys.argv[4]) if len(sys.argv) > 4 else 2
fs = glob.glob(os.path.join(out, "**", "profile_log_device.csv"), recursive=True)
if not fs:
    print(f"ZONE {arm} {case} NOLOG")
    sys.exit(0)
zones = defaultdict(dict)  # (op, core) -> {(risc, zone): [start, end]}
cnts = defaultdict(dict)  # (op, core) -> {name: (value, ref)}
with open(fs[0]) as f:
    next(f)
    for r in csv.DictReader(f, skipinitialspace=True):
        try:
            op = int(r["run host ID"])
            core = (int(r["core_x"]), int(r["core_y"]))
            t = int(r["time[cycles since reset]"])
        except (KeyError, ValueError, TypeError):
            continue
        md = r.get("meta data") or ""
        if "counter type" in md:
            try:
                d = json.loads(md.replace(";", ","))
                cnts[(op, core)][d["counter type"]] = (float(d.get("value", 0)), float(d.get("ref cnt", 0)))
            except (ValueError, KeyError):
                pass
            continue
        name, typ, risc = r.get("zone name", ""), r.get("type", ""), r.get("RISC processor type", "")
        if not name.endswith("-KERNEL") and not name.startswith("GMG-"):
            continue
        z = zones[(op, core)].setdefault((risc, name), [None, None])
        if typ == "ZONE_START" and z[0] is None:
            z[0] = t
        elif typ == "ZONE_END" and z[1] is None:
            z[1] = t
ops = sorted({k[0] for k in zones if any(n.startswith("TRISC") for (_, n) in zones[k])} | {k[0] for k in cnts})
keep = set(ops[n_warm:])


def med(v):
    return statistics.median(v) if v else float("nan")


acc = defaultdict(list)
for (op, core), zz in zones.items():
    if op not in keep:
        continue
    ks = [v[0] for (rk, n), v in zz.items() if n.endswith("-KERNEL") and v[0] is not None]
    ke = [v[1] for (rk, n), v in zz.items() if n.endswith("-KERNEL") and v[1] is not None]
    if not ks or not ke:
        continue
    t0 = min(ks)
    acc["kspan"].append(max(ke) - t0)
    for (rk, n), (s, e) in zz.items():
        if s is None or e is None:
            continue
        tag = rk.replace("TRISC_", "T").replace("BRISC", "B").replace("NCRISC", "N")
        nm = n.replace("TRISC-KERNEL", "K").replace("BRISC-KERNEL", "K").replace("NCRISC-KERNEL", "K").replace("GMG-", "")
        acc[f"{tag}.{nm}.s"].append(s - t0)
        acc[f"{tag}.{nm}.e"].append(e - t0)
        acc[f"{tag}.{nm}.d"].append(e - s)
n = len(acc["kspan"])
if n:
    print(f"ZONE {arm} {case} {n} " + " ".join(f"{k}={med(v):.0f}" for k, v in sorted(acc.items())))
cacc = defaultdict(list)
refs = []
for (op, core), cc in cnts.items():
    if op not in keep:
        continue
    for name, (v, ref) in cc.items():
        cacc[name].append(v)
        refs.append(ref)
if cacc:
    print(f"CNT {arm} {case} {len(refs) // max(1, len(cacc))} ref={med(refs):.0f} " + " ".join(f"{k}={med(v):.1f}" for k, v in sorted(cacc.items())))
