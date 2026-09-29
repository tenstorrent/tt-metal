# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per launch perf counter reduction of a multipass tracy capture.

For every op of the newest (or given) tracy report whose OP CODE contains FILTER: each counter as a fraction of the
same core's ref_cnt, the median over the launched cores and over the active cores (MATH_COUNTER above half a
percent), then the median over the launches after the first. The wall is the op's device kernel duration.
usage: launch_counters.py FILTER [report_dir] [--first N]   (N: keep only the first N matching launches)
Prints one JSON object."""
import csv
import glob
import json
import os
import statistics
import sys
from collections import defaultdict

COUNTER_ID, FULL_REF_ID = 9090, 9091
REPORT = [
    "FPU_COUNTER", "SFPU_COUNTER", "MATH_COUNTER", "UNPACK0_BUSY_THREAD0", "PACKER_BUSY",
    "WAITING_FOR_NONZERO_SEM_0", "WAITING_FOR_NONZERO_SEM_1", "WAITING_FOR_NONZERO_SEM_2",
    "MATH_INSTRN_AVAILABLE", "MATH_INSTRN_STARTED", "WAITING_FOR_SRCA_VALID", "WAITING_FOR_SRCB_VALID",
    "L1_0_UNPACKER_0",
    "L1_0_NOC_RING0_INCOMING_0", "L1_0_NOC_RING0_INCOMING_1", "L1_0_NOC_RING0_OUTGOING_0", "L1_0_NOC_RING0_OUTGOING_1",
    "L1_1_NOC_RING1_INCOMING_0", "L1_1_NOC_RING1_INCOMING_1", "L1_1_NOC_RING1_OUTGOING_0", "L1_1_NOC_RING1_OUTGOING_1",
]


def newest_report():
    base = os.environ.get("TTM", ".") + "/generated/profiler/reports/*/"
    return max(glob.glob(base), key=os.path.getmtime)


def main():
    argv, first = sys.argv[1:], None
    if "--first" in argv:
        i = argv.index("--first")
        first = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    args = argv
    flt = args[0]
    rep = args[1] if len(args) > 1 else newest_report()
    ops_csv = glob.glob(os.path.join(rep, "ops_perf_results*.csv"))[0]
    ops = [r for r in csv.DictReader(open(ops_csv)) if flt in r["OP CODE"]]
    if first:
        ops = ops[:first]
    walls = {int(r["GLOBAL CALL COUNT"]): float(r["DEVICE KERNEL DURATION [ns]"]) / 1000 for r in ops}
    log = os.path.join(rep, "profile_log_device.csv")
    if not os.path.exists(log):
        log = os.environ.get("TTM", ".") + "/generated/profiler/.logs/profile_log_device.csv"
    vals = defaultdict(dict)  # (rid, x, y) -> counter -> (value, ref24)
    full = defaultdict(list)  # (rid, x, y) -> full 32 bit refs
    with open(log) as f:
        f.readline()
        hdr = [h.strip() for h in f.readline().split(",")]
        ix = {h: i for i, h in enumerate(hdr)}
        for line in f:
            p = line.rstrip("\n").split(",", len(hdr) - 1)
            if len(p) < len(hdr):
                continue
            tid = p[ix["timer_id"]].strip()
            if tid not in ("9090", "9091"):
                continue
            try:
                rid = int(p[ix["run host ID"]])
            except ValueError:
                continue
            if rid not in walls:
                continue
            key = (rid, int(p[ix["core_x"]]), int(p[ix["core_y"]]))
            if tid == "9091":
                full[key].append(int(p[ix["data"]]) & 0xFFFFFFFF)
                continue
            meta = json.loads(p[ix["meta data"]].strip().replace(";", ","))
            vals[key][meta["counter type"]] = (float(meta["value"]), float(meta["ref cnt"]))

    def ref_of(key, ref24):
        for r in full.get(key, []):
            if (r & 0xFFFFFF) == int(ref24) & 0xFFFFFF:
                return float(r)
        return ref24

    per_launch = []
    for rid in sorted(walls):
        cores = [k for k in vals if k[0] == rid]
        if not cores:
            continue
        frac = {c: {n: v / ref_of(c, r) for n, (v, r) in vals[c].items() if r > 0} for c in cores}
        active = [c for c in cores if frac[c].get("MATH_COUNTER", 0) > 0.005]
        ent = {"rid": rid, "wall_us": walls[rid], "cores": len(cores), "active": len(active)}
        for scope, cs in (("all", cores), ("act", active)):
            for n in REPORT:
                xs = [frac[c][n] for c in cs if n in frac[c]]
                if xs:
                    ent[f"{scope}:{n}"] = statistics.median(xs)
        per_launch.append(ent)
    timed = per_launch[1:] or per_launch
    out = {"filter": flt, "report": os.path.basename(os.path.normpath(rep)), "launches": len(per_launch)}
    for k in timed[0]:
        if k == "rid":
            continue
        xs = [e[k] for e in timed if k in e]
        out[k] = round(statistics.median(xs), 4)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
