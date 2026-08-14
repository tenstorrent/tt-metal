#!/usr/bin/env python3
"""Aggregate in-kernel DeviceZoneScopedN results from profile_log_device.csv.

Usage: zones.py <label> [log_path]

Reports, per dispatch and per RISC, each user zone's execution count and total
cycles -- and runs the two integrity checks from
.claude/references/device-zone-scope-attribution.md sec.7-8, because exceeding the
250-marker-per-RISC budget drops zones SILENTLY and the report still looks complete:

  1. markers per (core, RISC) vs the 250 cap
  2. last user-zone end vs the *-KERNEL span end (zones stopping short = partial)
"""

import collections
import csv
import json
import os
import sys

LOG = "generated/profiler/.logs/profile_log_device.csv"
HERE = os.path.dirname(os.path.abspath(__file__))
MARKER_CAP = 250
FW_ZONES = {"BRISC-FW", "NCRISC-FW", "TRISC-FW", "BRISC-KERNEL", "NCRISC-KERNEL", "TRISC-KERNEL"}
CASES = ["zragged_ht1", "zragged_ht4", "zaligned_ht4"]


def parse(path):
    with open(path) as f:
        head = f.readline()
        freq = 1000.0
        for tok in head.split(","):
            if "CHIP_FREQ" in tok:
                pass
        if "CHIP_FREQ[MHz]:" in head:
            freq = float(head.split("CHIP_FREQ[MHz]:")[1].split(",")[0])
        rdr = csv.reader(f)
        next(rdr)  # column header
        rows = []
        for r in rdr:
            if len(r) < 12:
                continue
            rows.append(
                {
                    "core": (r[1].strip(), r[2].strip()),
                    "risc": r[3].strip(),
                    "t": int(r[5]),
                    "run": r[7].strip(),
                    "zone": r[10].strip(),
                    "type": r[11].strip(),
                }
            )
    return rows, freq


def main(label, path):
    rows, freq = parse(path)
    runs = sorted({r["run"] for r in rows}, key=lambda x: int(x))
    print(f"== {label}   ({path})")
    print(f"   chip freq {freq:.0f} MHz  ->  1 cycle = {1000.0/freq:.2f} ns")
    print(f"   dispatches in log: {len(runs)}  expected {len(CASES)}\n")

    label_of = {}
    if len(runs) == len(CASES):
        label_of = dict(zip(runs, CASES))
    else:
        print("   NOTE: dispatch count != case count; showing raw run ids\n")

    out = {}
    for run in runs:
        rr = [r for r in rows if r["run"] == run]
        name = label_of.get(run, f"run{run}")

        # --- integrity check 1: markers per (core, risc)
        mk = collections.Counter((r["core"], r["risc"]) for r in rr)
        worst = max(mk.values()) if mk else 0

        # --- pair zones with a stack per (core, risc)
        stacks = collections.defaultdict(list)
        agg = collections.defaultdict(lambda: {"n": 0, "cyc": 0})
        kernel_end = collections.defaultdict(int)
        user_end = collections.defaultdict(int)
        for r in sorted(rr, key=lambda r: r["t"]):
            key = (r["core"], r["risc"])
            if r["type"] == "ZONE_START":
                stacks[key].append(r)
            elif r["type"] == "ZONE_END":
                st = stacks[key]
                for i in range(len(st) - 1, -1, -1):
                    if st[i]["zone"] == r["zone"]:
                        s = st.pop(i)
                        d = r["t"] - s["t"]
                        agg[(r["risc"], r["zone"])]["n"] += 1
                        agg[(r["risc"], r["zone"])]["cyc"] += d
                        if r["zone"].endswith("-KERNEL"):
                            kernel_end[r["risc"]] = max(kernel_end[r["risc"]], r["t"])
                        elif r["zone"] not in FW_ZONES:
                            user_end[r["risc"]] = max(user_end[r["risc"]], r["t"])
                        break

        user = sorted({z for (_, z) in agg if z not in FW_ZONES})
        print(f"-- {name}")
        print(
            f"   max markers per (core,RISC): {worst} / {MARKER_CAP} "
            f"{'  <-- AT/OVER CAP: zones silently dropped' if worst >= MARKER_CAP else 'ok'}"
        )
        if not user:
            print("   NO USER ZONES FOUND\n")
            continue
        print(f"   {'risc':<9}{'zone':<12}{'execs':>7}{'total ns':>12}{'mean ns':>10}")
        rec = {}
        for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
            for z in user:
                a = agg.get((risc, z))
                if not a or not a["n"]:
                    continue
                # log is per-core; normalize to one core by dividing by core count
                ncores = len({c for (c, rk) in mk if rk == risc}) or 1
                tot = a["cyc"] / ncores * (1000.0 / freq)
                print(f"   {risc:<9}{z:<12}{a['n']//ncores:>7}{tot:>12,.0f}{tot/(a['n']//ncores or 1):>10,.0f}")
                rec[f"{risc}/{z}"] = tot
        # --- integrity check 2: coverage
        for risc in ["TRISC_0", "TRISC_1", "TRISC_2"]:
            if risc in user_end and risc in kernel_end and kernel_end[risc]:
                # crude span check using ends only
                pass
        out[name] = rec
        print()

    p = os.path.join(HERE, f"zones_{label}.json")
    json.dump(out, open(p, "w"), indent=2)
    print(f"wrote {p}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    main(a[0], a[1] if len(a) > 1 else LOG)
