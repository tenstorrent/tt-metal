# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-role busy time of the flat expert's last launch (a --profile run with MIMO_SE_ZONES=1): for each zone@RISC,
the per-core sum of its durations over the launch, mean and max over the cores that have it, as a share of the
launch's kernel span. On the compute TRISCs a zone includes the thread's waits; the unpack thread (TRISC0) waits are
separate zones (W_X / W_W: x / weights on the gate/up cores, SE_H_WAIT: h on the down cores).

    python models/demos/mimo_v2_d_p/tests/perf/analyze_role_busy.py generated/profiler/.logs/profile_log_device.csv
"""

import collections
import csv
import sys

ZONES = [
    ("SE_GU_MM", "TRISC_0"),
    ("W_X", "TRISC_0"),
    ("W_W", "TRISC_0"),
    ("SE_GU_ACT_PACK", "TRISC_2"),
    ("SE_DOWN", "TRISC_0"),
    ("SE_H_WAIT", "TRISC_0"),
    ("TZ_BLK", "TRISC_0"),
    ("XMC_CRED", "NCRISC"),
    ("XRD_FULL", "BRISC"),
]


def main(path):
    f = open(path)
    next(f)
    rows = list(csv.DictReader(f, skipinitialspace=True))
    last = max(int(r["run host ID"]) for r in rows)
    rows = [r for r in rows if int(r["run host ID"]) == last]
    ks = [int(r["time[cycles since reset]"]) for r in rows if r["zone name"].strip().endswith("-KERNEL")]
    span = (max(ks) - min(ks)) / 1350.0
    start, tot = {}, collections.defaultdict(lambda: collections.defaultdict(float))
    for r in rows:
        z, risc = r["zone name"].strip(), r["RISC processor type"].strip()
        core = (r["core_x"], r["core_y"])
        key = (z, risc, core)
        t = int(r["time[cycles since reset]"])
        if r["type"].strip() == "ZONE_START":
            start[key] = t
        elif key in start:
            tot[(z, risc)][core] += (t - start.pop(key)) / 1350.0
    print(f"launch {last}: kernel span {span:.1f} us")
    for z, risc in ZONES:
        v = list(tot[(z, risc)].values())
        if v:
            m = sum(v) / len(v)
            print(
                f"  {z:>14s}@{risc:<8s} {len(v):3d} cores  mean {m:8.1f} us ({m / span * 100:5.1f}%)  max {max(v):8.1f} us"
            )


if __name__ == "__main__":
    main(sys.argv[1])
