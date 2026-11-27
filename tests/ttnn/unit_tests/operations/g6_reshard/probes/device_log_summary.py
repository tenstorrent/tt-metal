# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only summary of a Tracy device log for g6_reshard runs (no device needed).

Per op (run host ID, in log order): the kernel span over all cores, per RISC the
max/p50 of the per-core kernel duration, issue and wait zones, and the slowest cores.
All times in cycles.

    python tests/ttnn/unit_tests/operations/g6_reshard/probes/device_log_summary.py \
        generated/profiler/reports/<ts>/profile_log_device.csv [labels.txt]
"""

import collections
import csv
import statistics
import sys


def main(path, labels=None):
    zones = collections.defaultdict(dict)  # (run, x, y, risc, zone) -> {start, end}
    with open(path) as f:
        next(f)
        for row in csv.DictReader(f, skipinitialspace=True):
            name = row["zone name"]
            if not (name.endswith("-KERNEL") or name.startswith("RESHARD")):
                continue
            key = (int(row["run host ID"]), int(row["core_x"]), int(row["core_y"]), row["RISC processor type"], name)
            zones[key]["start" if row["type"] == "ZONE_START" else "end"] = int(row["time[cycles since reset]"])

    runs = sorted({k[0] for k in zones})
    for i, run in enumerate(runs):
        label = labels[i] if labels and i < len(labels) else str(run)
        kern = {k: v for k, v in zones.items() if k[0] == run and k[4].endswith("-KERNEL")}
        start = min(v["start"] for v in kern.values())
        end = max(v["end"] for v in kern.values())
        print(f"== {label}: span {end - start} cyc, launch skew {max(v['start'] for v in kern.values()) - start}")
        for risc in ("NCRISC", "BRISC"):
            durs = {
                (k[1], k[2]): v["end"] - v["start"]
                for k, v in kern.items()
                if k[3] == risc and "start" in v and "end" in v
            }
            if not durs:
                continue
            ends = {(k[1], k[2]): v["end"] - start for k, v in kern.items() if k[3] == risc}
            issue = [
                v["end"] - v["start"]
                for k, v in zones.items()
                if k[0] == run and k[3] == risc and k[4] == "RESHARD-ISSUE"
            ]
            wait = [
                v["end"] - v["start"]
                for k, v in zones.items()
                if k[0] == run and k[3] == risc and k[4] == "RESHARD-WAIT"
            ]
            slow = sorted(ends.items(), key=lambda kv: -kv[1])[:4]
            print(
                f"  {risc:6s} kernel max {max(durs.values())} p50 {int(statistics.median(durs.values()))}"
                f" | issue max {max(issue, default=0)} p50 {int(statistics.median(issue)) if issue else 0}"
                f" | wait max {max(wait, default=0)} p50 {int(statistics.median(wait)) if wait else 0}"
                f" | latest end (core: cyc from op start) {slow}"
            )


if __name__ == "__main__":
    labels = open(sys.argv[2]).read().split() if len(sys.argv) > 2 else None
    main(sys.argv[1], labels)
