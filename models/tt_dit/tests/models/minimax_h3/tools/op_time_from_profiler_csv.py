# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-call op time from a Tracy ops CSV: max over devices of DEVICE KERNEL DURATION, matched by
per-device call position (GLOBAL CALL COUNT is per-device and must not be used as the join key).

    python op_time_from_profiler_csv.py <ops_perf_results.csv> [op-code substring, default RingJointSDPA]
"""
import csv
import sys
from collections import defaultdict


def main() -> None:
    path = sys.argv[1]
    needle = sys.argv[2] if len(sys.argv) > 2 else "RingJointSDPA"
    by_dev = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if needle in row["OP CODE"] and row.get("OP TYPE") != "signpost":
                by_dev[row["DEVICE ID"]].append(int(float(row["DEVICE KERNEL DURATION [ns]"])))
    if not by_dev:
        sys.exit(f"no rows matching {needle!r} in {path}")
    n = min(len(v) for v in by_dev.values())
    print(f"{needle}: {len(by_dev)} devices, {n} calls each")
    per_call = []
    for i in range(n):
        vals = [v[i] for v in by_dev.values()]
        per_call.append(max(vals))
        print(f"call {i}: max {max(vals) / 1e6:8.2f} ms   min-over-devices {min(vals) / 1e6:8.2f} ms")
    print(f"steady-state (min over calls of max-over-devices): {min(per_call) / 1e6:.2f} ms")


if __name__ == "__main__":
    main()
