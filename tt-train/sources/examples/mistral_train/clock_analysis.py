# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Effective device clock and idle time per step from a Tracy ``ops_perf_results*.csv``.

Tracy converts device cycles to ns at the clock it read when the device was opened (``CHIP_FREQ`` in
``profile_log_device.csv``, 1350 MHz on Blackhole), so if the chip runs slower under load every kernel duration
is reported too short. This script reads the raw cycle counters and host timestamps: step period in cycles over
the host step period gives the clock the chip actually ran at; the sum of kernel cycles against the step span
gives the device idle fraction (host/dispatch gap).

    python clock_analysis.py --warmup 2 [--wall-ms 434.0] 8:evidence/tracy_8layers/reports/*/ops_perf_results*.csv
"""

from __future__ import annotations

import argparse
import csv
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from analyze_tracy import load, split_steps  # noqa: E402

csv.field_size_limit(10**9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="depth:csv")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reported-mhz", type=float, default=1350.0, help="CHIP_FREQ Tracy used for the ns columns")
    p.add_argument("--wall-ms", type=float, default=None, help="unprofiled wall-clock per step at this depth, if known")
    args = p.parse_args()
    for spec in args.runs:
        depth, path = spec.split(":", 1)
        steps = split_steps(load(path))
        print(f"depth {depth}: {len(steps)} steps in the CSV; steady steps from {args.warmup + 1}")
        print(
            "step | period cycles | host period ms | effective GHz | kernel cycles | span cycles | idle in span % | inter-step idle ms@eff"
        )
        effs = []
        for i in range(args.warmup, len(steps) - 1):
            s, nxt = steps[i], steps[i + 1]
            c0, c1 = int(s[0]["DEVICE FW START CYCLE"]), int(nxt[0]["DEVICE FW START CYCLE"])
            h0, h1 = int(s[0]["HOST START TS"]), int(nxt[0]["HOST START TS"])
            period_cycles, host_ms = c1 - c0, (h1 - h0) / 1e6
            eff = period_cycles / (h1 - h0)  # cycles per ns = GHz
            # Kernel time in cycles: Tracy's DEVICE KERNEL DURATION is cycles / reported clock (FW cycles overlap
            # across consecutive ops because of dispatch pipelining, so they are not summed).
            kernel_cycles = int(sum(float(r["DEVICE KERNEL DURATION [ns]"]) for r in s) * args.reported_mhz / 1e3)
            span = int(s[-1]["DEVICE FW END CYCLE"]) - c0
            inter = (c1 - int(s[-1]["DEVICE FW END CYCLE"])) / eff / 1e6
            effs.append(eff)
            print(
                f"{i + 1:4d} | {period_cycles:13d} | {host_ms:13.2f} | {eff:12.3f} | {kernel_cycles:17d} | {span:11d} | {100 * (span - kernel_cycles) / span:13.1f} | {inter:8.2f}"
            )
        if effs:
            eff = sum(effs) / len(effs)
            scale = args.reported_mhz / 1e3 / eff
            print(
                f"mean effective clock {eff * 1e3:.0f} MHz: Tracy's ns columns (converted at {args.reported_mhz:.0f} MHz) are {scale:.3f}x too short"
            )
            if args.wall_ms:
                st = steps[args.warmup]
                span_cycles = int(steps[args.warmup + 1][0]["DEVICE FW START CYCLE"]) - int(
                    st[0]["DEVICE FW START CYCLE"]
                )
                print(
                    f"step period cycles / unprofiled wall {args.wall_ms} ms = {span_cycles / args.wall_ms / 1e6:.3f} GHz"
                )
        # idle-dominated step 1 as the clock control
        s0, s1 = steps[0], steps[1]
        c = int(s1[0]["DEVICE FW START CYCLE"]) - int(s0[0]["DEVICE FW START CYCLE"])
        h = int(s1[0]["HOST START TS"]) - int(s0[0]["HOST START TS"])
        print(f"control, step 1 (mostly idle: compiles and untilize): {c / h:.4f} GHz")


if __name__ == "__main__":
    main()
