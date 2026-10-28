# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device time per ``bench_<name>`` zone of a tracy ops CSV from moe_experts_bench.py (MIMO_PROFILE_ZONES=1): per
zone, the ops on one device grouped by op (kernel source), median and count of DEVICE KERNEL DURATION, and the sum
per iteration. usage: moe_bench_parse.py <ops_perf_results.csv> [device]"""

import statistics
import sys
from collections import defaultdict

import pandas as pd


def main(csv, device=0):
    cols = ["OP CODE", "OP TYPE", "DEVICE ID", "DEVICE KERNEL DURATION [ns]", "DATA MOVEMENT KERNEL SOURCE"]
    df = pd.read_csv(csv, usecols=cols, low_memory=False)
    zone, per = None, defaultdict(lambda: defaultdict(list))
    order = []
    for code, typ, dev, dur, src in df.itertuples(index=False, name=None):
        if typ == "signpost":
            text = str(code).strip("`").replace("TT_SIGNPOST: ", "")
            if text.startswith("MIMO_ZONE_START bench_"):
                zone = text[len("MIMO_ZONE_START bench_") :]
                order.append(zone)
            elif text.startswith("MIMO_ZONE_END bench_"):
                zone = None
            continue
        if zone is None or dev != device or dur != dur:
            continue
        name = str(src).split("/")[-1].strip("'] ") if "GenericOp" in str(code) else str(code)
        per[zone][name].append(dur / 1e3)
    for z in order:
        ops = per[z]
        iters = max((len(v) for v in ops.values()), default=1)
        total = sum(sum(v) for v in ops.values()) / iters
        parts = ", ".join(f"{k} {statistics.median(v):.1f}us x{len(v)}" for k, v in ops.items())
        print(f"MOE_BENCH_DEVICE zone={z} us_per_iter={total:.1f} | {parts}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
