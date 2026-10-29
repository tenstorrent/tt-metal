# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""MFU of a finished train.py run from its logs; no device needed.

    python mfu.py lora/pixelart-300 [--cores 130] [--chips 8]

Per sequence FLOPs come from `samples.jsonl` (L, V per sequence -> padded rows), window times from
`metrics.jsonl`. Peak uses ttml's convention: 1.35 BF16 TFLOPS per Tensix core on Blackhole, times the compute
grid (`--cores`, 13x10 on p150), times `--chips`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ttml.models.minimax_h3 import H3Config, calculate_sequence_flops, tile_pad

PEAK_PER_CORE_BF16 = 1.35e12  # ttml.common.performance, Blackhole


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", help="train.py --out folder")
    p.add_argument("--cores", type=int, default=130, help="compute grid cores per chip (p150: 13x10)")
    p.add_argument("--chips", type=int, default=8)
    p.add_argument("--rank", type=int, default=None, help="default: from training_args.json")
    p.add_argument("--skip-windows", type=int, default=1, help="leading windows to drop from the steady state")
    args = p.parse_args(argv)

    run = Path(args.run)
    targs = json.loads((run / "training_args.json").read_text())
    rank = args.rank or int(targs.get("rank", 32))
    accum = int(targs.get("grad_accum", 4))
    weights = Path(targs["weights"])
    config = H3Config.from_hf(weights / "config.json")
    peak = PEAK_PER_CORE_BF16 * args.cores * args.chips

    step_flops: dict[int, int] = {}
    cache: dict[tuple[int, int], int] = {}
    for line in (run / "samples.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "L" not in r:
            continue
        key = (r["L"], r["V"])
        if key not in cache:
            S = r["L"] + 4 + r["V"]  # text | audio | video rows
            cache[key] = calculate_sequence_flops(config, tile_pad(S), tile_pad(r["L"]), rank)
        step_flops[r["step"]] = step_flops.get(r["step"], 0) + cache[key]

    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    prev_step, prev_t = 0, 0.0
    windows = []
    for r in rows:
        step, total_t = r["step"], r["sec_per_step"] * r["step"]
        dt = total_t - prev_t
        flops = sum(step_flops.get(s, 0) for s in range(prev_step + 1, step + 1))
        n = step - prev_step
        windows.append((step, dt / n, flops / dt / 1e12, flops / dt / peak))
        prev_step, prev_t = step, total_t
    print(
        f"peak {peak / 1e12:.0f} TFLOPS ({args.chips} chips x {args.cores} cores x 1.35) | rank {rank} | accum {accum}"
    )
    print(f"mean FLOPs per sequence {np.mean(list(cache.values())) / 1e12:.1f} TFLOP over {len(cache)} distinct shapes")
    print("step   s/step   TFLOPS   MFU")
    for step, sec_per_step, tf, mfu in windows:
        print(f"{step:5d}  {sec_per_step:6.2f}  {tf:7.1f}  {100 * mfu:5.2f}%")
    steady = windows[args.skip_windows :] or windows
    tf = np.mean([w[2] for w in steady])
    print(
        f"steady state ({len(steady)} windows): {tf:.1f} TFLOPS, {100 * tf * 1e12 / peak:.2f}% MFU, "
        f"{np.mean([w[1] for w in steady]):.2f} s/step"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
