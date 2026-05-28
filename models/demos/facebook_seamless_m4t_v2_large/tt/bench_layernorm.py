# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Steady-state micro-benchmark for the SeamlessM4T-v2 TTNN LayerNorm block.

Runs N synchronized iterations after a warmup pass and reports mean/median
us/op. Used by the optimization-worker to compare configs.
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.demos.facebook_seamless_m4t_v2_large.tt.layernorm import LayerNorm

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "layernorm.pt"


def bench(iters: int = 200, warmup: int = 20) -> None:
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    weight: torch.Tensor = golden["weight"]
    bias: torch.Tensor = golden["bias"]
    eps: float = float(golden["eps"])
    dim = x_torch.shape[-1]

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        tt_layernorm = LayerNorm(
            device=device,
            dim=dim,
            weight=weight,
            bias=bias,
            eps=eps,
            weight_dtype=ttnn.bfloat16,
        )
        tt_input = ttnn.from_torch(
            x_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # warmup
        for _ in range(warmup):
            out = tt_layernorm(tt_input)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)

        # measure
        samples_us = []
        for _ in range(iters):
            t0 = time.perf_counter()
            out = tt_layernorm(tt_input)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            samples_us.append((t1 - t0) * 1e6)
            ttnn.deallocate(out)

        mean_us = statistics.mean(samples_us)
        median_us = statistics.median(samples_us)
        min_us = min(samples_us)
        p90_us = statistics.quantiles(samples_us, n=10)[8]
        print(
            f"layernorm bench (iters={iters}): mean={mean_us:.2f}us median={median_us:.2f}us min={min_us:.2f}us p90={p90_us:.2f}us"
        )
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()
    bench(iters=args.iters, warmup=args.warmup)
