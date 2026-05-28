# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Experimental: try ttnn.layer_norm with width-sharded input + program_config.

Goal: shave device cycles on the 1x128x1024 layernorm case.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

import ttnn

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "reference" / "golden" / "layernorm.pt"


def make_width_sharded_mem_config(shape, core_grid):
    """Width-shard a (1, M, K) tensor across `core_grid` cores along K."""
    n_rows, n_cols = core_grid
    n_cores = n_rows * n_cols
    M, K = shape[-2], shape[-1]
    assert K % n_cores == 0, f"K={K} not divisible by n_cores={n_cores}"
    per_core_K = K // n_cores
    shard_shape = (M, per_core_K)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cols - 1, n_rows - 1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def make_block_sharded_mem_config(shape, core_grid):
    """Block-shard (1, M, K) across (rows, cols) grid: rows split M, cols split K."""
    n_rows, n_cols = core_grid
    M, K = shape[-2], shape[-1]
    assert M % n_rows == 0, f"M={M} not divisible by rows={n_rows}"
    assert K % n_cols == 0, f"K={K} not divisible by cols={n_cols}"
    per_core_M = M // n_rows
    per_core_K = K // n_cols
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(n_cols - 1, n_rows - 1))}),
        (per_core_M, per_core_K),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


def bench_config(label, *, sharded_mem_config, program_config, iters=200, warmup=30, batch=50):
    golden = torch.load(GOLDEN_PATH, weights_only=False)
    x_torch: torch.Tensor = golden["input"]
    weight: torch.Tensor = golden["weight"]
    bias: torch.Tensor = golden["bias"]
    eps: float = float(golden["eps"])
    dim = x_torch.shape[-1]
    ref_out: torch.Tensor = golden["output"]

    device = ttnn.open_device(device_id=0, l1_small_size=16384, trace_region_size=32 * 1024 * 1024)
    try:
        tt_weight = ttnn.from_torch(
            weight.reshape(1, 1, 1, dim),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_bias = ttnn.from_torch(
            bias.reshape(1, 1, 1, dim),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        tt_input_dram = ttnn.from_torch(
            x_torch,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        tt_input = ttnn.to_memory_config(tt_input_dram, sharded_mem_config)

        # Sanity: run once and verify PCC
        out = ttnn.layer_norm(
            tt_input,
            epsilon=eps,
            weight=tt_weight,
            bias=tt_bias,
            compute_kernel_config=compute_kernel_config,
            program_config=program_config,
            memory_config=sharded_mem_config,
        )
        out_torch = ttnn.to_torch(out).to(torch.float32).reshape(ref_out.shape)
        from models.common.utility_functions import comp_pcc

        passing, pcc_msg = comp_pcc(ref_out, out_torch, 0.99)
        print(f"[{label}] PCC: {pcc_msg}")
        ttnn.deallocate(out)

        # warmup
        for _ in range(warmup):
            out = ttnn.layer_norm(
                tt_input,
                epsilon=eps,
                weight=tt_weight,
                bias=tt_bias,
                compute_kernel_config=compute_kernel_config,
                program_config=program_config,
                memory_config=sharded_mem_config,
            )
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)

        # Capture trace
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        outs = []
        for _ in range(batch):
            outs.append(
                ttnn.layer_norm(
                    tt_input,
                    epsilon=eps,
                    weight=tt_weight,
                    bias=tt_bias,
                    compute_kernel_config=compute_kernel_config,
                    program_config=program_config,
                    memory_config=sharded_mem_config,
                )
            )
        ttnn.end_trace_capture(device, tid, cq_id=0)
        for o in outs:
            ttnn.deallocate(o)

        samples_us = []
        for _ in range(iters):
            t0 = time.perf_counter()
            ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(device)
            t1 = time.perf_counter()
            samples_us.append((t1 - t0) * 1e6 / batch)

        mean_us = statistics.mean(samples_us)
        median_us = statistics.median(samples_us)
        min_us = min(samples_us)
        print(f"[{label}] trace: mean={mean_us:.2f}us median={median_us:.2f}us min={min_us:.2f}us")
        ttnn.release_trace(device, tid)
    finally:
        ttnn.close_device(device)


def main():
    shape = (1, 128, 1024)

    # Config A: width-sharded 1x8 — 128 stays whole, K=1024 split across 8 cores (each 128 cols)
    # block_h must equal M_per_core / 32 = 128/32 = 4; block_w = K_per_core / 32 = 128/32 = 4
    print("=== Config A: width-sharded 1x8, block_h=4 block_w=4 subblock_w=4 ===")
    bench_config(
        "A",
        sharded_mem_config=make_width_sharded_mem_config(shape, (1, 8)),
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1),
            subblock_w=4,
            block_h=4,
            block_w=4,
            inplace=False,
        ),
    )

    # Config B: width-sharded 1x16 → 64 cols/core, block_w=2
    print("=== Config B: width-sharded 1x16, block_h=4 block_w=2 subblock_w=2 ===")
    bench_config(
        "B",
        sharded_mem_config=make_width_sharded_mem_config(shape, (1, 16)),
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(16, 1),
            subblock_w=2,
            block_h=4,
            block_w=2,
            inplace=False,
        ),
    )

    # Config C: width-sharded 1x32 → 32 cols/core (1 tile each), block_w=1
    print("=== Config C: width-sharded 1x32, block_h=4 block_w=1 subblock_w=1 ===")
    bench_config(
        "C",
        sharded_mem_config=make_width_sharded_mem_config(shape, (1, 32)),
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(32, 1),
            subblock_w=1,
            block_h=4,
            block_w=1,
            inplace=False,
        ),
    )

    # Config D: block-sharded 4x8 → M=32 (1 row-tile) and K=128 (4 col-tiles) per core
    print("=== Config D: block-sharded 4x8, block_h=1 block_w=4 subblock_w=4 ===")
    bench_config(
        "D",
        sharded_mem_config=make_block_sharded_mem_config(shape, (4, 8)),
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 4),
            subblock_w=4,
            block_h=1,
            block_w=4,
            inplace=False,
        ),
    )

    # Config E: block-sharded 4x4 (only 16 cores) → M=32, K=256
    print("=== Config E: block-sharded 4x4 ===")
    bench_config(
        "E",
        sharded_mem_config=make_block_sharded_mem_config(shape, (4, 4)),
        program_config=ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(4, 4),
            subblock_w=4,
            block_h=1,
            block_w=8,
            inplace=False,
        ),
    )


if __name__ == "__main__":
    main()
