# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Validates the DRAM-sharded matmul path when weights are resharded on-device
via ttnn.interleaved_to_sharded (DRAM interleaved -> DRAM WIDTH_SHARDED),
vs. the baseline of weights loaded directly as DRAM WIDTH_SHARDED.

This exercises the interleaved_to_sharded -> DRAM destination path that
enables MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig for tt-mlir
compiler flows, which cannot pre-allocate sharded weights host-side.
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import torch2tt_tensor
from tests.ttnn.utils_for_testing import assert_with_pcc


N_ITERS = 200
RESHARD_ITERS = 50
PCC = 0.99


def pad_to_dram_banks(num, num_banks):
    lcm = 32 * num_banks
    rem = num % lcm
    return num if rem == 0 else num + (lcm - rem)


# Shapes sorted by weight size (K * N). Larger entries would OOM the staging CB
# in interleaved_to_sharded without the L1-budgeted chunking in the reader/writer
# kernels; they exercise the chunked path with progressively smaller chunks.
@pytest.mark.parametrize(
    "M, K, N, grid",
    [
        (32, 8192, 1024, (8, 1)),
        (32, 8192, 1280, (8, 1)),
        (32, 8192, 2048, (8, 1)),
        (32, 8192, 4096, (8, 1)),
        (32, 16384, 2048, (8, 1)),
        (32, 16384, 4096, (8, 1)),
        (32, 32768, 1024, (8, 1)),
        (32, 32768, 2048, (8, 1)),
    ],
)
def test_matmul_dram_sharded_via_reshard(device, M, K, N, grid):
    num_banks = device.dram_grid_size().x
    N_padded = pad_to_dram_banks(N, num_banks)
    num_cores = grid[0] * grid[1]

    in0_block_w = K // num_cores // 32
    per_core_M = M // 32
    per_core_N = N // num_cores // 32

    interleaved_dram = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
    sharded_l1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1)

    compute_cfg = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    torch.manual_seed(0)
    in0_raw = torch.randn([1, 1, M, K]).bfloat16().float()
    in1_raw = torch.randn([1, 1, K, N]).bfloat16().float()
    expected = in0_raw @ in1_raw

    # in0: interleaved DRAM -> L1 WIDTH_SHARDED (shared by both variants)
    in0_dram = torch2tt_tensor(in0_raw, device, tt_memory_config=interleaved_dram, tt_dtype=ttnn.bfloat16)
    in0_l1 = ttnn.interleaved_to_sharded(
        in0_dram,
        grid,
        [M, in0_block_w * 32],
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.ShardOrientation.ROW_MAJOR,
    )

    # DRAM WIDTH_SHARDED memory config for weights. Shard grid uses DRAM bank
    # logical coords; the program factory remaps them to compute worker cores
    # for kernel/CB placement.
    dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    in1_shard_spec = ttnn.ShardSpec(dram_grid, [K, N_padded // num_banks], ttnn.ShardOrientation.ROW_MAJOR)
    in1_dram_sharded_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, in1_shard_spec
    )

    prog_cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w // 4,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        fused_activation=None,
    )

    def run(weights):
        return ttnn.matmul(
            in0_l1,
            weights,
            program_config=prog_cfg,
            memory_config=sharded_l1,
            dtype=ttnn.bfloat16,
            compute_kernel_config=compute_cfg,
        )

    def run_variant(weights, label):
        out = run(weights)
        out_torch = ttnn.to_torch(ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG))
        assert_with_pcc(expected, out_torch, pcc=PCC)
        ttnn.deallocate(out)

        t0 = time.perf_counter()
        for _ in range(N_ITERS):
            out = run(weights)
        out.cpu()
        us = (time.perf_counter() - t0) / N_ITERS * 1e6
        logger.info(f"  {label}: {us:6.1f} us/iter")
        return us

    # Variant 1: weights loaded directly as DRAM WIDTH_SHARDED (baseline)
    w_direct = torch2tt_tensor(in1_raw, device, tt_memory_config=in1_dram_sharded_cfg, tt_dtype=ttnn.bfloat8_b)
    direct_us = run_variant(w_direct, "dram_sharded (direct)")
    ttnn.deallocate(w_direct)

    # Variant 2: weights loaded as DRAM interleaved, resharded via interleaved_to_sharded
    w_interleaved = torch2tt_tensor(in1_raw, device, tt_memory_config=interleaved_dram, tt_dtype=ttnn.bfloat8_b)
    w_reshard = ttnn.interleaved_to_sharded(w_interleaved, in1_dram_sharded_cfg)
    reshard_us = run_variant(w_reshard, "dram_reshrd (via interleaved_to_sharded)")
    ttnn.deallocate(w_reshard)

    # Variant 3: same reshard but through ttnn.to_memory_config. Per PR #41413,
    # to_memory_config reroutes to ttnn::copy when the i2s staging CB would
    # exceed L1 — so for large shards this path compares chunked i2s (var 2) vs
    # copy-based reshard (var 3).
    w_tmc = ttnn.to_memory_config(w_interleaved, in1_dram_sharded_cfg)
    tmc_us = run_variant(w_tmc, "dram_tomem_cfg (via to_memory_config)")
    ttnn.deallocate(w_tmc)

    # Measure the reshard step itself (matmul perf above is identical across
    # variants since all three produce the same DRAM-sharded weights layout).
    def time_reshard(reshard_fn, label):
        out = reshard_fn()
        out.cpu()
        ttnn.deallocate(out)
        t0 = time.perf_counter()
        for _ in range(RESHARD_ITERS):
            out = reshard_fn()
            out.cpu()
            ttnn.deallocate(out)
        us = (time.perf_counter() - t0) / RESHARD_ITERS * 1e6
        logger.info(f"  {label}: {us:7.1f} us/reshard")
        return us

    i2s_us = time_reshard(lambda: ttnn.interleaved_to_sharded(w_interleaved, in1_dram_sharded_cfg), "reshard via i2s")
    copy_us = time_reshard(lambda: ttnn.to_memory_config(w_interleaved, in1_dram_sharded_cfg), "reshard via to_mem_cfg")

    ttnn.deallocate(w_interleaved)

    logger.info(
        f"  M={M} K={K} N={N}: "
        f"matmul reshard/direct={reshard_us/direct_us:.2f}x tmc/direct={tmc_us/direct_us:.2f}x | "
        f"reshard-cost copy/i2s={copy_us/i2s_us:.2f}x"
    )
