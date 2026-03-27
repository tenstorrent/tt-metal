# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal test: matmul with L1-sharded activation and DRAM-sharded weights.
Weights go through DRAM interleaved → DRAM WIDTH_SHARDED via interleaved_to_sharded.
"""

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc


def test_matmul_dram_sharded_weights_via_interleaved_to_sharded(device):
    torch.manual_seed(0)

    M, K, N = 32, 8192, 1024
    num_cores = 8
    num_dram_banks = device.dram_grid_size().x

    # Pad N so it's divisible by (32 * num_dram_banks)
    alignment = 32 * num_dram_banks
    N_padded = ((N + alignment - 1) // alignment) * alignment

    # --- torch reference ---
    act_torch = torch.randn(1, 1, M, K, dtype=torch.bfloat16)
    weights_torch = torch.randn(1, 1, K, N, dtype=torch.bfloat16)
    expected = act_torch @ weights_torch

    # --- activation: to device (DRAM interleaved) → L1 WIDTH_SHARDED ---
    act_tt = ttnn.from_torch(act_torch, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device)

    act_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    act_shard_spec = ttnn.ShardSpec(act_shard_grid, (M, K // num_cores), ttnn.ShardOrientation.ROW_MAJOR)
    act_l1_cfg = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, act_shard_spec)
    act_tt = ttnn.to_memory_config(act_tt, act_l1_cfg)

    # --- weights: DRAM interleaved → DRAM WIDTH_SHARDED via interleaved_to_sharded ---
    # The shard grid uses DRAM bank coordinates (0,0)-(N_banks-1, 0).
    # The program factory internally remaps these to DRAM-adjacent compute worker
    # cores for kernel/CB placement, while keeping DRAM bank coords for buffer
    # allocation and shard address generation.
    weights_tt = ttnn.from_torch(weights_torch, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, device=device)

    dram_shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_dram_banks - 1, 0))})
    dram_shard_spec = ttnn.ShardSpec(dram_shard_grid, (K, N_padded // num_dram_banks), ttnn.ShardOrientation.ROW_MAJOR)
    weights_dram_sharded_cfg = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, dram_shard_spec
    )
    weights_tt = ttnn.interleaved_to_sharded(weights_tt, weights_dram_sharded_cfg)

    # --- matmul ---
    program_config = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=K // num_cores // 32 // 4,
        per_core_M=M // 32,
        per_core_N=N // num_cores // 32,
        fused_activation=None,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    out_tt = ttnn.matmul(
        act_tt,
        weights_tt,
        program_config=program_config,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        compute_kernel_config=compute_config,
    )

    # --- validate ---
    out_tt = ttnn.to_memory_config(out_tt, ttnn.L1_MEMORY_CONFIG)
    out_torch = ttnn.to_torch(out_tt)
    assert_with_pcc(expected, out_torch, pcc=0.999)
