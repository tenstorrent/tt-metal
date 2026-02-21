# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Tests for shared expert operation.

Fuses: Activation Mcast + Gate/Up Matmul + Gather + GatedLocalReduce
       + Mcast1 + Mcast2 + Down Proj Matmul + Add + Output Gather

Run:
    pytest models/demos/deepseek_v3_b1/tests/unit_tests/test_shared_expert.py -v -s
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.blitz_decode_weights import GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC, BlitzDecodeWeights
from models.demos.deepseek_v3_b1.fused_ops.down_proj.op import DownProj
from models.demos.deepseek_v3_b1.fused_ops.shared_expert.op import SharedExpertOp


@pytest.mark.parametrize(
    "K_gate, N_per_core, weights_dtype",
    [
        pytest.param(
            7168, 64, ttnn.bfloat8_b, marks=pytest.mark.skip_post_commit
        ),  # Full MLP: K_gate=7168, K_down=256, N=7168
        pytest.param(
            2048, 64, ttnn.bfloat8_b, marks=pytest.mark.skip_post_commit
        ),  # Mid: K_gate=2048, K_down=256, N=7168
        pytest.param(
            1024, 32, ttnn.bfloat8_b, marks=pytest.mark.skip_post_commit
        ),  # Small: K_gate=1024, K_down=256, N=3584
        (7168, 64, ttnn.bfloat4_b),  # bfloat4 weights
    ],
)
def test_shared_expert(device, K_gate, N_per_core, weights_dtype):
    """Test shared expert: activation → gate/up matmul → gated reduce → down proj + bias."""

    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    M = 1
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    k_parallel = cfg.k_parallel
    n_parallel = cfg.n_parallel
    K_down = n_parallel * 32  # 256
    N = N_per_core * DownProj.NUM_MATMUL_CORES
    k_per_core = (K_gate // 32) // k_parallel
    assert k_per_core * k_parallel * 32 == K_gate, "K_gate must be divisible by 32 * k_parallel"

    use_bdw = K_gate == cfg.gate_proj_shape[0] and weights_dtype == ttnn.bfloat4_b

    logger.info("=" * 70)
    logger.info("Testing Shared Expert:")
    logger.info(f"  K_gate={K_gate}, K_down={K_down}, N={N}, N_per_core={N_per_core}")
    logger.info(f"  k_parallel={k_parallel}, n_parallel={n_parallel}, k_per_core={k_per_core}")
    logger.info(f"  weights_dtype={weights_dtype}")
    logger.info("=" * 70)

    # Tile definitions
    a_tile = ttnn.Tile([M, 32])
    b_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([M, 32])

    # Core grids (activation / bias / output placement)
    mcast_gather_core = DownProj.MCAST_GATHER_CORE
    sender_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(mcast_gather_core, mcast_gather_core)])

    # ========================================================================
    # Create test data
    # ========================================================================
    torch.manual_seed(42)

    torch_activation = torch.randn((M, K_gate), dtype=torch.bfloat16)
    torch_gate_weights = torch.randn((K_gate, K_down), dtype=torch.bfloat16)
    torch_up_weights = torch.randn((K_gate, K_down), dtype=torch.bfloat16)
    torch_down_weights = torch.randn((K_down, N), dtype=torch.bfloat16)
    torch_bias = torch.randn((M, N), dtype=torch.bfloat16)

    # Golden reference
    torch_expected = SharedExpertOp.golden(
        torch_activation.float(),
        torch_gate_weights.float(),
        torch_up_weights.float(),
        torch_down_weights.float(),
        torch_bias.float(),
    ).bfloat16()
    logger.info(f"Golden output shape: {torch_expected.shape}")

    # ========================================================================
    # Activation tensor: [1, K_gate] HEIGHT_SHARDED on sender (12,9)
    # ========================================================================
    act_shard_spec = ttnn.ShardSpec(sender_core_grid, (M, K_gate), ttnn.ShardOrientation.ROW_MAJOR)
    act_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, act_shard_spec)

    ttnn_activation = ttnn.from_torch(
        torch_activation,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=act_mem,
        tile=a_tile,
    )

    # ========================================================================
    # Gate/Up/Down weights
    # ========================================================================
    # BlitzDecodeWeights hard-codes weights to bfloat4_b; use the old flow to test bfloat8_b weights
    if use_bdw:
        bdw = BlitzDecodeWeights(device)
        gate_ov, _up_ov, ttnn_down_weights = bdw.get_tt_moe_shared_expert_weights(
            torch_gate_weights,
            torch_up_weights,
            torch_down_weights,
        )
        ttnn_gate_up_weights = gate_ov.fused_tensor
        logger.info("Created shared expert weights via BlitzDecodeWeights")
    else:
        a_cores_list, b_cores_list = SharedExpertOp.build_ab_grids()
        compute_cores_list = a_cores_list + b_cores_list

        weight_shards = []
        for i, core in enumerate(a_cores_list):
            k_idx = i // n_parallel
            n_idx = i % n_parallel
            k_start = k_idx * k_per_core * 32
            k_end = k_start + k_per_core * 32
            n_start = n_idx * 32
            n_end = n_start + 32
            weight_shards.append(torch_gate_weights[k_start:k_end, n_start:n_end])

        for i, core in enumerate(b_cores_list):
            k_idx = i // n_parallel
            n_idx = i % n_parallel
            k_start = k_idx * k_per_core * 32
            k_end = k_start + k_per_core * 32
            n_start = n_idx * 32
            n_end = n_start + 32
            weight_shards.append(torch_up_weights[k_start:k_end, n_start:n_end])

        torch_gate_up_stacked = torch.cat(weight_shards, dim=0)
        logger.info(f"Gate/Up weights stacked shape: {torch_gate_up_stacked.shape}")

        compute_core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in compute_cores_list])
        gu_shard_spec = ttnn.ShardSpec(compute_core_grid, (k_per_core * 32, 32), ttnn.ShardOrientation.ROW_MAJOR)
        gu_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gu_shard_spec)

        ttnn_gate_up_weights = ttnn.from_torch(
            torch_gate_up_stacked,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=gu_mem,
            tile=b_tile,
        )

        matmul_core_grid = DownProj.build_matmul_core_grid()
        down_shard_spec = ttnn.ShardSpec(matmul_core_grid, (K_down, N_per_core), ttnn.ShardOrientation.ROW_MAJOR)
        down_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, down_shard_spec)

        ttnn_down_weights = ttnn.from_torch(
            torch_down_weights,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=down_mem,
            tile=b_tile,
        )
        logger.info(f"Down weights: shard ({K_down}, {N_per_core}) on {DownProj.NUM_MATMUL_CORES} cores")

    # ========================================================================
    # Bias: [1, N] HEIGHT_SHARDED on sender (12,9)
    # ========================================================================
    bias_shard_spec = ttnn.ShardSpec(sender_core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    bias_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, bias_shard_spec)

    ttnn_bias = ttnn.from_torch(
        torch_bias,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=bias_mem,
        tile=out_tile,
    )

    # ========================================================================
    # Output: [1, N] HEIGHT_SHARDED on sender (12,9)
    # ========================================================================
    output_shard_spec = ttnn.ShardSpec(sender_core_grid, (M, N), ttnn.ShardOrientation.ROW_MAJOR)
    output_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    ttnn_output = ttnn.from_torch(
        torch.zeros((M, N), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem,
        tile=out_tile,
    )

    # ========================================================================
    # Run shared expert
    # ========================================================================
    logger.info("-" * 70)
    logger.info("Running shared expert ...")

    ttnn_result = SharedExpertOp.op(
        ttnn_activation,
        ttnn_gate_up_weights,
        ttnn_down_weights,
        ttnn_bias,
        ttnn_output,
        k_parallel,
        n_parallel,
    )

    output_torch = ttnn.to_torch(ttnn_result)
    logger.info(f"Output shape: {output_torch.shape}")

    expected_shape = (M, N)
    assert output_torch.shape == expected_shape, f"Expected {expected_shape}, got {output_torch.shape}"

    pcc_threshold = 0.97
    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(f"PCC comparison: {pcc_message}")

    assert passing, f"PCC check failed: {pcc_message}"
    logger.info("=" * 70)
    logger.info("Shared expert test PASSED!")
    logger.info("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
