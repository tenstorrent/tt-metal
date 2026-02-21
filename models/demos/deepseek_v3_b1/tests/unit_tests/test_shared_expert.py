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


def test_shared_expert(device):
    """Test shared expert: activation → gate/up matmul → gated reduce → down proj + bias."""

    device_grid = device.compute_with_storage_grid_size()
    if device_grid.x < 13 or device_grid.y < 10:
        pytest.skip(f"Device grid {device_grid.x}x{device_grid.y} too small for 13x10")

    M = 1
    K_gate = 7168
    cfg = GATE_UP_PROJ_SINGLE_DEVICE_OVERLAP_SPEC
    k_parallel = cfg.k_parallel
    n_parallel = cfg.n_parallel
    K_down = n_parallel * 32  # 256
    N = 64 * DownProj.NUM_MATMUL_CORES  # 7168

    logger.info("=" * 70)
    logger.info("Testing Shared Expert:")
    logger.info(f"  K_gate={K_gate}, K_down={K_down}, N={N}")
    logger.info(f"  k_parallel={k_parallel}, n_parallel={n_parallel}")
    logger.info("=" * 70)

    # Tile definitions
    a_tile = ttnn.Tile([M, 32])
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
    # Gate/Up/Down weights via BlitzDecodeWeights
    # ========================================================================
    bdw = BlitzDecodeWeights(device)
    gate_ov, _up_ov, ttnn_down_weights = bdw.get_tt_moe_shared_expert_weights(
        torch_gate_weights,
        torch_up_weights,
        torch_down_weights,
    )
    ttnn_gate_up_weights = gate_ov.fused_tensor
    logger.info("Created shared expert weights via BlitzDecodeWeights")

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
