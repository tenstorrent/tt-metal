# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Tests for gated local reduce with SiLU activation.

Two parallel reductions:
    group1_result = SiLU(group1[0] + group1[1] + ... + group1[n-1])
    group2_result = group2[0] + group2[1] + ... + group2[m-1]
    output = group1_result * group2_result

This pattern is used in gated MLP where:
  - Group 1 is the "gate" path (with SiLU)
  - Group 2 is the "up" path (no SiLU)
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_b1.fused_ops.gated_local_reduce.op import GatedLocalReduceOp


@pytest.mark.parametrize(
    "tile_h, tile_w, group1_num_tiles, group2_num_tiles",
    [
        (32, 32, 2, 2),  # Minimal case: 2 tiles each
        (32, 32, 4, 2),  # More group1 tiles
        (32, 32, 2, 4),  # More group2 tiles
        (32, 32, 4, 4),  # Equal larger groups
        (16, 16, 4, 4),  # Smaller tiles
        (32, 32, 8, 8),  # Larger groups (simulates MoE expert accumulation)
    ],
)
def test_gated_local_reduce(device, tile_h, tile_w, group1_num_tiles, group2_num_tiles):
    """Test gated local reduce with SiLU on group1."""
    tile = ttnn.Tile([tile_h, tile_w])

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    logger.info(
        f"Testing gated local reduce: tile=[{tile_h}, {tile_w}], "
        f"group1={group1_num_tiles} tiles, group2={group2_num_tiles} tiles"
    )

    # Create input tensors for both groups
    torch.manual_seed(42)
    group1_inputs = [torch.randn((tile_h, tile_w), dtype=torch.bfloat16) for _ in range(group1_num_tiles)]
    group2_inputs = [torch.randn((tile_h, tile_w), dtype=torch.bfloat16) for _ in range(group2_num_tiles)]

    # Golden reference
    torch_expected = GatedLocalReduceOp.golden(
        [t.float() for t in group1_inputs], [t.float() for t in group2_inputs]
    ).bfloat16()

    # Stack inputs into tensors
    torch_group1_stacked = torch.cat(group1_inputs, dim=0)  # [N*tile_h, tile_w]
    torch_group2_stacked = torch.cat(group2_inputs, dim=0)  # [M*tile_h, tile_w]

    # Create sharded memory config for group1 input
    group1_shard_shape = (group1_num_tiles * tile_h, tile_w)
    group1_shard_spec = ttnn.ShardSpec(
        core_grid,
        group1_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    group1_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, group1_shard_spec)

    ttnn_group1 = ttnn.from_torch(
        torch_group1_stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=group1_mem_config,
        tile=tile,
    )

    # Create sharded memory config for group2 input
    group2_shard_shape = (group2_num_tiles * tile_h, tile_w)
    group2_shard_spec = ttnn.ShardSpec(
        core_grid,
        group2_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    group2_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, group2_shard_spec)

    ttnn_group2 = ttnn.from_torch(
        torch_group2_stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=group2_mem_config,
        tile=tile,
    )

    # Create output tensor
    output_shard_shape = (tile_h, tile_w)
    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    ttnn_output = ttnn.from_torch(
        torch.zeros((tile_h, tile_w), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    logger.info(f"Running gated local reduce: group1={group1_num_tiles} + SiLU, group2={group2_num_tiles}...")
    ttnn_result = GatedLocalReduceOp.op(
        ttnn_group1,
        ttnn_group2,
        ttnn_output,
        group1_num_tiles,
        group2_num_tiles,
    )

    # Convert back to torch and verify
    output_torch = ttnn.to_torch(ttnn_result)
    assert output_torch.shape == (tile_h, tile_w), f"Expected shape ({tile_h}, {tile_w}), got {output_torch.shape}"

    # SiLU uses approximation, so we use a slightly lower PCC threshold
    pcc_threshold = 0.998
    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info(f"Test passed! (tile=[{tile_h}, {tile_w}], group1={group1_num_tiles}, group2={group2_num_tiles})")


@pytest.mark.parametrize(
    "group1_num_tiles, group2_num_tiles",
    [
        (2, 2),  # Minimal MoE case
        (4, 4),  # Small MoE case
        (8, 8),  # Medium MoE case
    ],
)
def test_gated_local_reduce_moe_pattern(device, group1_num_tiles, group2_num_tiles):
    """
    Test pattern matching MoE gated MLP computation.

    In MoE, we often compute:
        gate_output = SiLU(sum of gate projections)
        up_output = sum of up projections
        final = gate_output * up_output  (element-wise multiplication)

    This test verifies the gated local reduce pattern works for MoE-like workloads.
    """
    tile_h, tile_w = 32, 32
    tile = ttnn.Tile([tile_h, tile_w])

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    logger.info(f"Testing MoE pattern: group1={group1_num_tiles} (gate+SiLU), group2={group2_num_tiles} (up)")

    # Simulate MoE expert outputs (each expert contributes one tile)
    torch.manual_seed(123)
    gate_tiles = [torch.randn((tile_h, tile_w), dtype=torch.bfloat16) for _ in range(group1_num_tiles)]
    up_tiles = [torch.randn((tile_h, tile_w), dtype=torch.bfloat16) for _ in range(group2_num_tiles)]

    # Golden: SiLU(sum(gate)) + sum(up)
    torch_expected = GatedLocalReduceOp.golden(
        [t.float() for t in gate_tiles], [t.float() for t in up_tiles]
    ).bfloat16()

    # Stack inputs
    torch_gate_stacked = torch.cat(gate_tiles, dim=0)
    torch_up_stacked = torch.cat(up_tiles, dim=0)

    # Create tensors on device
    gate_shard_spec = ttnn.ShardSpec(
        core_grid,
        (group1_num_tiles * tile_h, tile_w),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    gate_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, gate_shard_spec)

    up_shard_spec = ttnn.ShardSpec(
        core_grid,
        (group2_num_tiles * tile_h, tile_w),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    up_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, up_shard_spec)

    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        (tile_h, tile_w),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    ttnn_gate = ttnn.from_torch(
        torch_gate_stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=gate_mem_config,
        tile=tile,
    )

    ttnn_up = ttnn.from_torch(
        torch_up_stacked,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=up_mem_config,
        tile=tile,
    )

    ttnn_output = ttnn.from_torch(
        torch.zeros((tile_h, tile_w), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=tile,
    )

    # Run operation
    logger.info("Running gated local reduce for MoE pattern...")
    ttnn_result = GatedLocalReduceOp.op(
        ttnn_gate,
        ttnn_up,
        ttnn_output,
        group1_num_tiles,
        group2_num_tiles,
    )

    # Verify
    output_torch = ttnn.to_torch(ttnn_result)

    pcc_threshold = 0.998
    passing, pcc_message = comp_pcc(torch_expected, output_torch, pcc_threshold)
    logger.info(pcc_message)

    assert passing, pcc_message
    logger.info(f"MoE pattern test passed! (group1={group1_num_tiles}, group2={group2_num_tiles})")
