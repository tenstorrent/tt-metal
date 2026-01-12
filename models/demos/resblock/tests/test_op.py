# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.resblock.op import FusedResblock

"""
Test fused ResBlock operation with inputs [B, K] @ [K, K] -> [B, K]
"""


@pytest.mark.parametrize(
    "B, K",
    [
        (1, 32),
    ],
)
def test_resblock(device, B, K):
    a_tile = ttnn.Tile([B, K])
    weight_tile = ttnn.Tile([K, K])
    out_tile = ttnn.Tile([B, K])

    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(core, core)})

    logger.info(f"Testing fused ResBlock with shape [{B}, {K}] x [{K}, {K}]")

    torch.manual_seed(0)
    torch_a = torch.randn((B, K), dtype=torch.bfloat16)
    weight0 = torch.randn((K, K), dtype=torch.bfloat16)
    weight1 = torch.randn((K, K), dtype=torch.bfloat16)

    expected = FusedResblock.golden(torch_a, weight0, weight1)
    print(expected)

    input_a_shard_shape = (B, K)
    input_a_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_a_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_a_shard_spec
    )
    ttnn_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created TTNN input tensor with shard shape {input_a_shard_shape}")

    def create_weight_tensor(weight, tile):
        weight_shard_shape = weight.shape
        weight_shard_spec = ttnn.ShardSpec(
            core_grid,
            weight_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        weight_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weight_shard_spec
        )
        return ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=weight_mem_config,
            tile=tile,
        )

    weight0_tensor = create_weight_tensor(weight0, weight_tile)
    weight1_tensor = create_weight_tensor(weight1, weight_tile)

    output_shard_shape = (B, K)
    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)

    # Create output tensor
    torch_output_zeros = torch.zeros((B, K), dtype=torch.bfloat16)
    ttnn_output = ttnn.from_torch(
        torch_output_zeros,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )
    logger.info(f"Created output tensor with shard shape {output_shard_shape}")

    ttnn_output = FusedResblock.op(
        ttnn_a,
        weight0_tensor,
        weight1_tensor,
        ttnn_output,
    )

    # Convert back to torch for comparison
    torch_output = ttnn.to_torch(ttnn_output)

    # Verify output shape
    assert torch_output.shape == (B, K), f"Expected shape ({B}, {K}), got {torch_output.shape}"

    passing, pcc_message = comp_pcc(expected, torch_output, 0.99)
    logger.info(pcc_message)

    assert passing, pcc_message

    logger.info("✓ Fused ResBlock test passed!")
