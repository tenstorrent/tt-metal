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
        (1, 64),
    ],
)
def test_resblock(device, B, K):
    a_tile = ttnn.Tile([B, 32])
    weight_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile([B, 32])

    # For now we only support cleanly divisible K by weight tile width
    assert (
        K % weight_tile.tile_shape[1] == 0
    ), f"K ({K}) must be divisible by weight tile width ({weight_tile.tile_shape[1]})"
    number_of_matmul_cores = K // weight_tile.tile_shape[1]
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(number_of_matmul_cores - 1, 0))})

    logger.info(f"Testing fused ResBlock with shape [{B}, {K}] x [{K}, {K}] on {number_of_matmul_cores} cores")

    torch.manual_seed(0)

    torch_a = torch.randn((B, K), dtype=torch.bfloat16)
    weight0 = torch.randn((K, K), dtype=torch.bfloat16)
    weight1 = torch.randn((K, K), dtype=torch.bfloat16)

    expected = FusedResblock.golden(torch_a.float(), weight0.float(), weight1.float()).bfloat16()

    input_a_shard_shape = (B, K)  # This tensor is unique because it is replicated across all matmul cores
    input_a_shard_spec = ttnn.ShardSpec(
        core_grid,
        input_a_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_a_mem_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, input_a_shard_spec
    )
    # TODO: For now we replicate input across cores by replicating by number_of_matmul_times and then height sharding, but in the future we should have the op itself do the mcast
    ttnn_a = ttnn.from_torch(
        torch_a.repeat(number_of_matmul_cores, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created TTNN input tensor with shard shape {input_a_shard_shape}")

    def create_weight_tensor(weight, tile, core_grid):
        # Weights are width-sharded across all cores
        weight_shard_shape = (weight.shape[0], weight.shape[1] // number_of_matmul_cores)
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

    weight0_tensor = create_weight_tensor(weight0, weight_tile, core_grid)
    weight1_tensor = create_weight_tensor(weight1, weight_tile, core_grid)

    output_shard_shape = (B, K // number_of_matmul_cores)
    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((B, K), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    logger.info("Running Fused ResBlock operation")
    ttnn_output = FusedResblock.op(
        ttnn_a,
        weight0_tensor,
        weight1_tensor,
        ttnn_output,
    )

    logger.info("Converting TTNN output to torch")
    torch_output = ttnn.to_torch(ttnn_output)
    print(torch_output)
    assert torch_output.shape == (B, K), f"Expected shape ({B}, {K}), got {torch_output.shape}"

    passing, pcc_message = comp_pcc(expected, torch_output, 0.999)
    logger.info(pcc_message)

    assert passing, pcc_message
    logger.info("Fused ResBlock test passed!")
