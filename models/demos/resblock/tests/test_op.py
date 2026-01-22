# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.resblock.op import FusedResblock


def create_random_tensor(shape, random_tensor_gen):
    if random_tensor_gen == "uniform":
        return torch.empty(shape, dtype=torch.bfloat16).uniform_(-1.0, 1.0)
    if random_tensor_gen == "rand":
        return torch.rand(shape, dtype=torch.bfloat16)
    if random_tensor_gen == "randn":
        return torch.randn(shape, dtype=torch.bfloat16)
    raise ValueError(f"Unsupported random_tensor_gen: {random_tensor_gen}")


@pytest.mark.parametrize(
    "num_layers",
    [1, 2, 4, 8, 12],
)
@pytest.mark.parametrize(
    "B, K, core_grid",
    [
        (1, 32, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})),
        (1, 64, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})),
        (1, 512, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})),
        (1, 1024, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 7))})),
        (1, 2048, ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})),
    ],
)
@pytest.mark.parametrize(
    "tile_size",
    [
        (1, 32),
    ],
)
@pytest.mark.parametrize(
    "activation_dtype, weight_dtype",
    [(ttnn.bfloat16, ttnn.bfloat16), (ttnn.bfloat16, ttnn.bfloat8_b)],
)
@pytest.mark.parametrize(
    "generation_type",
    ["randn", "uniform"],
)
@pytest.mark.parametrize(
    "use_custom_mm",
    [True, False],
)
def test_resblock(
    device, B, K, core_grid, generation_type, tile_size, activation_dtype, weight_dtype, num_layers, use_custom_mm
):
    if (
        (K >= 2048 and num_layers >= 12)
        or (K >= 2048 and num_layers >= 8 and weight_dtype == ttnn.bfloat16)
        or (K >= 1024 and num_layers >= 12 and weight_dtype == ttnn.bfloat16)
    ):
        pytest.skip("Test uses too much L1 memory")

    torch.manual_seed(1234)

    a_tile = ttnn.Tile(tile_size)
    weight_tile = ttnn.Tile([32, 32])
    out_tile = ttnn.Tile(tile_size)

    # For now we only support cleanly divisible K by weight tile width
    assert (
        K % weight_tile.tile_shape[1] == 0
    ), f"K ({K}) must be divisible by weight tile width ({weight_tile.tile_shape[1]})"
    number_of_matmul_cores = K // weight_tile.tile_shape[1]
    logger.info(f"Testing fused ResBlock with shape [{B}, {K}] x [{K}, {K}] on {number_of_matmul_cores} cores")

    torch_a = create_random_tensor((B, K), generation_type)

    # Create per-layer weights
    weights = [
        (
            create_random_tensor((K, K), generation_type),
            create_random_tensor((K, K), generation_type),
        )
        for _ in range(num_layers)
    ]

    # Compute golden reference with per-layer weights
    expected = FusedResblock.golden(torch_a, weights)

    # Stack weights along row dimension for TTNN: [num_layers * K, K]
    stacked_weight0 = torch.cat([w0 for w0, _ in weights], dim=0)
    stacked_weight1 = torch.cat([w1 for _, w1 in weights], dim=0)

    # Pad input up to tile height
    torch_a = torch.nn.functional.pad(torch_a, (0, 0, 0, a_tile.tile_shape[0] - torch_a.shape[0]))

    input_a_shard_shape = (max(B, a_tile.tile_shape[0]), K)  # This tensor is replicated across all matmul cores
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
        dtype=activation_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=input_a_mem_config,
        tile=a_tile,
    )
    logger.info(f"Created TTNN input tensor with shard shape {input_a_shard_shape}")

    def create_weight_tensor(weight, tile, core_grid):
        # Stacked weights have shape [num_layers * K, K]
        weight_shard_shape = (weight.shape[0], weight.shape[1] // number_of_matmul_cores)
        weight_shard_spec = ttnn.ShardSpec(
            core_grid,
            weight_shard_shape,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        # Weights are width-sharded across all cores
        weight_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, weight_shard_spec
        )
        return ttnn.from_torch(
            weight,
            dtype=weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=weight_mem_config,
            tile=tile,
        )

    # Convert stacked weights to TTNN tensors
    weight0_tensor = create_weight_tensor(stacked_weight0, weight_tile, core_grid)
    weight1_tensor = create_weight_tensor(stacked_weight1, weight_tile, core_grid)

    output_shard_shape = (max(B, out_tile.tile_shape[0]), K // number_of_matmul_cores)
    output_shard_spec = ttnn.ShardSpec(
        core_grid,
        output_shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, output_shard_spec)
    ttnn_output = ttnn.from_torch(
        torch.zeros((max(B, out_tile.tile_shape[0]), K), dtype=torch.bfloat16),
        dtype=activation_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=output_mem_config,
        tile=out_tile,
    )

    logger.info(f"Running Fused ResBlock operation with use_custom_mm={use_custom_mm}")
    ttnn_output = FusedResblock.op(
        ttnn_a,
        weight0_tensor,
        weight1_tensor,
        ttnn_output,
        num_layers=num_layers,
        use_custom_mm=use_custom_mm,
        fp32_dest_acc_en=True,
    )

    logger.info("Converting TTNN output to torch")
    torch_output = ttnn.to_torch(ttnn_output)[:B, :]  # Slice off padding that we might have when using full tiles

    assert torch_output.shape == (B, K), f"Expected shape ({B}, {K}), got {torch_output.shape}"

    # Fail if we have infs or nans in the output
    if torch.isinf(torch_output).any() or torch.isnan(torch_output).any():
        pytest.fail("Output contains infs or nans")

    passing, pcc_message = comp_pcc(expected, torch_output, 0.9925)
    logger.info(pcc_message)

    assert passing, pcc_message
    logger.info("Fused ResBlock test passed!")
