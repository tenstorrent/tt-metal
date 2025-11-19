# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import pytest


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_where_with_height_sharding(
    device, input_a_sharded, condition, input_b_sharded, out_sharded, shard_orientation, tor_dtype, ttnn_dtype
):
    torch.manual_seed(0)
    shape = (1, 1, 512, 512)

    C = torch.ones(shape, dtype=tor_dtype) * condition
    # Set zeros at flattened indices which are multiples of 8
    C_flat = C.flatten()
    C_flat[::8] = 1 - condition
    C = C_flat.reshape(shape)

    torch_input_tensor_a = C
    torch_input_tensor_b = torch.rand(shape, dtype=tor_dtype)
    scalar = 15.5

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (512 // 8, 512)
    else:
        shard_shape = (512, 512 // 8)

    height_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_input_tensor_a.bool(), torch_input_tensor_b, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, height_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, height_sharded_mem_config)

    if out_sharded:
        out_mem_config = height_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(input_tensor_a, input_tensor_b, scalar, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_add_with_width_sharding(
    device, input_a_sharded, input_b_sharded, out_sharded, condition, shard_orientation, tor_dtype, ttnn_dtype
):
    torch.manual_seed(0)
    shape = (1, 1, 512, 512)
    C = torch.ones(shape, dtype=tor_dtype) * condition
    # Set zeros at flattened indices which are multiples of 8
    C_flat = C.flatten()
    C_flat[::8] = 1 - condition
    C = C_flat.reshape(shape)

    torch_input_tensor_a = C
    torch_input_tensor_b = torch.rand(shape, dtype=tor_dtype)
    scalar = 15.5

    if shard_orientation == ttnn.ShardOrientation.ROW_MAJOR:
        shard_shape = (512, 512 // 8)
    else:
        shard_shape = (512 // 8, 512)

    width_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_input_tensor_a.bool(), torch_input_tensor_b, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, width_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, width_sharded_mem_config)

    if out_sharded:
        out_mem_config = width_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(input_tensor_a, input_tensor_b, scalar, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(torch_output_tensor, output_tensor)


@pytest.mark.parametrize("input_a_sharded", [True, False])
@pytest.mark.parametrize("input_b_sharded", [True, False])
@pytest.mark.parametrize("out_sharded", [True, False])
@pytest.mark.parametrize("condition", [1, 0])
@pytest.mark.parametrize("shard_orientation", [ttnn.ShardOrientation.ROW_MAJOR, ttnn.ShardOrientation.COL_MAJOR])
@pytest.mark.parametrize(
    "tor_dtype, ttnn_dtype",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
)
def test_add_with_block_sharding(
    device, input_a_sharded, input_b_sharded, out_sharded, condition, shard_orientation, tor_dtype, ttnn_dtype
):
    torch.manual_seed(0)
    shape = (1, 1, 512, 512)
    C = torch.ones(shape, dtype=tor_dtype) * condition
    # Set zeros at flattened indices which are multiples of 8
    C_flat = C.flatten()
    C_flat[::8] = 1 - condition
    C = C_flat.reshape(shape)

    torch_input_tensor_a = C
    torch_input_tensor_b = torch.rand(shape, dtype=tor_dtype)
    scalar = 15.5

    shard_shape = (512 // 2, 512 // 4)

    block_sharded_mem_config = ttnn.create_sharded_memory_config(
        shape=shard_shape,
        core_grid=ttnn.CoreGrid(y=2, x=4),
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=shard_orientation,
        use_height_and_width_as_shard_shape=True,
    )

    torch_output_tensor = torch.where(torch_input_tensor_a.bool(), torch_input_tensor_b, scalar)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_a_sharded:
        input_tensor_a = ttnn.to_memory_config(input_tensor_a, block_sharded_mem_config)

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    if input_b_sharded:
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, block_sharded_mem_config)

    if out_sharded:
        out_mem_config = block_sharded_mem_config
    else:
        out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    output_tensor = ttnn.where(input_tensor_a, input_tensor_b, scalar, memory_config=out_mem_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert torch.equal(torch_output_tensor, output_tensor)
