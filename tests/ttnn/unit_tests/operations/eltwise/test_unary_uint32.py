# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


def create_full_range_tensor(input_shape, dtype, value_ranges):
    num_elements = torch.prod(torch.tensor(input_shape)).item()

    num_ranges = len(value_ranges)
    elements_per_range = num_elements // num_ranges
    remainder = num_elements % num_ranges

    segments = []
    for i, (low, high) in enumerate(value_ranges):
        range_elements = elements_per_range + (1 if i < remainder else 0)

        segment = torch.linspace(low, high, steps=range_elements, dtype=dtype)
        segments.append(segment)

    in_data = torch.cat(segments)
    in_data = in_data.reshape(input_shape)
    return in_data


@pytest.mark.parametrize(
    "ttnn_op, value_ranges",
    [
        (
            ttnn.logical_not,
            [
                (0, 1e3),
                (1e3, 1e5),
                (1e5, 1e8),
                (1e8, 2e9),
                (2e9, 4e9),
                (4e9, 4294967295),
            ],
        ),
        (
            ttnn.square,
            [(0, 100), (100, 1e3), (1e3, 5e3), (5e3, 1e4), (1e4, 46340)],
        ),
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        torch.Size([1, 2, 32, 128]),
    ],
)
def test_unary_uint32(ttnn_op, value_ranges, input_shape, device_module):
    device = device_module
    torch_input_tensor = create_full_range_tensor(input_shape=input_shape, dtype=torch.int64, value_ranges=value_ranges)
    if ttnn_op == ttnn.logical_not:
        torch_input_tensor[..., ::5] = 0  # every 5th element is zero
        corner_case = torch.tensor([4294967295], dtype=torch.int64)
        torch_input_tensor = torch_input_tensor.flatten()
        torch_input_tensor[-len(corner_case) :] = corner_case
        torch_input_tensor = torch_input_tensor.reshape(input_shape)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn_op(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int64)

    assert torch.equal(output_tensor, torch_output_tensor)


height_sharded_memory_config = ttnn.create_sharded_memory_config(
    [128, 160],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (1, 6)), ttnn.CoreRange((3, 0), (3, 6))}),
    strategy=ttnn.ShardStrategy.HEIGHT,
    orientation=ttnn.ShardOrientation.COL_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

width_sharded_memory_config = ttnn.create_sharded_memory_config(
    [2240, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((2, 2), (2, 3)), ttnn.CoreRange((0, 0), (0, 1))}),
    strategy=ttnn.ShardStrategy.WIDTH,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)

block_sharded_memory_config = ttnn.create_sharded_memory_config(
    [320, 32],
    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((1, 0), (4, 6))}),
    strategy=ttnn.ShardStrategy.BLOCK,
    orientation=ttnn.ShardOrientation.ROW_MAJOR,
    use_height_and_width_as_shard_shape=True,
)


@pytest.mark.parametrize(
    "a_shape",
    [(torch.Size([5, 7, 64, 128]))],
)
@pytest.mark.parametrize(
    "sharded_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.square,
        ttnn.logical_not,
    ],
)
def test_unary_uint32_sharded(a_shape, sharded_config, ttnn_op, device_module):
    device = device_module
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor = torch.linspace(0, 4000, num_elements, dtype=torch.int32)
    if ttnn_op == ttnn.logical_not:
        torch_input_tensor[::7] = 0  # every 7th element is zero
    torch_input_tensor = torch_input_tensor[:num_elements].reshape(a_shape)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    output_tensor = ttnn_op(input_tensor, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


# Inputs beyond 65535 will give an output > 2^32-1 when squared, causing overflow.
# This test checks that overflow is handled correctly.
def test_unary_square_uint32_overflow(device_module):
    device = device_module
    torch_input_tensor = torch.tensor([0, 1, 46340, 65535, 65536, 70000])
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.square(input_tensor)
    golden_function = ttnn.get_golden_function(ttnn.square)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)
    torch_output_tensor = ttnn.from_torch(
        torch_output_tensor,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    comparison_result = ttnn.eq(output_tensor, torch_output_tensor)
    comparison_torch = ttnn.to_torch(comparison_result)

    assert torch.all(comparison_torch), f"Mismatch found in uint32 square results"
    # square of 65536 and 70000 should overflow to 0 and 4900000000 - 4294967296 = 605032704, respectively
    # output_tensor = ttnn.Tensor([0, 1, 2147395600, 4294836225, 0, 605032704], shape=Shape([6]), dtype=DataType::UINT32, layout=Layout::TILE)
