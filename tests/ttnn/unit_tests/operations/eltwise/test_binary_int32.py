# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -100, 100),
        (-300, 300, -250, 250),
        (-500, 500, -750, 750),
        (-1000, 1000, -500, 1000),
        (-1e4, 1e4, -5e3, 5e3),
        (2e9, 2077000000, 2e9, 2147483647),  # large positive input
        (-2147483647, -2e9, -2077000000, -2e9),  # large negative input
    ],
)
@pytest.mark.parametrize(
    "logical_op",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
    ],
)
def test_binary_logical_int32(input_shapes, low_a, high_a, low_b, high_b, logical_op, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a[::5] = 0  # every 5th element is zero
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(input_shapes).nan_to_num(0.0)

    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b[::10] = 0  # every 10th element is zero
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(input_shapes).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(logical_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = logical_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([1, 2, 32]), torch.Size([1, 2, 32])),
        (torch.Size([1]), torch.Size([1, 5, 12])),
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
        (torch.Size([]), torch.Size([])),
        (torch.Size([5]), torch.Size([1])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (-100, 100, -100, 100),
        (-300, 300, -250, 250),
        (-500, 500, -750, 750),
        (-1000, 1000, -500, 1000),
        (-1e4, 1e4, -5e3, 5e3),
        (2e9, 2077000000, 2e9, 2147483647),  # large positive input
        (-2147483647, -2e9, -2077000000, -2e9),  # large negative input
    ],
)
@pytest.mark.parametrize(
    "logical_op",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
    ],
)
def test_binary_logical_int32_bcast(a_shape, b_shape, low_a, high_a, low_b, high_b, logical_op, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(logical_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = logical_op(input_tensor_a, input_tensor_b, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

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
    "a_shape, b_shape",
    ((torch.Size([5, 7, 64, 128]), torch.Size([5, 7, 64, 128])),),
)
@pytest.mark.parametrize(
    "sharded_config",
    [
        height_sharded_memory_config,
        width_sharded_memory_config,
        block_sharded_memory_config,
    ],
)
@pytest.mark.parametrize("ttnn_fn", ("logical_or", "logical_xor"))
def test_binary_logical_int32_sharded(a_shape, b_shape, sharded_config, ttnn_fn, device):
    ttnn_op = getattr(ttnn, ttnn_fn)
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(0, 100, num_elements, dtype=torch.int32)
    torch_input_tensor_a[::5] = 0
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(100, 200, num_elements, dtype=torch.int32)
    torch_input_tensor_b[::10] = 0
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, memory_config=sharded_config, use_legacy=False)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "logical_op",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
    ],
)
def test_binary_logical_int32_edge_cases(logical_op, device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 1, -1, 2147483647, -2147483647, 2147483647, 0])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, -1, 1, -1, 2147483647, -2147483647, 0, -2147483647])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(logical_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = logical_op(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)
