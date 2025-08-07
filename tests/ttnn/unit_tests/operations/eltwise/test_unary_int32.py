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
    "low_a, high_a",
    [
        (-1000, 1000),
        (-1e4, 1e4),
        (-1e6, 1e6),
        (-2147483647, 0),  # max negative to zero
        (0, 2147483647),  # zero to max positive
        (2e9, 2147483647),  # large positive input
        (-2147483647, -2e9),  # large negative input
        (-2147483647, 2147483647),  # whole range
    ],
)
@pytest.mark.parametrize(
    "logical_op",
    [
        ttnn.logical_not,
    ],
)
def test_unary_logical_int32(input_shapes, low_a, high_a, logical_op, device):
    num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a[::5] = 0  # every 5th element is zero
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(logical_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = logical_op(input_tensor_a)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "logical_op",
    [
        ttnn.logical_not,
    ],
)
def test_unary_logical_int32_edge_cases(logical_op, device):
    # Note: torch.logical_not(-2147483648) returns False, whereas ttnn.logical_not(-2147483648) returns True.
    # This discrepancy occurs because, in Wormhole, the value -2147483648 wraps around to 0 due to integer overflow.
    torch_input_tensor_a = torch.tensor([0, 1, -1, 2147483647, -2147483647])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(logical_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, device=device)

    output_tensor = logical_op(input_tensor_a)
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
        "logical_not",
        "square",
    ],
)
def test_unary_logical_int32_sharded(a_shape, sharded_config, ttnn_op, device):
    ttnn_op = getattr(ttnn, ttnn_op)
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(-100, 100, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, device=device)

    output_tensor = ttnn_op(input_tensor_a, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([])),
        (torch.Size([1, 64])),
        (torch.Size([1, 2, 32])),
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([1, 2, 32, 64, 125])),
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a",
    [
        (-100, 100),
        (-1000, 1000),
        (-1e4, 1e4),
        (-46340, 0),
        (0, 46340),
        (-46340, 46340),  # output overflows beyond this range
    ],
)
def test_unary_square_int32(input_shapes, low_a, high_a, device):
    if len(input_shapes) == 0:
        torch_input_tensor = torch.randint(low=-(2**15), high=2**15, size=(), dtype=torch.int32)
    else:
        num_elements = max(int(torch.prod(torch.tensor(input_shapes)).item()), 1)
        torch_input_tensor = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)
        torch_input_tensor = torch_input_tensor[:num_elements].reshape(input_shapes)

    golden_function = ttnn.get_golden_function(ttnn.square)
    torch_output_tensor = golden_function(torch_input_tensor, device=device)

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.int32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn.square(input_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(output_tensor, torch_output_tensor)
