# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn


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
        (100, 500, 0, 100),  # Ensure a > b to avoid negative results
        (1000, 10000, 500, 1000),
        (30000, 40000, 10000, 15000),
        (50000, 55000, 1000, 2000),
        (80000, 200000, 0, 70000),
    ],
)
def test_binary_sub_uint32_bcast(a_shape, b_shape, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)
    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

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
        (0, 100, 200, 300),  # Test underflow cases (a < b)
        (500, 1000, 1000, 10000),
        (10000, 15000, 30000, 40000),
        (1000, 2000, 50000, 55000),
        (0, 70000, 80000, 200000),
    ],
)
def test_binary_sub_uint32_underflow(a_shape, b_shape, low_a, high_a, low_b, high_b, device):
    """Test uint32 subtraction with underflow cases (wrap-around behavior)"""
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize("use_legacy", [True, False])
def test_binary_sub_uint32_edge_cases(use_legacy, device):
    """Test uint32 subtraction with edge cases including underflow"""
    torch_input_tensor_a = torch.tensor(
        [4294967295, 2147483647, 1000000000, 500, 0, 100, 1, 0]  # uint32 max, int32 max, large, medium, min
    )
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    torch_input_tensor_b = torch.tensor(
        [1, 1000000000, 2147483647, 1000, 1, 50, 2, 1]  # Various values to test different scenarios
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b, use_legacy=use_legacy)
    torch_output_tensor = torch_output_tensor.to(dtype=torch.uint32)
    # Since ttnn.to_torch does not support int64, we cannot compare with the torch output. We can manually check the outputs:
    # Torch output: tensor([4294967294, 1147483647, 3147483649, 4294966796, 4294967295,   50,
    #                       4294967295, 4294967295], dtype=torch.uint32)
    # TT output: ttnn.Tensor([4294967294, 1147483647, 3147483649, 4294966796, 4294967295,  50,
    #                         4294967295, 4294967295], shape=Shape([8]), dtype=DataType::UINT32, layout=Layout::TILE)


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
def test_binary_sub_uint32_sharded(a_shape, b_shape, sharded_config, device):
    """Test uint32 subtraction with sharded memory configurations"""
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(50000, 100000, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(0, 40000, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b, memory_config=sharded_config)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)
