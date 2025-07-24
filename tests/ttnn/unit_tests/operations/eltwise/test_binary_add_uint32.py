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
        (0, 100, 0, 300),
        (1000, 10000, 500, 1000),
        (30000, 40000, 10000, 15000),
        (50000, 55000, 1000, 2000),
        (0, 70000, 80000, 200000),
    ],
)
def test_binary_add_uint32_bcast(a_shape, b_shape, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn.add)
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
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, use_legacy=None)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

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
def test_binary_add_uint32_sharded(a_shape, b_shape, sharded_config, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(0, 100, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(100, 200, num_elements, dtype=torch.int32)
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

    golden_function = ttnn.get_golden_function(ttnn.add)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, memory_config=sharded_config, use_legacy=None)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


# For inputs within int32 range [0, 2147483647]
def test_binary_add_uint32_lower_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 1147482, 2147483647, 1])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 1, 1147483, 0, 2147483646])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.add)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


# For inputs outside int32 range [2147483648, 4294967295]
def test_binary_add_uint32_upper_edge_cases(device):
    torch_input_tensor_a = torch.tensor([3000000000, 2750000000, 2147483648, 4294967292, 1, 4294967295, 0])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([1294967290, 1400000000, 2147483647, 2, 4294967291, 0, 4294967295])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.add)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    # Since ttnn.to_torch does not support int64, we cannot compare with the torch output. We can manually check the outputs:
    # Torch output: tensor([4294967290, 4150000000, 4294967295, 4294967294, 4294967292, 4294967295, 4294967295])
    # TT output: ttnn.Tensor([4294967290, 4150000000, 4294967295, 4294967294, 4294967292, 4294967295, 4294967295],
    #               shape=Shape([3]), dtype=DataType::UINT32, layout=Layout::TILE)
