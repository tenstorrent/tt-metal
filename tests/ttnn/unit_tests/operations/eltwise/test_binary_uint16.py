# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn

pytestmark = pytest.mark.use_module_device


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
        (0, 32000, 0, 32000),
    ],
)
def test_binary_add_uint16_bcast(a_shape, b_shape, low_a, high_a, low_b, high_b, device):
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
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.add(input_tensor_a, input_tensor_b, use_legacy=None)
    # Since to_torch converts ttnn.uint16 to int16 and then to int32, there will be an output mismatch
    # We can typecast ttnn.uint16 to ttnn.uint32 to avoid this mismatch
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


def test_binary_add_uint16_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 11, 500, 32767, 30000, 65535])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 1, 2, 7727, 1, 30000, 0])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.add)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.add(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
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
@pytest.mark.parametrize("ttnn_fn", ("add", "mul", "logical_and", "logical_or", "logical_xor"))
def test_binary_uint16_sharded(a_shape, b_shape, sharded_config, ttnn_fn, device):
    ttnn_op = getattr(ttnn, ttnn_fn)
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(0, 100, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(100, 200, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor_sharded = ttnn_op(input_tensor_a, input_tensor_b, memory_config=sharded_config, use_legacy=None)
    # Since typecast does not support sharded config, we can convert the sharded tensor to interleaved
    output_tensor = ttnn.to_memory_config(output_tensor_sharded, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
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
        (101, 300, 0, 100),
        (1000, 10000, 500, 999),
        (30000, 40000, 10000, 15000),
        (50000, 55000, 1000, 2000),
        (32001, 65000, 0, 32000),
    ],
)
def test_binary_sub_uint16_bcast(a_shape, b_shape, low_a, high_a, low_b, high_b, device):
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
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b, use_legacy=None)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


def test_binary_sub_uint16_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, 11, 7727, 65535, 65535, 65535])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 2, 500, 1, 30000, 0])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.sub(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


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
def test_binary_sub_uint16_sharded(a_shape, b_shape, sharded_config, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(151, 300, num_elements, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(0, 150, num_elements, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(ttnn.sub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor_sharded = ttnn.sub(input_tensor_a, input_tensor_b, memory_config=sharded_config, use_legacy=None)
    output_tensor = ttnn.to_memory_config(output_tensor_sharded, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
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
        (0, 100, 0, 300),
        (1000, 10000, 500, 1000),
        (30000, 40000, 10000, 15000),
        (50000, 55000, 1000, 2000),
        (0, 32000, 0, 32000),
    ],
)
@pytest.mark.parametrize(
    "bitwise_op",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
    ],
)
def test_binary_bitwise_op_uint16(a_shape, b_shape, low_a, high_a, low_b, high_b, bitwise_op, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements, dtype=torch.int32)
    corner_cases = torch.tensor([0, 1, 65535, 0, 65535, 1], dtype=torch.int32)
    torch_input_tensor_a = torch.cat([torch_input_tensor_a, corner_cases])
    torch_input_tensor_a = torch_input_tensor_a[-num_elements:].reshape(a_shape)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements, dtype=torch.int32)
    corner_cases = torch.tensor([0, 1, 65535, 65535, 1, 0], dtype=torch.int32)
    torch_input_tensor_b = torch.cat([torch_input_tensor_b, corner_cases])
    torch_input_tensor_b = torch_input_tensor_b[-num_elements:].reshape(b_shape)

    golden_function = ttnn.get_golden_function(bitwise_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = bitwise_op(input_tensor_a, input_tensor_b, use_legacy=None)
    # Since to_torch converts ttnn.uint16 to int16 and then to int32, there will be an output mismatch
    # We can typecast ttnn.uint16 to ttnn.uint32 to avoid this mismatch
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


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
@pytest.mark.parametrize(
    "bitwise_op",
    [
        ttnn.bitwise_and,
        ttnn.bitwise_or,
        ttnn.bitwise_xor,
    ],
)
def test_bitwise_op_uint16_sharded(a_shape, b_shape, sharded_config, bitwise_op, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(0, 100, num_elements, dtype=torch.int32)
    corner_cases = torch.tensor([0, 1, 65535, 0, 65535, 1], dtype=torch.int32)
    torch_input_tensor_a = torch.cat([torch_input_tensor_a, corner_cases])
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape).nan_to_num(0.0)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(100, 200, num_elements, dtype=torch.int32)
    corner_cases = torch.tensor([0, 1, 65535, 65535, 1, 0], dtype=torch.int32)
    torch_input_tensor_b = torch.cat([torch_input_tensor_b, corner_cases])
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape).nan_to_num(0.0)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=sharded_config,
    )

    golden_function = ttnn.get_golden_function(bitwise_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor_sharded = bitwise_op(input_tensor_a, input_tensor_b, memory_config=sharded_config, use_legacy=None)
    # Since typecast does not support sharded config, we can convert the sharded tensor to interleaved
    output_tensor = ttnn.to_memory_config(output_tensor_sharded, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
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
        (0, 255, 0, 255),
        (0, 100, 100, 300),
        (0, 150, 300, 430),
        (0, 32, 1000, 2000),
        (0, 4, 10000, 16000),
        (0, 2, 16000, 32767),
        (0, 1, 32767, 65535),
    ],
)
def test_binary_mul_uint16_bcast(a_shape, b_shape, low_a, high_a, low_b, high_b, device):
    num_elements_a = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    if high_a in (32, 4, 2, 1):
        values_a = torch.arange(low_a, high_a + 1, dtype=torch.int32)
        torch_input_tensor_a = values_a[torch.randint(0, len(values_a), (num_elements_a,))]
    else:
        torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements_a, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements_a].reshape(a_shape).nan_to_num(0.0)

    num_elements_b = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements_b, dtype=torch.int32)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements_b].reshape(b_shape).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn.mul)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b, use_legacy=None)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


def test_binary_mul_uint16_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 32767, 65535, 65535])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 1, 2, 1, 0])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.mul)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)
    output_tensor = ttnn.mul(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
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
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
    ],
)
@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (0, 100, 0, 300),
        (1000, 10000, 500, 1000),
        (30000, 40000, 10000, 15000),
        (50000, 55000, 1000, 2000),
        (0, 32000, 0, 32000),
    ],
)
def test_binary_logical_uint16_bcast(a_shape, b_shape, ttnn_op, low_a, high_a, low_b, high_b, device):
    num_elements = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(low_a, high_a, num_elements, dtype=torch.int32)
    torch_input_tensor_a[::5] = 0  # every 5th element is zero
    torch_input_tensor_a = torch_input_tensor_a[:num_elements].reshape(a_shape)

    num_elements = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(low_b, high_b, num_elements, dtype=torch.int32)
    torch_input_tensor_b[::10] = 0  # every 10th element is zero
    torch_input_tensor_b = torch_input_tensor_b[:num_elements].reshape(b_shape)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, use_legacy=None)
    # Since to_torch converts ttnn.uint16 to int16 and then to int32, there will be an output mismatch
    # We can typecast ttnn.uint16 to ttnn.uint32 to avoid this mismatch
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.logical_or,
        ttnn.logical_xor,
        ttnn.logical_and,
    ],
)
@pytest.mark.parametrize("use_legacy", [True, False])
def test_binary_logical_uint16_edge_cases(ttnn_op, use_legacy, device):
    torch_input_tensor_a = torch.tensor([0, 1, 0, 32767, 65534, 65535])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 0, 1, 32767, 0, 65535])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)
    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, use_legacy=use_legacy)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
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
        (10, 100, 0, 9),
        (305, 400, 150, 300),
    ],
)
def test_binary_squared_difference_uint16_bcast(a_shape, b_shape, low_a, high_a, low_b, high_b, device):
    num_elements_a = max(int(torch.prod(torch.tensor(a_shape)).item()), 1)
    torch_input_tensor_a = torch.linspace(high_a, low_a, num_elements_a, dtype=torch.int32)
    num_elements_b = max(int(torch.prod(torch.tensor(b_shape)).item()), 1)
    torch_input_tensor_b = torch.linspace(high_b, low_b, num_elements_b, dtype=torch.int32)
    torch_input_tensor_a = torch_input_tensor_a[:num_elements_a].reshape(a_shape).nan_to_num(0.0)
    torch_input_tensor_b = torch_input_tensor_b[:num_elements_b].reshape(b_shape).nan_to_num(0.0)

    golden_function = ttnn.get_golden_function(ttnn.squared_difference)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.squared_difference(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    assert torch.equal(output_tensor, torch_output_tensor)


@pytest.mark.parametrize(
    "low_a, high_a, low_b, high_b",
    [
        (0, 100, 101, 300),
        (0, 32000, 32001, 65535),
    ],
)
def test_binary_rsub_uint16(low_a, high_a, low_b, high_b, device):
    torch_input_tensor_a = torch.randint(low_a, high_a, torch.Size([1, 3, 320, 384]), dtype=torch.int32)
    torch_input_tensor_b = torch.randint(low_b, high_b, torch.Size([1, 3, 320, 384]), dtype=torch.int32)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.rsub(input_tensor_a, input_tensor_b)

    golden_function = ttnn.get_golden_function(ttnn.rsub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)
    assert torch.equal(output_tensor, torch_output_tensor)


def test_binary_rsub_uint16_edge_cases(device):
    torch_input_tensor_a = torch.tensor([0, 0, 2, 500, 1, 30000, 0, 65530, 65528])
    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_input_tensor_b = torch.tensor([0, 1, 11, 7727, 32767, 30000, 65535, 65535, 65535])
    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn.uint16,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    golden_function = ttnn.get_golden_function(ttnn.rsub)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, device=device)

    output_tensor = ttnn.rsub(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.typecast(output_tensor, dtype=ttnn.uint32)
    output_tensor = ttnn.to_torch(output_tensor, dtype=torch.int32)

    assert torch.equal(output_tensor, torch_output_tensor)


def test_binary_bitwise_left_shift(device):
    x_torch = torch.tensor([0, 1, 2, 3, 15, 31, 255, 127, 63, 31, 15, 1], dtype=torch.int32)

    y_torch = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15], dtype=torch.int32)

    golden_fn = ttnn.get_golden_function(ttnn.bitwise_left_shift)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.bitwise_left_shift(x_tt, y_tt)

    z_tt_out = ttnn.typecast(z_tt_out, dtype=ttnn.uint32)
    tt_out = ttnn.to_torch(z_tt_out, dtype=torch.int32)
    assert torch.equal(tt_out, z_torch)


def test_binary_bitwise_right_shift(device):
    x_torch = torch.tensor(
        [0, 1, 2, 3, 15, 31, 255, 127, 63, 31, 15, 1, 65535, 32768, 16384, 8192, 1], dtype=torch.int32
    )

    y_torch = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 0, 1, 1, 13, 20], dtype=torch.int32)

    golden_fn = ttnn.get_golden_function(ttnn.bitwise_right_shift)
    z_torch = golden_fn(x_torch, y_torch)
    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt_out = ttnn.bitwise_right_shift(x_tt, y_tt)

    z_tt_out = ttnn.typecast(z_tt_out, dtype=ttnn.uint32)
    tt_out = ttnn.to_torch(z_tt_out, dtype=torch.int32)
    assert torch.equal(tt_out, z_torch)
