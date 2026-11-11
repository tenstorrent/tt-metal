# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.subtract on INT8: the result keeps the input dtype and wraps modulo 256, as torch.sub does."""

import pytest
import torch
import ttnn


def _to_device(torch_tensor, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch_tensor, dtype=ttnn.int8, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        (torch.Size([256, 256]), torch.Size([256, 256])),  # full int8 range
        (torch.Size([1, 1, 32, 32]), torch.Size([1, 1, 32, 32])),
        (torch.Size([5, 32, 32]), torch.Size([32, 32])),
        (torch.Size([1, 2, 32]), torch.Size([1, 2, 32])),
        (torch.Size([1]), torch.Size([1, 5, 12])),
        (torch.Size([1, 2, 32, 64, 125]), torch.Size([1, 2, 32, 1, 1])),
        (torch.Size([]), torch.Size([])),
        (torch.Size([5]), torch.Size([1])),
    ],
)
def test_binary_sub_int8(device, a_shape, b_shape):
    if a_shape == b_shape == torch.Size([256, 256]):
        values = torch.arange(-128, 128, dtype=torch.int32)
        torch_input_tensor_a = values.reshape(256, 1).expand(256, 256).to(torch.int8).contiguous()
        torch_input_tensor_b = values.reshape(1, 256).expand(256, 256).to(torch.int8).contiguous()
    else:
        torch.manual_seed(0)
        torch_input_tensor_a = torch.randint(-128, 128, a_shape, dtype=torch.int32).to(torch.int8)
        torch_input_tensor_b = torch.randint(-128, 128, b_shape, dtype=torch.int32).to(torch.int8)
    expected = torch.sub(torch_input_tensor_a, torch_input_tensor_b)
    assert expected.dtype == torch.int8

    result = ttnn.subtract(_to_device(torch_input_tensor_a, device), _to_device(torch_input_tensor_b, device))
    assert result.dtype == ttnn.int8
    assert torch.equal(ttnn.to_torch(result), expected)


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("scalar_pos", ["rhs", "lhs"])
def test_sub_int8_scalar(device, scalar_pos, layout):
    values = torch.arange(-128, 128, dtype=torch.int32)
    torch_input_tensor_a = values.reshape(8, 32).to(torch.int8).contiguous()
    a = _to_device(torch_input_tensor_a, device, layout)
    for scalar in sorted({-128, -127, 0, 1, 5, 126, 127}):
        if scalar_pos == "rhs":
            expected = torch.sub(torch_input_tensor_a, torch.tensor(scalar, dtype=torch.int8))
            result = ttnn.subtract(a, scalar)
        else:
            expected = torch.sub(torch.tensor(scalar, dtype=torch.int8), torch_input_tensor_a)
            result = ttnn.subtract(scalar, a)
        assert result.dtype == ttnn.int8
        assert torch.equal(ttnn.to_torch(result), expected), f"scalar={scalar}"


@pytest.mark.parametrize("scalar", [float(256), -129.0, 2.5, float("inf")])
def test_sub_int8_overflow_scalar(device, expect_error, scalar):
    a = _to_device(torch.zeros((1, 1, 32, 32), dtype=torch.int8), device)
    with expect_error(RuntimeError, r"cannot represent the scalar"):
        ttnn.subtract(a, scalar)


BCAST_SHAPES = [
    pytest.param((1, 2, 64, 96), (1, 2, 1, 96), id="row_b"),
    pytest.param((1, 2, 1, 96), (1, 2, 64, 96), id="row_a"),
    pytest.param((1, 2, 64, 96), (1, 2, 64, 1), id="col_b"),
    pytest.param((1, 2, 64, 1), (1, 2, 64, 96), id="col_a"),
    pytest.param((1, 2, 64, 96), (1, 1, 1, 1), id="scalar_b"),
    pytest.param((1, 1, 1, 1), (1, 2, 64, 96), id="scalar_a"),
    pytest.param((1, 2, 1, 96), (1, 2, 64, 1), id="row_a_col_b"),
    pytest.param((1, 2, 64, 1), (1, 2, 1, 96), id="row_b_col_a"),
    pytest.param((1, 1, 33, 70), (1, 1, 1, 70), id="row_b_unaligned"),
    pytest.param((1, 1, 33, 70), (1, 1, 33, 1), id="col_b_unaligned"),
    pytest.param((3, 1, 33, 70), (1, 1, 1, 1), id="scalar_b_unaligned"),
    pytest.param((5,), (1,), id="rank1_scalar"),
    pytest.param((1, 5, 12), (1,), id="rank1_scalar_b"),
]


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("a_shape, b_shape", BCAST_SHAPES)
def test_sub_int8_subtile_broadcast(device, layout, a_shape, b_shape):
    torch.manual_seed(3)
    torch_input_tensor_a = torch.randint(-128, 128, a_shape, dtype=torch.int32).to(torch.int8)
    torch_input_tensor_b = torch.randint(-128, 128, b_shape, dtype=torch.int32).to(torch.int8)
    expected = torch.sub(torch_input_tensor_a, torch_input_tensor_b)

    a = _to_device(torch_input_tensor_a, device, layout)
    b = _to_device(torch_input_tensor_b, device, layout)
    result = ttnn.subtract(a, b)
    assert result.dtype == ttnn.int8
    assert torch.equal(ttnn.to_torch(result), expected)


SHARD_CASES = [
    pytest.param((256, 128), (256, 128), ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=2, x=1), id="height"),
    pytest.param((128, 256), (128, 256), ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(y=1, x=2), id="width"),
    pytest.param((256, 256), (256, 256), ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(y=2, x=2), id="block"),
]


@pytest.mark.parametrize("a_shape, b_shape, strategy, core_grid", SHARD_CASES)
def test_sub_int8_sharded(device, a_shape, b_shape, strategy, core_grid):
    torch.manual_seed(4)
    torch_input_tensor_a = torch.randint(-128, 128, a_shape, dtype=torch.int32).to(torch.int8)
    torch_input_tensor_b = torch.randint(-128, 128, b_shape, dtype=torch.int32).to(torch.int8)
    mem_config = ttnn.create_sharded_memory_config(a_shape, core_grid, strategy)
    a = _to_device(torch_input_tensor_a, device)
    b = _to_device(torch_input_tensor_b, device)
    a = ttnn.to_memory_config(a, mem_config)
    b = ttnn.to_memory_config(b, mem_config)

    result = ttnn.subtract(a, b)
    assert result.memory_config().is_sharded()
    assert torch.equal(ttnn.to_torch(result), torch.sub(torch_input_tensor_a, torch_input_tensor_b))

    scalar = -7
    result = ttnn.subtract(a, scalar)
    assert torch.equal(ttnn.to_torch(result), torch.sub(torch_input_tensor_a, torch.tensor(scalar, dtype=torch.int8)))
