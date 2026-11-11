# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""ttnn.subtract on UINT8 / INT8: the result keeps the input dtype and wraps modulo 256, as torch.sub does."""

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

DTYPES = [
    pytest.param(ttnn.uint8, torch.uint8, 0, 255, id="uint8"),
    pytest.param(ttnn.int8, torch.int8, -128, 127, id="int8"),
]


def _to_device(torch_tensor, dtype, device, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        torch_tensor, dtype=dtype, device=device, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


@pytest.mark.parametrize("ttnn_dtype, torch_dtype, low, high", DTYPES)
def test_sub_int8_all_value_pairs(device, ttnn_dtype, torch_dtype, low, high):
    # Row i of A is i and column j of B is j, so one 256x256 call covers every (a, b) pair, including every wrap.
    values = torch.arange(low, high + 1, dtype=torch.int32)
    torch_a = values.reshape(256, 1).expand(256, 256).to(torch_dtype).contiguous()
    torch_b = values.reshape(1, 256).expand(256, 256).to(torch_dtype).contiguous()
    expected = torch.sub(torch_a, torch_b)
    assert expected.dtype == torch_dtype

    result = ttnn.subtract(_to_device(torch_a, ttnn_dtype, device), _to_device(torch_b, ttnn_dtype, device))
    assert result.dtype == ttnn_dtype
    assert torch.equal(ttnn.to_torch(result), expected)


@pytest.mark.parametrize("ttnn_dtype, torch_dtype, low, high", DTYPES)
@pytest.mark.parametrize(
    "a_shape, b_shape",
    [
        ((1, 1, 32, 32), (1, 1, 32, 32)),
        ((2, 3, 64, 96), (2, 3, 64, 96)),
        ((1, 2, 33, 70), (1, 2, 33, 70)),
        ((4, 1, 32, 64), (1, 5, 32, 64)),
        ((5, 32, 32), (32, 32)),
    ],
)
def test_sub_int8_shapes(device, ttnn_dtype, torch_dtype, low, high, a_shape, b_shape):
    torch.manual_seed(0)
    torch_a = torch.randint(low, high + 1, a_shape, dtype=torch.int32).to(torch_dtype)
    torch_b = torch.randint(low, high + 1, b_shape, dtype=torch.int32).to(torch_dtype)
    expected = torch.sub(torch_a, torch_b)

    result = ttnn.subtract(_to_device(torch_a, ttnn_dtype, device), _to_device(torch_b, ttnn_dtype, device))
    assert torch.equal(ttnn.to_torch(result), expected)


@pytest.mark.parametrize("ttnn_dtype, torch_dtype, low, high", DTYPES)
@pytest.mark.parametrize("scalar_pos", ["rhs", "lhs"])
def test_sub_int8_scalar(device, ttnn_dtype, torch_dtype, low, high, scalar_pos):
    values = torch.arange(low, high + 1, dtype=torch.int32)
    torch_a = values.reshape(8, 32).to(torch_dtype).contiguous()
    a = _to_device(torch_a, ttnn_dtype, device)
    for scalar in sorted({low, low + 1, 0, 1, 5, high - 1, high}):
        if scalar_pos == "rhs":
            expected = torch.sub(torch_a, torch.tensor(scalar, dtype=torch_dtype))
            result = ttnn.subtract(a, scalar)
        else:
            expected = torch.sub(torch.tensor(scalar, dtype=torch_dtype), torch_a)
            result = ttnn.subtract(scalar, a)
        assert result.dtype == ttnn_dtype
        assert torch.equal(ttnn.to_torch(result), expected), f"scalar={scalar}"


@pytest.mark.parametrize("ttnn_dtype, torch_dtype, low, high", DTYPES)
def test_sub_int8_inplace(device, ttnn_dtype, torch_dtype, low, high):
    torch.manual_seed(1)
    torch_a = torch.randint(low, high + 1, (1, 1, 64, 64), dtype=torch.int32).to(torch_dtype)
    torch_b = torch.randint(low, high + 1, (1, 1, 64, 64), dtype=torch.int32).to(torch_dtype)
    a = _to_device(torch_a, ttnn_dtype, device)
    ttnn.subtract_(a, _to_device(torch_b, ttnn_dtype, device))
    assert torch.equal(ttnn.to_torch(a), torch.sub(torch_a, torch_b))


@pytest.mark.parametrize("ttnn_dtype, torch_dtype, low, high", DTYPES)
def test_sub_int8_row_major(device, ttnn_dtype, torch_dtype, low, high):
    torch.manual_seed(2)
    torch_a = torch.randint(low, high + 1, (1, 1, 32, 64), dtype=torch.int32).to(torch_dtype)
    torch_b = torch.randint(low, high + 1, (1, 1, 32, 64), dtype=torch.int32).to(torch_dtype)
    a = _to_device(torch_a, ttnn_dtype, device, ttnn.ROW_MAJOR_LAYOUT)
    b = _to_device(torch_b, ttnn_dtype, device, ttnn.ROW_MAJOR_LAYOUT)
    result = ttnn.subtract(a, b)
    assert torch.equal(ttnn.to_torch(result), torch.sub(torch_a, torch_b))


@pytest.mark.parametrize("ttnn_dtype, torch_dtype, low, high", DTYPES)
@pytest.mark.parametrize("scalar", [float(256), -129.0, 2.5, float("inf")])
def test_sub_int8_rejects_unrepresentable_scalar(device, expect_error, ttnn_dtype, torch_dtype, low, high, scalar):
    a = _to_device(torch.zeros((1, 1, 32, 32), dtype=torch_dtype), ttnn_dtype, device)
    with expect_error(RuntimeError, r"cannot represent the scalar"):
        ttnn.subtract(a, scalar)


# Every sub-tile broadcast kind binary_ng distinguishes: row, column and scalar, on either operand, and the mixed
# row x column pairs. The tile-aligned and the unaligned shapes exercise the fill helpers differently.
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


@pytest.mark.parametrize("ttnn_dtype, torch_dtype, low, high", DTYPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("a_shape, b_shape", BCAST_SHAPES)
def test_sub_int8_subtile_broadcast(device, ttnn_dtype, torch_dtype, low, high, layout, a_shape, b_shape):
    torch.manual_seed(3)
    torch_a = torch.randint(low, high + 1, a_shape, dtype=torch.int32).to(torch_dtype)
    torch_b = torch.randint(low, high + 1, b_shape, dtype=torch.int32).to(torch_dtype)
    expected = torch.sub(torch_a, torch_b)

    a = _to_device(torch_a, ttnn_dtype, device, layout)
    b = _to_device(torch_b, ttnn_dtype, device, layout)
    result = ttnn.subtract(a, b)
    assert result.dtype == ttnn_dtype
    assert torch.equal(ttnn.to_torch(result), expected)


SHARD_CASES = [
    pytest.param((256, 128), (256, 128), ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=2, x=1), id="height"),
    pytest.param((128, 256), (128, 256), ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(y=1, x=2), id="width"),
    pytest.param((256, 256), (256, 256), ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(y=2, x=2), id="block"),
]


@pytest.mark.parametrize("ttnn_dtype, torch_dtype, low, high", DTYPES)
@pytest.mark.parametrize("a_shape, b_shape, strategy, core_grid", SHARD_CASES)
def test_sub_int8_sharded(device, ttnn_dtype, torch_dtype, low, high, a_shape, b_shape, strategy, core_grid):
    torch.manual_seed(4)
    torch_a = torch.randint(low, high + 1, a_shape, dtype=torch.int32).to(torch_dtype)
    torch_b = torch.randint(low, high + 1, b_shape, dtype=torch.int32).to(torch_dtype)
    mem_config = ttnn.create_sharded_memory_config(a_shape, core_grid, strategy)
    a = _to_device(torch_a, ttnn_dtype, device)
    b = _to_device(torch_b, ttnn_dtype, device)
    a = ttnn.to_memory_config(a, mem_config)
    b = ttnn.to_memory_config(b, mem_config)

    result = ttnn.subtract(a, b)
    assert result.memory_config().is_sharded()
    assert torch.equal(ttnn.to_torch(result), torch.sub(torch_a, torch_b))

    scalar = 7 if torch_dtype == torch.uint8 else -7
    result = ttnn.subtract(a, scalar)
    assert torch.equal(ttnn.to_torch(result), torch.sub(torch_a, torch.tensor(scalar, dtype=torch_dtype)))


def test_sub_int8_rejects_mixed_int8_uint8(device, expect_error):
    a = _to_device(torch.zeros((1, 1, 32, 32), dtype=torch.int8), ttnn.int8, device)
    b = _to_device(torch.zeros((1, 1, 32, 32), dtype=torch.uint8), ttnn.uint8, device)
    with expect_error(RuntimeError, r"Mixed dtype is not supported for binary operation"):
        ttnn.subtract(a, b)
