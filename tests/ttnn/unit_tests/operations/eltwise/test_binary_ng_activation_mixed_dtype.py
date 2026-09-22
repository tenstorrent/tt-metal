# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch
import ttnn

pytestmark = pytest.mark.use_module_device

_DTYPE_PAIRS = [
    pytest.param(ttnn.float32, ttnn.bfloat16, id="fp32-bf16"),
    pytest.param(ttnn.bfloat16, ttnn.float32, id="bf16-fp32"),
    pytest.param(ttnn.bfloat16, ttnn.bfloat16, id="bf16-bf16"),
]
_FULL = (1, 2, 32, 96)
_SHAPES = [
    pytest.param((1, 1, 32, 288), (1, 1, 32, 288), id="nine-tiles"),
    pytest.param((1, 1, 1, 96), _FULL, id="lhs-row"),
    pytest.param(_FULL, (1, 1, 1, 96), id="rhs-row"),
    pytest.param((1, 1, 32, 1), _FULL, id="lhs-column"),
    pytest.param(_FULL, (1, 1, 32, 1), id="rhs-column"),
    pytest.param((1, 1, 1, 1), _FULL, id="lhs-scalar-tensor"),
    pytest.param(_FULL, (1, 1, 1, 1), id="rhs-scalar-tensor"),
    pytest.param((1, 1, 1, 96), (1, 1, 64, 1), id="row-column"),
    pytest.param((1, 1, 64, 1), (1, 1, 1, 96), id="column-row"),
    pytest.param((1, 1, 32, 96), _FULL, id="batch"),
]


def _one_core():
    # Force repeated chunks instead of distributing each tile onto a separate core.
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})


def _input(device, shape, dtype, offset):
    host = ((torch.arange(math.prod(shape)) * 17 + offset) % 101 - 50).float().reshape(shape) / 8
    return ttnn.from_torch(host, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def _check_abs(op, a, b, side, **kwargs):
    lhs = side in ("lhs", "both")
    rhs = side in ("rhs", "both")
    reference_a = ttnn.abs(a) if lhs else a
    reference_b = (ttnn.abs(b) if isinstance(b, ttnn.Tensor) else abs(b)) if rhs else b
    grid = _one_core()
    expected = ttnn.to_torch(op(reference_a, reference_b, sub_core_grids=grid, **kwargs)).float()
    activation = [ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]
    actual = ttnn.to_torch(
        op(
            a,
            b,
            input_tensor_a_activations=activation if lhs else [],
            input_tensor_b_activations=activation if rhs else [],
            sub_core_grids=grid,
            **kwargs,
        )
    ).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("op", [ttnn.add, ttnn.multiply], ids=["add", "multiply"])
@pytest.mark.parametrize("a_dtype,b_dtype", _DTYPE_PAIRS[:2])
@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_fused_abs_formats_smoke(device, op, a_dtype, b_dtype, side):
    shape = (1, 1, 32, 288)
    _check_abs(op, _input(device, shape, a_dtype, 3), _input(device, shape, b_dtype, 13), side)


# Column, scalar-tensor, and combined row/column broadcast in both directions.
@pytest.mark.parametrize("a_shape,b_shape", _SHAPES[3:9])
def test_fused_abs_broadcast_smoke(device, a_shape, b_shape):
    a = _input(device, a_shape, ttnn.bfloat16, 3)
    b = _input(device, b_shape, ttnn.bfloat16, 13)
    _check_abs(ttnn.multiply, a, b, "both")


@pytest.mark.parametrize("op", [ttnn.add, ttnn.multiply], ids=["add", "multiply"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_fused_abs_python_scalar(device, op, dtype, side):
    _check_abs(op, _input(device, (1, 1, 32, 288), dtype, 3), -2.5, side)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("fast", [True, False], ids=["fast", "accurate"])
@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_fused_abs_scalar_first(device, dtype, fast, side):
    # Noncommutative op: logical LHS is the scalar, but physical c_0 is the tensor.
    # Fast block-float subtraction uses FPU with a BF16 scalar, so its SrcA starts
    # from c_1/c_4. FP32 dispatches to SFPU even in fast mode and still loads c_0 first.
    tensor = _input(device, (1, 1, 32, 288), dtype, 3)
    reference_tensor = ttnn.to_torch(tensor).float()
    scalar = -2.5
    lhs = side in ("lhs", "both")
    rhs = side in ("rhs", "both")
    expected = (abs(scalar) if lhs else scalar) - (reference_tensor.abs() if rhs else reference_tensor)
    activation = [ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)]
    actual = ttnn.to_torch(
        ttnn.subtract(
            scalar,
            tensor,
            fast_and_approximate_mode=fast,
            input_tensor_a_activations=activation if lhs else [],
            input_tensor_b_activations=activation if rhs else [],
            dtype=ttnn.bfloat16,
            sub_core_grids=_one_core(),
        )
    ).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("a_dtype,b_dtype", _DTYPE_PAIRS[:2])
@pytest.mark.parametrize("side", ["lhs", "rhs", "both"])
def test_fused_abs_sharded_multiply(device, a_dtype, b_dtype, side):
    # Nine tiles exercise complete SFPU chunks plus a remainder on one shard.
    shape = (1, 1, 32, 288)
    memory = ttnn.create_sharded_memory_config(
        shape, ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    a = ttnn.to_memory_config(_input(device, shape, a_dtype, 3), memory)
    b = ttnn.to_memory_config(_input(device, shape, b_dtype, 13), memory)
    _check_abs(ttnn.multiply, a, b, side, memory_config=memory)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
@pytest.mark.parametrize("both", [False, True])
@pytest.mark.parametrize("a_shape,b_shape", _SHAPES[:5])
def test_activation_intermediate_changes_format(device, dtype, both, a_shape, b_shape):
    a = _input(device, a_shape, dtype, 3)
    b = _input(device, b_shape, dtype, 13)
    reference_a, reference_b = ttnn.to_torch(a).float(), ttnn.to_torch(b).float()
    activation = [ttnn.UnaryWithParam(ttnn.UnaryOpType.ABS)] if both else []
    if both:
        reference_a, reference_b = reference_a.abs(), reference_b.abs()
    actual = ttnn.to_torch(
        ttnn.logaddexp(
            a,
            b,
            input_tensor_a_activations=activation,
            input_tensor_b_activations=activation,
            dtype=ttnn.bfloat16,
            sub_core_grids=_one_core(),
        )
    ).float()
    torch.testing.assert_close(actual, torch.logaddexp(reference_a, reference_b), rtol=0.03, atol=0.03)
