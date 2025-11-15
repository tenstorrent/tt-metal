# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose
from tests.ttnn.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range


@pytest.mark.parametrize(
    "op,rtol,atol",
    [
        ("reciprocal", 0.01, 0.01),
        ("neg", 0, 0),
        ("exp", 0.01, 0.02),
        ("atan", 0.01, 0.005),
    ],
)
@pytest.mark.parametrize("shape", [[320, 384]])
def test_unary_expression(device, op, rtol, atol, shape):
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    expression_op = getattr(ttnn.experimental.expression, op)

    torch_output = torch_op(torch_input)
    ttnn_output = ttnn.experimental.materialize(expression_op(ttnn_input))

    assert_allclose(torch_output, ttnn_output, rtol, atol)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "pow", "atan2"])
@pytest.mark.parametrize("shape", [[320, 384]])
def test_binary_expression(device, op, shape):
    torch.manual_seed(42)

    torch_a = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    torch_b = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    if op == "div":
        torch_b += 1
    elif op == "pow":
        torch_a += 1
    ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    expression_op = getattr(ttnn.experimental.expression, op)

    torch_output = torch_op(torch_a, torch_b)
    ttnn_output = ttnn.experimental.materialize(expression_op(ttnn_a, ttnn_b))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0.02)


@pytest.mark.parametrize("shape", [[320, 384]])
def test_logical_not_expression(device, shape):
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_output = torch.logical_not(torch_input).bfloat16()
    ttnn_output = ttnn.experimental.materialize(ttnn.experimental.expression.logical_not(ttnn_input))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["logical_and", "logical_or", "logical_xor"])
@pytest.mark.parametrize("shape", [[320, 384]])
def test_binary_logical_expression(device, op, shape):
    torch.manual_seed(42)

    torch_a = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    torch_b = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    expression_op = getattr(ttnn.experimental.expression, op)

    torch_output = torch_op(torch_a, torch_b).bfloat16()
    ttnn_output = ttnn.experimental.materialize(expression_op(ttnn_a, ttnn_b))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("shape", [[320, 384]])
def test_unary_compare_expression(device, op, shape):
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    expression_op = getattr(ttnn.experimental.expression, f"{op}z")

    torch_output = torch_op(torch_input, 0).bfloat16()
    ttnn_output = ttnn.experimental.materialize(expression_op(ttnn_input))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("shape", [[320, 384]])
def test_binary_compare_expression(device, op, shape):
    torch.manual_seed(42)

    torch_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_b = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    expression_op = getattr(ttnn.experimental.expression, op)

    torch_output = torch_op(torch_a, torch_b).bfloat16()
    ttnn_output = ttnn.experimental.materialize(expression_op(ttnn_a, ttnn_b))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [[320, 384]])
def test_where_expression(device, shape):
    torch.manual_seed(42)

    torch_cond = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_other = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_cond = ttnn.from_torch(torch_cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_other = ttnn.from_torch(torch_other, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_output = torch.where(torch_cond, torch_input, torch_other)
    ttnn_output = ttnn.experimental.materialize(ttnn.experimental.expression.where(ttnn_cond, ttnn_input, ttnn_other))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_atan2_expression(input_shapes, device):
    torch.manual_seed(42)

    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -150, 150, device)

    torch_output = torch.atan2(in_data1, in_data2)
    ttnn_output = ttnn.experimental.materialize(ttnn.experimental.expression.atan2(input_tensor1, input_tensor2))

    assert_allclose(torch_output, ttnn_output, rtol=0.01, atol=0)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("value", [1.0, 5.0, 10.0])
def test_addcmul_expression(input_shapes, value, device):
    torch.manual_seed(42)

    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)
    in_data2, input_tensor2 = data_gen_with_range(input_shapes, -80, 80, device)
    in_data3, input_tensor3 = data_gen_with_range(input_shapes, -90, 90, device)

    torch_output = torch.addcmul(in_data1, in_data2, in_data3, value=value)
    ttnn_output = ttnn.experimental.materialize(
        ttnn.experimental.expression.add(
            input_tensor1,
            ttnn.experimental.expression.mul(value, ttnn.experimental.expression.mul(input_tensor2, input_tensor3)),
        )
    )

    assert_allclose(torch_output, ttnn_output, rtol=0.015, atol=0)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize("threshold", [1.0, 10.0, 100.0, -5, -8.0, -100.0])
@pytest.mark.parametrize("value", [10.0, 100.0, -7.0, -85.5])
def test_threshold_expression(input_shapes, threshold, value, device):
    torch.manual_seed(42)

    in_data1, input_tensor1 = data_gen_with_range(input_shapes, -100, 100, device)

    torch_threshold = torch.nn.Threshold(threshold, value)
    torch_output = torch_threshold(in_data1)
    ttnn_output = ttnn.experimental.materialize(
        ttnn.experimental.expression.where(
            ttnn.experimental.expression.gt(input_tensor1, threshold), input_tensor1, value
        )
    )

    assert_allclose(torch_output, ttnn_output)
