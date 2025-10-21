# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_allclose


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
def test_lazy_unary(device, op, rtol, atol, shape):
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16) + 0.5
    ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    lazy_op = getattr(ttnn.lazy, op)

    torch_output = torch_op(torch_input)
    ttnn_output = ttnn.materialize(lazy_op(ttnn_input))

    assert_allclose(torch_output, ttnn_output, rtol, atol)


@pytest.mark.parametrize("op", ["add", "sub", "mul", "div", "pow", "atan2"])
@pytest.mark.parametrize("shape", [[320, 384]])
def test_lazy_binary(device, op, shape):
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
    lazy_op = getattr(ttnn.lazy, op)

    torch_output = torch_op(torch_a, torch_b)
    lazy_output = ttnn.materialize(lazy_op(ttnn_a, ttnn_b))

    assert_allclose(torch_output, lazy_output, rtol=0, atol=0.02)


@pytest.mark.parametrize("shape", [[320, 384]])
def test_lazy_logical_not(device, shape):
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_output = torch.logical_not(torch_input).bfloat16()
    ttnn_output = ttnn.materialize(ttnn.lazy.logical_not(ttnn_input))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["logical_and", "logical_or", "logical_xor"])
@pytest.mark.parametrize("shape", [[320, 384]])
def test_lazy_binary_logical(device, op, shape):
    torch.manual_seed(42)

    torch_a = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    torch_b = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    lazy_op = getattr(ttnn.lazy, op)

    torch_output = torch_op(torch_a, torch_b).bfloat16()
    ttnn_output = ttnn.materialize(lazy_op(ttnn_a, ttnn_b))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("shape", [[320, 384]])
def test_lazy_unary_compare(device, op, shape):
    torch.manual_seed(42)

    torch_input = torch.rand(shape, dtype=torch.bfloat16) - 0.5
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    lazy_op = getattr(ttnn.lazy, f"{op}z")

    torch_output = torch_op(torch_input, 0).bfloat16()
    ttnn_output = ttnn.materialize(lazy_op(ttnn_input))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("shape", [[320, 384]])
def test_lazy_binary_compare(device, op, shape):
    torch.manual_seed(42)

    torch_a = torch.rand(shape, dtype=torch.bfloat16)
    torch_b = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_op = getattr(torch, op)
    lazy_op = getattr(ttnn.lazy, op)

    torch_output = torch_op(torch_a, torch_b).bfloat16()
    ttnn_output = ttnn.materialize(lazy_op(ttnn_a, ttnn_b))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [[320, 384]])
def test_lazy_where(device, shape):
    torch.manual_seed(42)

    torch_cond = torch.rand(shape, dtype=torch.bfloat16) > 0.5
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    torch_other = torch.rand(shape, dtype=torch.bfloat16)
    ttnn_cond = ttnn.from_torch(torch_cond, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_input = ttnn.from_torch(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn_other = ttnn.from_torch(torch_other, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    torch_output = torch.where(torch_cond, torch_input, torch_other)
    ttnn_output = ttnn.materialize(ttnn.lazy.where(ttnn_cond, ttnn_input, ttnn_other))

    assert_allclose(torch_output, ttnn_output, rtol=0, atol=0)
