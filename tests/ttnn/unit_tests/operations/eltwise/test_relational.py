# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_equal

pytestmark = pytest.mark.use_module_device


def run_relational_test(device, h, w, ttnn_function):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    # Cast bool→float because comp_equal uses subtraction which doesn't support bool tensors
    assert_equal(torch_output_tensor.float(), output_tensor.float())


def run_relational_z_test(device, h, w, ttnn_function):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor.float(), output_tensor.float())


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gtz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.gtz)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gt(device, h, w):
    run_relational_test(device, h, w, ttnn.gt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ltz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.ltz)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ge(device, h, w):
    run_relational_test(device, h, w, ttnn.ge)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.gez)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lt(device, h, w):
    run_relational_test(device, h, w, ttnn.lt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.lez)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_le(device, h, w):
    run_relational_test(device, h, w, ttnn.le)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_eqz(device, h, w):
    run_relational_z_test(device, h, w, ttnn.eqz)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_eq(device, h, w):
    run_relational_test(device, h, w, ttnn.eq)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nez(device, h, w):
    run_relational_z_test(device, h, w, ttnn.nez)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ne(device, h, w):
    run_relational_test(device, h, w, ttnn.ne)


def run_relational_test_with_scalar(device, h, w, scalar, ttnn_function):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, scalar)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.full((h, w), fill_value=scalar, device=device, layout=ttnn.TILE_LAYOUT)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_equal(torch_output_tensor.float(), output_tensor.float())


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_gt(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.gt)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_ge(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.ge)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_lt(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.lt)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_le(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.le)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_eq(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.eq)


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_scalarB_ne(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.ne)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_gt(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.gt)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_ge(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.ge)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_lt(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.lt)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_le(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.le)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_eq(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.eq)


@pytest.mark.parametrize("scalar", [-1])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_nscalarB_ne(device, h, w, scalar):
    run_relational_test_with_scalar(device, h, w, scalar, ttnn.ne)


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast(device, h, w):
    torch_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_b = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.lt)
    torch_output = golden_function(torch_a, torch_b)

    a = ttnn.from_torch(torch_a, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, layout=ttnn.TILE_LAYOUT, device=device)
    tt_output = ttnn.lt(a, b)
    tt_output = ttnn.to_torch(tt_output)

    assert_equal(torch_output.float(), tt_output.float())


@pytest.mark.parametrize("h", [500])
@pytest.mark.parametrize("w", [512])
def test_expand_and_broadcast_reversed(device, h, w):
    torch_input_tensor_a = torch.rand((1, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
    golden_function = ttnn.get_golden_function(ttnn.lt)
    torch_output = golden_function(torch_input_tensor_b, torch_input_tensor_a)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.lt(input_tensor_b, input_tensor_a)
    output = ttnn.to_torch(output)

    assert_equal(torch_output.float(), output.float())


@pytest.mark.parametrize("atol", [1e-8, 1e-10])
@pytest.mark.parametrize("rtol", [1e-5, 1e-9])
@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_isclose(device, h, w, atol, rtol):
    torch.manual_seed(0)

    torch_input_tensor_a = torch.randn((1, 1, h, w), dtype=torch.bfloat16)
    torch_input_tensor_b = torch.randn((1, 1, h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn.isclose)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, rtol=rtol, atol=atol)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn.isclose(input_tensor_a, input_tensor_b, rtol=rtol, atol=atol)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output_tensor.float(), output_tensor.float())


@pytest.mark.parametrize(
    "rtol, atol",
    [(1e-05, 1e-08), (0.01, 5), (0.05, 10), (1e-04, 0)],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 1, 768, 456]),
    ],
)
def test_isclose_int32(device, input_shapes, rtol, atol):
    torch.manual_seed(0)

    x_torch = torch.randint(-2_000_000, 2_000_000, input_shapes, dtype=torch.int32)
    delta = torch.randint(-200, 200, input_shapes, dtype=torch.int32)
    y_torch = x_torch + delta

    z_torch = torch.isclose(x_torch.float(), y_torch.float(), rtol=rtol, atol=atol)

    x_tt = ttnn.from_torch(x_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = ttnn.from_torch(y_torch, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.isclose(x_tt, y_tt, rtol=rtol, atol=atol)
    tt_out = ttnn.to_torch(z_tt)

    assert torch.equal(z_torch, tt_out.bool())


@pytest.mark.parametrize(
    "rtol, atol",
    [(1e-05, 1e-08), (1e-04, 0), (1e-3, 1e-6), (1e-1, 5e-1)],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 320, 384]),
        torch.Size([1, 1, 768, 456]),
    ],
)
@pytest.mark.parametrize(
    "a_dtype, b_dtype",
    [
        (ttnn.int32, ttnn.bfloat16),
        (ttnn.bfloat16, ttnn.int32),
    ],
)
def test_isclose_int32_mixed_dtype(device, input_shapes, rtol, atol, a_dtype, b_dtype):
    """Mixed-dtype coverage: verifies that every (int32 / bfloat16) pairing
    that flows through invoke_binary_ng_isclose produces results matching a
    float-based torch.isclose reference. Pairs containing INT32 exercise the
    INT32->FLOAT32 pre-promotion path; pure-float pairs verify the no-promotion
    fast path."""
    torch.manual_seed(0)

    ttnn_to_torch_dtype = {
        ttnn.int32: torch.int32,
        ttnn.bfloat16: torch.bfloat16,
    }

    x_int = torch.randint(-1000, 1000, input_shapes, dtype=torch.int32)
    delta = torch.randint(-3, 3, input_shapes, dtype=torch.int32)
    y_int = x_int + delta

    a_torch = x_int.to(ttnn_to_torch_dtype[a_dtype])
    b_torch = y_int.to(ttnn_to_torch_dtype[b_dtype])

    z_torch = torch.isclose(a_torch.float(), b_torch.float(), rtol=rtol, atol=atol)

    a_tt = ttnn.from_torch(a_torch, dtype=a_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b_torch, dtype=b_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.isclose(a_tt, b_tt, rtol=rtol, atol=atol)
    tt_out = ttnn.to_torch(z_tt)

    assert torch.equal(z_torch, tt_out.bool())


@pytest.mark.parametrize("equal_nan", [True, False])
@pytest.mark.parametrize(
    "input_shapes",
    [
        torch.Size([1, 1, 32, 32]),
        torch.Size([1, 1, 64, 128]),
    ],
)
def test_isclose_bfloat16_equal_nan(device, input_shapes, equal_nan):
    """Validate equal_nan semantics on bfloat16 inputs against torch.isclose."""
    torch.manual_seed(0)

    a = torch.randn(input_shapes, dtype=torch.bfloat16)
    b = a.clone()

    nan = float("nan")
    a[0, 0, 0, 0] = nan
    b[0, 0, 0, 0] = nan
    a[0, 0, 0, 1] = nan
    a[0, 0, 0, 2] = 1.0
    b[0, 0, 0, 2] = nan

    z_torch = torch.isclose(a.float(), b.float(), rtol=1e-5, atol=1e-8, equal_nan=equal_nan)

    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.isclose(a_tt, b_tt, rtol=1e-5, atol=1e-8, equal_nan=equal_nan)
    tt_out = ttnn.to_torch(z_tt)

    assert torch.equal(z_torch, tt_out.bool())


@pytest.mark.parametrize("shape", [torch.Size([1, 1, 32, 32])])
def test_isclose_zero_tolerance(device, shape):
    """With rtol=atol=0 only bit-identical values should compare as close."""
    torch.manual_seed(0)
    a = torch.randn(shape, dtype=torch.bfloat16)
    b = a.clone()
    b[0, 0, 0, 0] = b[0, 0, 0, 0] + torch.tensor(0.001, dtype=torch.bfloat16)

    z_torch = torch.isclose(a.float(), b.float(), rtol=0.0, atol=0.0)

    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    z_tt = ttnn.isclose(a_tt, b_tt, rtol=0.0, atol=0.0)

    assert torch.equal(z_torch, ttnn.to_torch(z_tt).bool())


def test_isclose_inf_divergence(device):
    # Hardware correctly returns False for unequal infinities, matching torch.isclose semantics.
    a = torch.tensor([[[[float("inf"), float("-inf")]]]]).to(torch.bfloat16)
    b = torch.tensor([[[[float("-inf"), float("inf")]]]]).to(torch.bfloat16)

    z_torch = torch.isclose(a.float(), b.float(), rtol=1e-5, atol=1e-8)

    a_tt = ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.isclose(a_tt, b_tt, rtol=1e-5, atol=1e-8)
    out = ttnn.to_torch(result)

    assert torch.equal(z_torch, out.bool())


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "range1, range2",
    [
        ((-5, 5), (-10, 10)),
        ((-100, 100), (-150, 150)),
        ((0, 1), (1, 2)),
        ((-1, 1), (-1, 1)),
    ],
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.eq,
        ttnn.ne,
        ttnn.lt,
        ttnn.le,
        ttnn.gt,
        ttnn.ge,
    ],
)
def test_binary_relational_ttnn(input_shapes, ttnn_function, range1, range2, device):
    torch.manual_seed(0)
    low1, high1 = range1
    low2, high2 = range2
    in_data1 = torch.randint(low1, high1, input_shapes, dtype=torch.int32)
    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    in_data2 = torch.randint(low2, high2, input_shapes, dtype=torch.int32)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_function(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data1, in_data2)
    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([64, 64])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.eq,
        ttnn.ne,
        ttnn.lt,
        ttnn.le,
        ttnn.gt,
        ttnn.ge,
    ],
)
def test_binary_relational_edge_case_ttnn(input_shapes, ttnn_function, device):
    torch.manual_seed(213919)

    # Generate a uniform range of values across the valid int32 range
    num_elements = torch.prod(torch.tensor(input_shapes)).item()
    uniform_values1 = torch.linspace(-2147483647, 2147483647, num_elements, dtype=torch.int32)
    uniform_values2 = torch.linspace(-2147483610, 2147483610, num_elements, dtype=torch.int32)

    corner_cases = torch.tensor([0, 1, -1, 2147483647, -2147483647], dtype=torch.int32)
    in_data1 = torch.cat([uniform_values1, corner_cases])
    in_data2 = torch.cat([uniform_values2, corner_cases])

    in_data1 = in_data1[-num_elements:].reshape(input_shapes)
    in_data2 = in_data2[-num_elements:].reshape(input_shapes)

    input_tensor1 = ttnn.from_torch(in_data1, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor2 = ttnn.from_torch(in_data2, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor1, input_tensor2)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data1, in_data2)

    output_tensor = ttnn.to_torch(output_tensor)

    assert torch.equal(golden_tensor, output_tensor)


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
    ),
)
@pytest.mark.parametrize(
    "ttnn_function",
    [
        ttnn.eq,
        ttnn.ne,
        ttnn.lt,
        ttnn.le,
        ttnn.gt,
        ttnn.ge,
    ],
)
@pytest.mark.parametrize("scalar", [-2, -1, 0, 1, 2])
def test_binary_relational_scalar_ttnn(device, input_shapes, scalar, ttnn_function):
    torch.manual_seed(0)
    in_data = torch.randint(-100, 100, input_shapes, dtype=torch.int32)
    input_tensor = ttnn.from_torch(in_data, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor, scalar)
    output_tensor = ttnn.to_torch(output_tensor)
    golden_function = ttnn.get_golden_function(ttnn_function)
    golden_tensor = golden_function(in_data, scalar)

    assert torch.equal(golden_tensor, output_tensor)
