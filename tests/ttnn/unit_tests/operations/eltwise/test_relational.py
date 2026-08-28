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


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_gt(device, h, w):
    run_relational_test(device, h, w, ttnn.gt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ge(device, h, w):
    run_relational_test(device, h, w, ttnn.ge)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_lt(device, h, w):
    run_relational_test(device, h, w, ttnn.lt)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_le(device, h, w):
    run_relational_test(device, h, w, ttnn.le)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_eq(device, h, w):
    run_relational_test(device, h, w, ttnn.eq)


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


# The isclose kernel classifies a lane as Inf/NaN by comparing the operand's abs
# *bit pattern* against 0x7F800000. Reading that constant as a float value instead
# yields the finite 2139095040.0, so coverage must include both Inf signs, NaN, and
# finite magnitudes that bracket 2139095040.0 from both sides.
_INF_BIT_PATTERN_AS_FLOAT = 2139095040.0  # float(0x7F800000)

_ISCLOSE_SPECIAL_VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    float("inf"),
    float("-inf"),
    float("nan"),
    float("-nan"),  # sign-bit-set NaN survives in float32; bfloat16 canonicalizes it
    2.0e9,  # below _INF_BIT_PATTERN_AS_FLOAT
    _INF_BIT_PATTERN_AS_FLOAT,
    3.0e9,  # above it
    -3.0e9,
    1.0e10,
    torch.finfo(torch.float32).max,
]


def _isclose_value_matrix(values, dtype):
    """Every ordered pair of `values`, packed into a single (1, 1, 32, 32) tile pair."""
    pairs = [(x, y) for x in values for y in values]
    slots = 32 * 32
    assert len(pairs) <= slots, f"{len(pairs)} pairs exceeds one {slots}-element tile"
    pad = slots - len(pairs)
    a = torch.tensor([p[0] for p in pairs] + [0.0] * pad, dtype=dtype).reshape(1, 1, 32, 32)
    b = torch.tensor([p[1] for p in pairs] + [0.0] * pad, dtype=dtype).reshape(1, 1, 32, 32)
    return a, b


def _assert_isclose_matches_torch(device, a, b, ttnn_dtype, rtol, atol, equal_nan=False):
    """Run ttnn.isclose and diff against torch.isclose on the same (post-rounding) values."""
    expected = torch.isclose(a.float(), b.float(), rtol=rtol, atol=atol, equal_nan=equal_nan)

    a_tt = ttnn.from_torch(a, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.isclose(a_tt, b_tt, rtol=rtol, atol=atol, equal_nan=equal_nan)).bool()

    if not torch.equal(expected, actual):
        exp_flat, act_flat = expected.flatten(), actual.flatten()
        a_flat, b_flat = a.float().flatten(), b.float().flatten()
        wrong = (exp_flat != act_flat).nonzero().flatten().tolist()
        detail = "\n".join(
            f"  a={a_flat[i].item():<24} b={b_flat[i].item():<24} "
            f"torch={exp_flat[i].item()} ttnn={act_flat[i].item()}"
            for i in wrong[:16]
        )
        raise AssertionError(f"{len(wrong)} mismatch(es) vs torch.isclose:\n{detail}")


@pytest.mark.parametrize("equal_nan", [True, False])
@pytest.mark.parametrize("ttnn_dtype, torch_dtype", [(ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)])
def test_isclose_special_values(device, ttnn_dtype, torch_dtype, equal_nan):
    """Full cross product of signed zeros, signed Inf, NaN and large finite magnitudes.

    Covers matching Inf (isclose(inf, inf) is True) and divergent Inf (False), plus
    finite values on both sides of the +Inf bit pattern read as a float.
    """
    a, b = _isclose_value_matrix(_ISCLOSE_SPECIAL_VALUES, torch_dtype)
    _assert_isclose_matches_torch(device, a, b, ttnn_dtype, rtol=1e-5, atol=1e-8, equal_nan=equal_nan)


@pytest.mark.parametrize("ttnn_dtype, torch_dtype", [(ttnn.float32, torch.float32), (ttnn.bfloat16, torch.bfloat16)])
def test_isclose_matching_infinities(device, ttnn_dtype, torch_dtype):
    """isclose(+inf, +inf) and isclose(-inf, -inf) are True; mismatched signs are False."""
    inf = float("inf")
    a = torch.tensor([[[[inf, -inf, inf, -inf]]]], dtype=torch_dtype)
    b = torch.tensor([[[[inf, -inf, -inf, inf]]]], dtype=torch_dtype)
    _assert_isclose_matches_torch(device, a, b, ttnn_dtype, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "magnitude",
    [
        2.0e9,
        _INF_BIT_PATTERN_AS_FLOAT,
        3.0e9,
        1.0e10,
        1.0e20,
        torch.finfo(torch.float32).max,
    ],
)
@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_isclose_large_finite_magnitudes(device, magnitude, sign):
    """Equal finite values stay close no matter how large the magnitude.

    Regression for treating the +Inf bit pattern as the float 2139095040.0, which
    misclassified every finite operand above that value as a special lane.
    """
    x = torch.full((1, 1, 32, 32), sign * magnitude, dtype=torch.float32)
    _assert_isclose_matches_torch(device, x, x.clone(), ttnn.float32, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("equal_nan", [True, False])
def test_isclose_negative_nan(device, equal_nan):
    """Sign-bit-set NaN must classify as NaN.

    SFPABS in float mode deliberately leaves -NaN as -NaN instead of clearing the
    sign bit (see the SFPABS functional model in tt-isa-documentation), so the
    kernel cannot derive the abs bit pattern from sfpi::abs() -- it has to mask.
    Built from raw bits because bfloat16 canonicalizes -NaN to +NaN.
    """
    neg_nan = torch.tensor([-4194304], dtype=torch.int32).view(torch.float32)  # 0xFFC00000
    pos_nan = torch.tensor([0x7FC00000], dtype=torch.int64).to(torch.int32).view(torch.float32)
    assert neg_nan.view(torch.int32).item() & 0xFFFFFFFF == 0xFFC00000, "expected sign-bit-set NaN"

    lhs = torch.tensor([neg_nan, neg_nan, neg_nan, pos_nan], dtype=torch.float32).reshape(1, 1, 1, 4)
    rhs = torch.tensor([neg_nan, pos_nan, 1.0, pos_nan], dtype=torch.float32).reshape(1, 1, 1, 4)
    _assert_isclose_matches_torch(device, lhs, rhs, ttnn.float32, rtol=1e-5, atol=1e-8, equal_nan=equal_nan)


def test_isclose_inf_bit_pattern_boundary(device):
    """Bit-exact ulp sweep of isclose(x, x) across float(0x7F800000).

    The three values are consecutive float32s straddling the constant, so a float
    magnitude compare against it changes answer mid-sweep while a bit compare
    does not.
    """
    boundary = _INF_BIT_PATTERN_AS_FLOAT
    ulp = 128.0  # spacing of float32 in [2^30, 2^31)
    values = [boundary - ulp, boundary, boundary + ulp]

    a = torch.tensor(values, dtype=torch.float32).repeat(32 * 32 // len(values) + 1)[: 32 * 32]
    a = a.reshape(1, 1, 32, 32)
    _assert_isclose_matches_torch(device, a, a.clone(), ttnn.float32, rtol=0.0, atol=0.0)


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
