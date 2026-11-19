# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp


def run_lerp_test(
    device,
    h,
    w,
    low,
    high,
    end,
    weight,
    ttnn_function,
    use_scalar_weight=False,
    ulp_threshold=1,
    input_dtype="bfloat16",
    output_dtype=None,
):
    torch_input_dtype = getattr(torch, input_dtype)

    torch_input_tensor_a = torch.linspace(low, high, steps=h * w, dtype=torch_input_dtype).reshape((h, w))
    torch_input_tensor_b = torch.full((h, w), end, dtype=torch_input_dtype)

    golden_function = ttnn.get_golden_function(ttnn_function)

    if use_scalar_weight:
        torch_weight = weight
        ttnn_weight = weight
    else:
        torch_weight = torch.full((h, w), weight, dtype=torch_input_dtype)
        ttnn_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    calculated_tensor = None
    if output_dtype is not None:
        torch_dtype = getattr(torch, output_dtype)
        ttnn_output_dtype = getattr(ttnn, output_dtype)
        torch_input_tensor_a = torch_input_tensor_a.to(torch_dtype)
        torch_input_tensor_b = torch_input_tensor_b.to(torch_dtype)
        calculated_tensor = ttnn.empty((h, w), dtype=ttnn_output_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    golden_output_tensor = golden_function(
        torch_input_tensor_a,
        torch_input_tensor_b,
        torch_weight,
    )

    calculated_tensor = ttnn_function(input_tensor_a, input_tensor_b, ttnn_weight, output_tensor=calculated_tensor)

    if output_dtype is not None:
        assert calculated_tensor.dtype == ttnn_output_dtype

    calculated_tensor = ttnn.to_torch(calculated_tensor)
    assert_with_ulp(expected_result=golden_output_tensor, actual_result=calculated_tensor, ulp_threshold=ulp_threshold)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.5])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_float_a(device, h, w, weight, input_dtype):
    run_lerp_test(device, h, w, 0, 90, 100, weight, ttnn.lerp, use_scalar_weight=True, input_dtype=input_dtype)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.75])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_float_b(device, h, w, weight, input_dtype):
    run_lerp_test(
        device, h, w, 1, 80, 99, weight, ttnn.lerp, use_scalar_weight=True, ulp_threshold=2, input_dtype=input_dtype
    )


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.5])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_tensor_a(device, h, w, weight, input_dtype):
    run_lerp_test(device, h, w, 0, 90, 100, weight, ttnn.lerp, use_scalar_weight=False, input_dtype=input_dtype)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.75])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_tensor_b(device, h, w, weight, input_dtype):
    run_lerp_test(
        device, h, w, 1, 80, 99, weight, ttnn.lerp, use_scalar_weight=False, ulp_threshold=2, input_dtype=input_dtype
    )


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [9472])
@pytest.mark.parametrize("weight", [0.75])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_fp32_preallocated_output(device, h, w, weight, input_dtype):
    """Lerp with bfloat16 inputs (two tensors + scalar weight) and preallocated float32 output.
    Checks that output is correct within 1 ULP for float32."""
    run_lerp_test(
        device,
        h,
        w,
        1,
        80,
        99,
        weight,
        ttnn.lerp,
        use_scalar_weight=True,
        ulp_threshold=1,
        output_dtype="float32",
        input_dtype=input_dtype,
    )


def _lerp_f64_golden(torch_input, torch_end, weight, torch_dtype):
    # torch.lerp forms (end - input) in the working precision and overflows with it, so the
    # reference is evaluated in float64 and rounded once to the output dtype.
    weight = weight.double() if torch.is_tensor(weight) else weight
    return ((1.0 - weight) * torch_input.double() + weight * torch_end.double()).to(torch_dtype)


def _run_lerp_exact_inputs(device, torch_input, torch_end, weight, use_scalar_weight):
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    end_tensor = ttnn.from_torch(torch_end, layout=ttnn.TILE_LAYOUT, device=device)
    if use_scalar_weight:
        ttnn_weight = weight
    else:
        torch_weight = torch.full(torch_input.shape, weight, dtype=torch_input.dtype)
        ttnn_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.lerp(input_tensor, end_tensor, ttnn_weight))


@pytest.mark.parametrize("magnitude", [2.0**127, 3.0e38])
@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("use_scalar_weight", [True, False])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_large_opposite_sign(device, magnitude, weight, use_scalar_weight, input_dtype):
    """|end - input| exceeds FLT_MAX although input, end and the interpolated value are all finite."""
    if not ttnn.device.is_blackhole(device):
        pytest.skip("Overflow-safe lerp is implemented on Blackhole only")
    torch_dtype = getattr(torch, input_dtype)
    h, w = 64, 64
    sign = torch.where(torch.arange(h * w) % 2 == 0, 1.0, -1.0).reshape((h, w))
    torch_input = (sign * magnitude).to(torch_dtype)
    torch_end = (-sign * magnitude).to(torch_dtype)

    calculated = _run_lerp_exact_inputs(device, torch_input, torch_end, weight, use_scalar_weight)

    assert torch.isfinite(calculated).all(), "lerp of finite inputs with weight in [0, 1] must stay finite"
    golden = _lerp_f64_golden(torch_input, torch_end, weight, torch_dtype)
    assert_with_ulp(expected_result=golden, actual_result=calculated, ulp_threshold=1)


@pytest.mark.parametrize("use_scalar_weight", [True, False])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_zero_weight_passthrough(device, use_scalar_weight, input_dtype):
    """weight == 0 returns input bit-exact across the finite range, also where end - input overflows."""
    if not ttnn.device.is_blackhole(device):
        pytest.skip("Overflow-safe lerp is implemented on Blackhole only")
    torch_dtype = getattr(torch, input_dtype)
    h, w = 64, 64
    torch.manual_seed(0)
    exponent = torch.randint(-120, 128, (2, h, w)).double()
    mantissa = torch.rand((2, h, w), dtype=torch.float64) + 1.0
    sign = torch.where(torch.rand((2, h, w)) < 0.5, -1.0, 1.0)
    values = (sign * mantissa * torch.pow(2.0, exponent)).to(torch_dtype)
    values = torch.where(torch.isfinite(values), values, torch.ones_like(values))
    torch_input, torch_end = values[0], values[1]

    calculated = _run_lerp_exact_inputs(device, torch_input, torch_end, 0.0, use_scalar_weight)

    assert torch.equal(calculated, torch_input)


FLT_MIN = 2.0**-126
WEIGHT_MODES = ["tensor", "scalar", "row_bcast", "col_bcast", "scalar_bcast"]
WEIGHT_SHAPES = {
    "row_bcast": lambda h, w: (1, w),
    "col_bcast": lambda h, w: (h, 1),
    "scalar_bcast": lambda h, w: (1, 1),
}


def _weight_arg(device, weight, weight_mode, shape, torch_dtype):
    """The weight as passed to ttnn.lerp and as a float64 golden operand, rounded to the input dtype."""
    if weight_mode == "scalar":
        return weight, torch.tensor(weight, dtype=torch_dtype).double()
    weight_shape = WEIGHT_SHAPES.get(weight_mode, lambda h, w: (h, w))(*shape)
    torch_weight = torch.full(weight_shape, weight, dtype=torch_dtype)
    return ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device), torch_weight.double()


def _run_lerp(device, torch_input, torch_end, ttnn_weight, **lerp_kwargs):
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    end_tensor = ttnn.from_torch(torch_end, layout=ttnn.TILE_LAYOUT, device=device)
    return ttnn.to_torch(ttnn.lerp(input_tensor, end_tensor, ttnn_weight, **lerp_kwargs))


def _ulp(magnitude, torch_dtype):
    """Spacing of torch_dtype at magnitude (float64), clamped to the finite normal range."""
    finfo = torch.finfo(torch_dtype)
    clamped = magnitude.abs().clamp(min=finfo.tiny, max=finfo.max)
    return torch.ldexp(torch.full_like(clamped, finfo.eps), torch.frexp(clamped).exponent - 1)


def _random_magnitudes(generator, shape, min_exponent, max_exponent, torch_dtype):
    exponent = torch.randint(min_exponent, max_exponent + 1, shape, generator=generator).double()
    mantissa = 1.0 + torch.rand(shape, generator=generator, dtype=torch.float64)
    return (mantissa * torch.pow(2.0, exponent)).clamp(max=torch.finfo(torch_dtype).max).to(torch_dtype)


def _assert_lerp_term_bound(torch_input, torch_end, golden_weight, calculated, torch_dtype, ulp_bound):
    """Finite where the exact result rounds to a finite value, there within ulp_bound ULP of
    max(|input|, |weight * (end - input)|); the exact ±inf where it does not."""
    a64, b64 = torch_input.double(), torch_end.double()
    term = golden_weight * (b64 - a64)
    exact = a64 + term
    golden = exact.to(torch_dtype)
    finite = torch.isfinite(golden).expand_as(calculated)
    assert torch.equal(
        calculated[~finite], golden.expand_as(calculated)[~finite]
    ), "expected ±inf where the exact overflows"
    assert torch.isfinite(calculated[finite]).all(), "lerp must stay finite where the exact result is finite"
    error = (calculated.double() - exact).abs() / _ulp(torch.maximum(a64.abs(), term.abs()), torch_dtype)
    worst = error[finite].max().item() if finite.any() else 0.0
    assert worst <= ulp_bound, f"|result - exact| reaches {worst:.3f} ULP of the larger term (bound {ulp_bound})"


@pytest.mark.parametrize("weight", [-1.0, 2.0, 1e30])
@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_extrapolation_large_opposite_sign(device, weight, weight_mode, input_dtype):
    """Extrapolation with input and end of opposite sign, magnitudes 2^-100 .. FLT_MAX: finite
    wherever the exact result is, within 2.5 ULP of the larger term, ±inf where the exact overflows."""
    if not ttnn.device.is_blackhole(device):
        pytest.skip("Overflow-safe lerp is implemented on Blackhole only")
    torch_dtype = getattr(torch, input_dtype)
    shape = (64, 64)
    generator = torch.Generator().manual_seed(0)
    sign = torch.where(torch.arange(shape[0] * shape[1]) % 2 == 0, 1.0, -1.0).reshape(shape).to(torch_dtype)
    torch_input = sign * _random_magnitudes(generator, shape, -100, 127, torch_dtype)
    torch_end = -sign * _random_magnitudes(generator, shape, -100, 127, torch_dtype)
    ttnn_weight, golden_weight = _weight_arg(device, weight, weight_mode, shape, torch_dtype)

    calculated = _run_lerp(device, torch_input, torch_end, ttnn_weight)

    _assert_lerp_term_bound(torch_input, torch_end, golden_weight, calculated, torch_dtype, 2.5)


# Blackhole evaluates 2 * (input / 2 + weight * (end / 2 - input / 2)). Below 2^-125 the halved
# values are subnormal and flush to zero, and the multiply-add drops a product whose factors'
# exponents sum below -127 (here weight and (end - input) / 2).
BOTTOM_RANGE_CASES = [
    # input / 2 and end / 2 flush: 0 where the exact is 1.18e-38.
    pytest.param(FLT_MIN, FLT_MIN * (1 + 2**-7), 0.5, 0.0, id="input_and_end_half_flush"),
    # input / 2 flushes: weight * end, the exact is 2^-125 + 2^-127.
    pytest.param(FLT_MIN, 2.0**-124, 0.5, 2.0**-125, id="input_half_flushes"),
    # weight 0 does not return an input in [2^-126, 2^-125).
    pytest.param(FLT_MIN, 1.0, 0.0, 0.0, id="zero_weight_bottom_binade"),
    # end / 2 flushes: weight 1 does not return an end in [2^-126, 2^-125).
    pytest.param(0.0, FLT_MIN, 1.0, 0.0, id="end_half_flushes"),
    # (end - input) / 2 = 2^-127 flushes: input, the exact is 2^-119 + 2^-125.
    pytest.param(2.0**-119, 2.0**-119 * (1 + 2**-7), 2.0, 2.0**-119, id="difference_flushes"),
    # The result 2^-126 is below 2^-125 at full scale.
    pytest.param(2.0**-125, 0.0, 0.5, 0.0, id="result_flushes"),
    # weight * (end - input) / 2 = 2^-128 is dropped: input, the exact is 2^-110 + 2^-127.
    pytest.param(2.0**-110, 2.0**-110 * (1 + 2**-7), 2.0**-10, 2.0**-110, id="product_dropped"),
    # Away from the bottom range the result is exact.
    pytest.param(1.0, 3.0, 0.25, 1.5, id="normal_range"),
]


@pytest.mark.parametrize("torch_input_value, torch_end_value, weight, expected", BOTTOM_RANGE_CASES)
@pytest.mark.parametrize("weight_mode", WEIGHT_MODES)
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_bottom_range(device, torch_input_value, torch_end_value, weight, expected, weight_mode, input_dtype):
    """Documents Blackhole lerp's flushes below 2^-125, see BOTTOM_RANGE_CASES."""
    if not ttnn.device.is_blackhole(device):
        pytest.skip("Half-scale lerp is implemented on Blackhole only")
    torch_dtype = getattr(torch, input_dtype)
    shape = (32, 64)
    torch_input = torch.full(shape, torch_input_value, dtype=torch_dtype)
    torch_end = torch.full(shape, torch_end_value, dtype=torch_dtype)
    ttnn_weight, _ = _weight_arg(device, weight, weight_mode, shape, torch_dtype)

    calculated = _run_lerp(device, torch_input, torch_end, ttnn_weight)

    assert torch.equal(calculated, torch.full(shape, expected, dtype=torch_dtype))
