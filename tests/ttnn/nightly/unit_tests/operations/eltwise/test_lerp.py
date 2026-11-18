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


@pytest.mark.parametrize("weight", [0.0, 1.0])
@pytest.mark.parametrize("use_scalar_weight", [True, False])
@pytest.mark.parametrize("input_dtype", ["bfloat16", "float32"])
def test_lerp_endpoint_passthrough(device, weight, use_scalar_weight, input_dtype):
    """weight == 0 returns input and weight == 1 returns end, bit-exact, across the finite range."""
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

    calculated = _run_lerp_exact_inputs(device, torch_input, torch_end, weight, use_scalar_weight)

    expected = torch_input if weight == 0.0 else torch_end
    assert torch.equal(calculated, expected)
