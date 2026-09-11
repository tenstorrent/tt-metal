# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Focused numerical coverage for the non-legacy sqrt / rsqrt / reciprocal paths.

Exercises approximate and precise modes (where exposed), BF16 and FP32 destinations,
edge cases (zero / negative / infinity / NaN), and representative input ranges.
"""

import math

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_ulp

pytestmark = pytest.mark.use_module_device


def _assert_matching_nonfinite(golden, actual, ulp_threshold=2):
    golden_nonfinite = ~torch.isfinite(golden)
    actual_nonfinite = ~torch.isfinite(actual)
    assert torch.equal(
        golden_nonfinite, actual_nonfinite
    ), "Non-finite mask mismatch between golden and device output"
    finite = torch.isfinite(golden) & torch.isfinite(actual)
    if finite.any():
        assert_with_ulp(
            expected_result=golden[finite],
            actual_result=actual[finite],
            ulp_threshold=ulp_threshold,
            allow_nonfinite=False,
        )


@pytest.mark.parametrize("ttnn_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize("ttnn_op", [ttnn.sqrt, ttnn.rsqrt])
def test_sqrt_rsqrt_ranges(device, ttnn_dtype, torch_dtype, approx, ttnn_op):
    torch.manual_seed(0)
    shape = (32, 64)
    if approx:
        torch_input = torch.empty(shape, dtype=torch_dtype).uniform_(1e-3, 100.0)
    else:
        # Precise mode also covers negatives for domain/edge behavior.
        torch_input = torch.empty(shape, dtype=torch_dtype).uniform_(-100.0, 100.0)

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_op(input_tensor, fast_and_approximate_mode=approx)
    golden = ttnn.get_golden_function(ttnn_op)(torch_input)
    actual = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    if approx:
        assert_with_ulp(expected_result=golden, actual_result=actual, ulp_threshold=2, allow_nonfinite=True)
    else:
        _assert_matching_nonfinite(golden, actual)


@pytest.mark.parametrize("ttnn_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
def test_reciprocal_ranges(device, ttnn_dtype, torch_dtype):
    torch.manual_seed(0)
    shape = (32, 64)
    torch_input = torch.empty(shape, dtype=torch_dtype).uniform_(-100.0, 100.0)
    # Keep values away from zero so the finite ULP check stays meaningful.
    torch_input = torch.where(
        torch_input.abs() < 1e-2,
        torch.copysign(torch.full_like(torch_input, 1e-2), torch_input),
        torch_input,
    )

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.reciprocal(input_tensor)
    golden = ttnn.get_golden_function(ttnn.reciprocal)(torch_input)
    actual = ttnn.to_torch(output_tensor, dtype=torch_dtype)
    _assert_matching_nonfinite(golden, actual)


@pytest.mark.parametrize("ttnn_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
@pytest.mark.parametrize("approx", [False, True])
@pytest.mark.parametrize("ttnn_op", [ttnn.sqrt, ttnn.rsqrt])
def test_sqrt_rsqrt_edge_cases(device, ttnn_dtype, torch_dtype, approx, ttnn_op):
    torch_input = torch.ones((32, 32), dtype=torch_dtype)
    specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        -2.0,
        2.0,
        1e-20,
        -1e-20,
        1e20,
        -1e20,
    ]
    for i, value in enumerate(specials):
        torch_input.view(-1)[i] = value

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn_op(input_tensor, fast_and_approximate_mode=approx)
    golden = ttnn.get_golden_function(ttnn_op)(torch_input)
    actual = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    for i, value in enumerate(specials):
        g = golden.view(-1)[i].item()
        a = actual.view(-1)[i].item()
        if math.isnan(g):
            assert math.isnan(a), f"{ttnn_op.__name__}({value}): expected NaN, got {a}"
        elif math.isinf(g):
            assert math.isinf(a) and math.copysign(1.0, a) == math.copysign(1.0, g), (
                f"{ttnn_op.__name__}({value}): expected {g}, got {a}"
            )
        elif math.isfinite(g) and math.isfinite(a):
            assert_with_ulp(
                expected_result=golden.view(-1)[i : i + 1],
                actual_result=actual.view(-1)[i : i + 1],
                ulp_threshold=8 if approx else 2,
                allow_nonfinite=False,
            )


@pytest.mark.parametrize("ttnn_dtype, torch_dtype", [(ttnn.bfloat16, torch.bfloat16), (ttnn.float32, torch.float32)])
def test_reciprocal_edge_cases(device, ttnn_dtype, torch_dtype):
    torch_input = torch.ones((32, 32), dtype=torch_dtype)
    specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        float("inf"),
        float("-inf"),
        float("nan"),
        -2.0,
        2.0,
        1e-20,
        -1e-20,
        1e20,
        -1e20,
    ]
    for i, value in enumerate(specials):
        torch_input.view(-1)[i] = value

    input_tensor = ttnn.from_torch(torch_input, dtype=ttnn_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    output_tensor = ttnn.reciprocal(input_tensor)
    golden = ttnn.get_golden_function(ttnn.reciprocal)(torch_input)
    actual = ttnn.to_torch(output_tensor, dtype=torch_dtype)

    for i, value in enumerate(specials):
        g = golden.view(-1)[i].item()
        a = actual.view(-1)[i].item()
        if math.isnan(g):
            assert math.isnan(a), f"reciprocal({value}): expected NaN, got {a}"
        elif math.isinf(g):
            assert math.isinf(a) and math.copysign(1.0, a) == math.copysign(1.0, g), (
                f"reciprocal({value}): expected {g}, got {a}"
            )
        elif math.isfinite(g) and math.isfinite(a):
            assert_with_ulp(
                expected_result=golden.view(-1)[i : i + 1],
                actual_result=actual.view(-1)[i : i + 1],
                ulp_threshold=2,
                allow_nonfinite=False,
            )


@pytest.mark.parametrize("use_welford", [False, True])
def test_layernorm_non_legacy_rsqrt(device, use_welford):
    """LayerNorm after legacy_rsqrt removal: default program config uses non-legacy rsqrt."""
    torch.manual_seed(0)
    h, w = 64, 128
    torch_input = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    torch_bias = torch.rand((w,), dtype=torch.bfloat16)
    golden = torch.nn.functional.layer_norm(
        torch_input, normalized_shape=[w], weight=torch_weight, bias=torch_bias
    )

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, layout=ttnn.TILE_LAYOUT, device=device)
    program_config = ttnn.LayerNormDefaultProgramConfig(legacy_reduction=False, use_welford=use_welford)
    output = ttnn.layer_norm(
        input_tensor,
        weight=weight,
        bias=bias,
        program_config=program_config,
    )
    actual = ttnn.to_torch(output)
    assert torch.allclose(golden, actual, rtol=1e-2, atol=1e-2)


def test_rmsnorm_non_legacy_rsqrt(device):
    torch.manual_seed(0)
    h, w = 64, 128
    torch_input = torch.rand((h, w), dtype=torch.bfloat16)
    torch_weight = torch.rand((w,), dtype=torch.bfloat16)
    variance = torch_input.pow(2).mean(-1, keepdim=True)
    golden = torch_input * torch.rsqrt(variance + 1e-5) * torch_weight

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.rms_norm(input_tensor, weight=weight, epsilon=1e-5)
    actual = ttnn.to_torch(output)
    assert torch.allclose(golden, actual, rtol=1e-2, atol=1e-2)
