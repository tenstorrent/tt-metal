# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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
    assert_with_ulp(golden_output_tensor, calculated_tensor, ulp_threshold=ulp_threshold)


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
