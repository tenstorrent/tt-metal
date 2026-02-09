# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import csv
import os

import pytest

import torch

import ttnn

from math import pi
from models.common.utility_functions import comp_ulp
from tests.ttnn.utils_for_testing import assert_with_ulp


def run_lerp_test_scalar_weight(device, h, w, low, high, end, weight, ttnn_function, torch_function, ulp_threshold=1):
    torch_input_tensor_a = torch.linspace(low, high, steps=h * w, dtype=torch.bfloat16).reshape((h, w))
    torch_input_tensor_b = torch.full((h, w), end, dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, torch_input_tensor_b, weight)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b, weight)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=ulp_threshold)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.5])
def test_lerp_float_a(device, h, w, weight):
    run_lerp_test_scalar_weight(device, h, w, 0, 90, 100, weight, ttnn.lerp, torch.lerp)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.75])
def test_lerp_float_b(device, h, w, weight):
    run_lerp_test_scalar_weight(device, h, w, 1, 80, 99, weight, ttnn.lerp, torch.lerp, ulp_threshold=2)


def run_lerp_test_tensor(
    device,
    h,
    w,
    low,
    high,
    end,
    weight,
    ttnn_function,
    torch_function,
    ulp_threshold=1,
    output_dtype=None,
):
    torch_input_tensor_a = torch.linspace(low, high, steps=h * w, dtype=torch.bfloat16).reshape((h, w))
    torch_input_tensor_b = torch.full((h, w), end, dtype=torch.bfloat16)
    torch_weight = torch.full((h, w), weight, dtype=torch.bfloat16)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)

    if output_dtype is None:
        # Output dtype = input dtype (default)
        torch_output_tensor = torch_function(torch_input_tensor_a, torch_input_tensor_b, torch_weight)
        output_tensor = ttnn_function(input_tensor_a, input_tensor_b, input_weight)
    else:
        # Preallocated output with given dtype; reference in same dtype
        torch_output_dtype = getattr(torch, output_dtype)
        ttnn_output_dtype = getattr(ttnn, output_dtype)

        torch_output_tensor = torch_function(
            torch_input_tensor_a.to(torch_output_dtype),
            torch_input_tensor_b.to(torch_output_dtype),
            torch_weight.to(torch_output_dtype),
        )
        output_tensor = ttnn.empty((h, w), dtype=ttnn_output_dtype, layout=ttnn.TILE_LAYOUT, device=device)
        ttnn_function(input_tensor_a, input_tensor_b, input_weight, output_tensor=output_tensor)

    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_ulp(torch_output_tensor, output_tensor, ulp_threshold=ulp_threshold)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.5])
def test_lerp_tensor_a(device, h, w, weight):
    run_lerp_test_tensor(device, h, w, 0, 90, 100, weight, ttnn.lerp, torch.lerp)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.75])
def test_lerp_tensor_b(device, h, w, weight):
    run_lerp_test_tensor(device, h, w, 1, 80, 99, weight, ttnn.lerp, torch.lerp, ulp_threshold=2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [9472])
@pytest.mark.parametrize("weight", [0.75])
def test_lerp_bf16_inputs_fp32_preallocated_output(device, h, w, weight):
    """Lerp with bfloat16 inputs (two tensors + scalar weight) and preallocated float32 output.
    Checks that output is correct within 2 ULP for float32."""
    run_lerp_test_tensor(
        device, h, w, 1, 80, 99, weight, ttnn.lerp, torch.lerp, ulp_threshold=1, output_dtype="float32"
    )
