# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from math import pi
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


def run_elt_binary_test_range(device, h, w, ttnn_function, low, high, pcc=0.9999):
    torch.manual_seed(0)
    low = low
    high = high
    torch_input_tensor_a = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch.manual_seed(42)
    torch_input_tensor_b = torch_random((h, w), low, high, dtype=torch.bfloat16)
    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_ldexp(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.ldexp, -60, 60, pcc=0.9995)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logaddexp(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logaddexp, -80, 80)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logaddexp2(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logaddexp2, -60, 100, pcc=0.993)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_and(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logical_and, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_or(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logical_or, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_logical_xor(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.logical_xor, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_xlogy(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.xlogy, 1e-6, 1e6)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_bias_gelu(device, h, w):
    run_elt_binary_test_range(device, h, w, ttnn.bias_gelu, -100, 100)


def run_elt_binary_test_min_max(device, h, w, ttnn_function, low, high, pcc=0.9999):
    torch.manual_seed(0)
    low = low
    high = high
    torch_input_tensor_a = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch.manual_seed(42)
    torch_input_tensor_b = torch_random((h, w), low, high, dtype=torch.bfloat16)

    golden_fn = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_fn(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_maximum(device, h, w):
    run_elt_binary_test_min_max(device, h, w, ttnn.maximum, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_minimum(device, h, w):
    run_elt_binary_test_min_max(device, h, w, ttnn.minimum, -100, 100)
