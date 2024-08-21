# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from math import pi
from tests.ttnn.utils_for_testing import assert_with_pcc


def run_lerp_test_float(device, h, w, low, high, end, weight, ttnn_function, torch_function, pcc=0.9999):
    torch_input_tensor_a = torch.linspace(low, high, steps=h * w, dtype=torch.bfloat16).reshape((h, w))
    torch_input_tensor_b = torch.full((h, w), end, dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, torch_input_tensor_b, weight)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b, weight)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.5])
def test_lerp_float_a(device, h, w, weight):
    run_lerp_test_float(device, h, w, 0, 90, 100, weight, ttnn.lerp, torch.lerp)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.75])
def test_lerp_float_b(device, h, w, weight):
    run_lerp_test_float(device, h, w, 1, 80, 99, weight, ttnn.lerp, torch.lerp, pcc=0.999)


def run_lerp_test_tensor(device, h, w, low, high, end, weight, ttnn_function, torch_function, pcc=0.9999):
    torch_input_tensor_a = torch.linspace(low, high, steps=h * w, dtype=torch.bfloat16).reshape((h, w))
    torch_input_tensor_b = torch.full((h, w), end, dtype=torch.bfloat16)
    torch_weight = torch.full((h, w), weight, dtype=torch.bfloat16)

    torch_output_tensor = torch_function(torch_input_tensor_a, torch_input_tensor_b, torch_weight)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_weight = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b, input_weight)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.5])
def test_lerp_tensor_a(device, h, w, weight):
    run_lerp_test_tensor(device, h, w, 0, 90, 100, weight, ttnn.lerp, torch.lerp)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
@pytest.mark.parametrize("weight", [0.75])
def test_lerp_tensor_b(device, h, w, weight):
    run_lerp_test_tensor(device, h, w, 1, 80, 99, weight, ttnn.lerp, torch.lerp, pcc=0.999)
