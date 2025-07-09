# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn
import random
from math import pi
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random


def run_math_binary_test(device, h, w, ttnn_function, pcc=0.9999):
    torch.manual_seed(0)
    torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
    torch.manual_seed(42)
    torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hypot_a(device, h, w):
    run_math_binary_test(device, h, w, ttnn.hypot)


def run_math_binary_test_atan2(device, h, w, a1, a2, b1, b2, ttnn_function, pcc=0.9999):
    torch_input_tensor_a = torch.linspace(a1, a2, steps=h * w, dtype=torch.bfloat16).reshape((h, w))
    torch_input_tensor_b = torch.linspace(b1, b2, steps=h * w, dtype=torch.bfloat16).reshape((h, w))

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atan2_a(device, h, w):
    run_math_binary_test_atan2(device, h, w, 20, 40, 40, 80, ttnn.atan2)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_atan2_b(device, h, w):
    run_math_binary_test_atan2(device, h, w, -1, 1, -1, 1, ttnn.atan2)


def run_math_binary_test_range(device, h, w, ttnn_function, low, high, pcc=0.9999):
    torch.manual_seed(0)
    torch_input_tensor_a = torch_random((h, w), low, high, dtype=torch.bfloat16)
    torch.manual_seed(42)
    torch_input_tensor_b = torch_random((h, w), low, high, dtype=torch.bfloat16)

    golden_function = ttnn.get_golden_function(ttnn_function)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b)

    input_tensor_a = ttnn.from_torch(torch_input_tensor_a, layout=ttnn.TILE_LAYOUT, device=device)
    input_tensor_b = ttnn.from_torch(torch_input_tensor_b, layout=ttnn.TILE_LAYOUT, device=device)

    output_tensor = ttnn_function(input_tensor_a, input_tensor_b)
    output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output_tensor, output_tensor, pcc)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_squared_difference(device, h, w):
    run_math_binary_test_range(device, h, w, ttnn.squared_difference, -100, 100)


@pytest.mark.parametrize("h", [64])
@pytest.mark.parametrize("w", [128])
def test_hypot(device, h, w):
    run_math_binary_test_range(device, h, w, ttnn.hypot, 0, 100)


@pytest.mark.parametrize(
    "input_shapes",
    [
        (torch.Size([100])),
        (torch.Size([32, 32])),
        (torch.Size([3, 128, 32])),
        (torch.Size([1, 3, 320, 384])),
        (torch.Size([1, 1, 32, 320, 12])),
    ],
)
@pytest.mark.parametrize(
    "torch_dtype, ttnn_dtype",
    [
        (torch.float32, ttnn.float32),
        (torch.bfloat16, ttnn.bfloat16),
    ],
)
@pytest.mark.parametrize(
    "ttnn_op",
    [
        ttnn.addalpha,
        ttnn.subalpha,
    ],
)
def test_addalpha_subalpha(input_shapes, torch_dtype, ttnn_dtype, ttnn_op, device):
    torch_input_tensor_a = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-100, 100)
    torch_input_tensor_b = torch.empty(input_shapes, dtype=torch_dtype).uniform_(-150, 150)
    alpha = random.uniform(-100.0, 100.0)

    golden_function = ttnn.get_golden_function(ttnn_op)
    torch_output_tensor = golden_function(torch_input_tensor_a, torch_input_tensor_b, alpha, device=device)

    input_tensor_a = ttnn.from_torch(
        torch_input_tensor_a,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_tensor_b = ttnn.from_torch(
        torch_input_tensor_b,
        dtype=ttnn_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    output_tensor = ttnn_op(input_tensor_a, input_tensor_b, alpha)
    assert_with_pcc(ttnn.to_torch(output_tensor), torch_output_tensor)
