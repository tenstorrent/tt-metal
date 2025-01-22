# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import torch_random, skip_for_grayskull


# Quantize to the [q_min, q_max] range (scale-then-shift)
#   Quant:   q = t / s + z
#   Dequant: t = (q - z) * s
def calculate_scale_zero_point(torch_input_tensor, q_min, q_max):
    i_min = torch.min(torch_input_tensor).item()
    i_max = torch.max(torch_input_tensor).item()

    scale = (i_max - i_min) / (q_max - q_min)
    zero_point = q_min - int(i_min / scale)

    return (scale, zero_point)


@pytest.mark.parametrize("n", [16, 31, 63, 128, 65536])
def test_quantize_1d(device, n):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(n, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(quantized_tensor)

    assert_with_pcc(torch_quantized_tensor.int_repr(), output_tensor)


@pytest.mark.parametrize("n", [16, 31, 63, 128, 65536])
def test_dequantize_1d(device, n):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(n, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)
    torch_dequantized_tensor = torch.dequantize(torch_quantized_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    dequantized_tensor = ttnn.dequantize(quantized_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert_with_pcc(torch_dequantized_tensor, output_tensor)


@pytest.mark.parametrize("n", [16, 41, 37, 128, 65536])
def test_requantize_1d(device, n):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(n, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)
    scale_new, zero_point_new = calculate_scale_zero_point(torch_input_tensor, -37, 73)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    requantized_tensor = ttnn.requantize(quantized_tensor, scale, zero_point, scale_new, zero_point_new)
    dequantized_tensor = ttnn.dequantize(requantized_tensor, scale_new, zero_point_new)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)


@pytest.mark.parametrize("h", [16, 41, 37, 128])
@pytest.mark.parametrize("w", [16, 31, 63, 128])
def test_quantize_2d(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(h, w, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(quantized_tensor)

    assert_with_pcc(torch_quantized_tensor.int_repr(), output_tensor)


@pytest.mark.parametrize("h", [16, 41, 37, 128])
@pytest.mark.parametrize("w", [16, 31, 63, 128])
def test_dequantize_2d(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(h, w, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)
    torch_dequantized_tensor = torch.dequantize(torch_quantized_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    dequantized_tensor = ttnn.dequantize(quantized_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert_with_pcc(torch_dequantized_tensor, output_tensor)


@pytest.mark.parametrize("h", [16, 41, 37, 128])
@pytest.mark.parametrize("w", [16, 31, 63, 128])
def test_requantize_2d(device, h, w):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(h, w, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)
    scale_new, zero_point_new = calculate_scale_zero_point(torch_input_tensor, -37, 73)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    requantized_tensor = ttnn.requantize(quantized_tensor, scale, zero_point, scale_new, zero_point_new)
    dequantized_tensor = ttnn.dequantize(requantized_tensor, scale_new, zero_point_new)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
def test_quantize_4d(device, x0, x1, x2, x3):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(quantized_tensor)

    assert_with_pcc(torch_quantized_tensor.int_repr(), output_tensor)


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
def test_dequantize_4d(device, x0, x1, x2, x3):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)

    torch_quantized_tensor = torch.quantize_per_tensor(torch_input_tensor, scale, zero_point, dtype=torch.qint32)
    torch_dequantized_tensor = torch.dequantize(torch_quantized_tensor)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    dequantized_tensor = ttnn.dequantize(quantized_tensor, scale, zero_point)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
    assert_with_pcc(torch_dequantized_tensor, output_tensor)


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
def test_requantize_4d(device, x0, x1, x2, x3):
    torch.manual_seed(0)

    torch_input_tensor = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point(torch_input_tensor, -128, 127)
    scale_new, zero_point_new = calculate_scale_zero_point(torch_input_tensor, -37, 73)

    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tensor = ttnn.quantize(input_tensor, scale, zero_point)
    requantized_tensor = ttnn.requantize(quantized_tensor, scale, zero_point, scale_new, zero_point_new)
    dequantized_tensor = ttnn.dequantize(requantized_tensor, scale_new, zero_point_new)
    output_tensor = ttnn.to_torch(dequantized_tensor)

    assert_with_pcc(torch_input_tensor, output_tensor)
