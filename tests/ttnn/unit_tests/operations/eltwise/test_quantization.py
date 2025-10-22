# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def convert_scalar_to_ttnn_tensor(device, scalar, dim, out_dtype):
    if dim == 0:
        return scalar
    scalar_tensor = torch.tensor([scalar])
    for i in range(1, dim):
        scalar_tensor = torch.unsqueeze(scalar_tensor, 0)
    return ttnn.from_torch(scalar_tensor, dtype=out_dtype, layout=ttnn.TILE_LAYOUT, device=device)


# Quantize to the [q_min, q_max] range (scale-then-shift)
#   Quant:   q = t / s + z
#   Dequant: t = (q - z) * s
def calculate_scale_zero_point_per_tensor(torch_input_tensor, q_min, q_max):
    i_min = torch.min(torch_input_tensor).item()
    i_max = torch.max(torch_input_tensor).item()

    scale = (i_max - i_min) / (q_max - q_min)
    zero_point = q_min - int(i_min / scale)

    return (scale, zero_point)


# PCC can't catch the case that the output tensor is all zeros (or any other constant)
# Torch.allclose is sensitive to outliers (e.g. quant-dequant of 3e-5 when most other values are around 1e-1)
# Instead, we assert that over 98% of the elements in the tensors are close enough
def check_match_ratio(golden, other, check_dtype):
    min_match_ratio = 0.98

    if check_dtype == ttnn.float32:
        golden = golden.to(torch.float32)
    elif check_dtype == ttnn.bfloat16:
        golden = golden.to(torch.bfloat16)
    else:
        golden = golden.int_repr()

    deduced_atol = 0.02 * torch.max(torch.abs(golden)).item()
    ratio = torch.count_nonzero(torch.isclose(golden, other, rtol=0.02, atol=deduced_atol)) / torch.numel(golden)
    assert ratio > min_match_ratio


# TODO: remove this once the accuracy issue of the composite op fallback is fixed (per-tensor & per-channel)
def check_pcc(golden, other, relax_for_composite):
    if relax_for_composite:
        assert_with_pcc(golden, other, 0.9998)
    else:
        assert_with_pcc(golden, other)


@pytest.mark.parametrize("x0", [32, 64])
@pytest.mark.parametrize("x1", [32, 64])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quant_dequant_per_tensor_2d(device, x0, x1, input_dtype):
    """Basic per-tensor quantize/dequantize test for 2D tensors"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    quantized_tr = torch.quantize_per_tensor(input_tr, scale, zero_point, dtype=torch.qint32)
    dequantized_tr = torch.dequantize(quantized_tr)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point)
    result_tr = ttnn.to_torch(quantized_tt)
    check_pcc(quantized_tr.int_repr(), result_tr, False)
    check_match_ratio(quantized_tr, result_tr, ttnn.int32)

    dequantized_tt = ttnn.dequantize(quantized_tt, scale, zero_point, dtype=input_dtype)
    result_tr = ttnn.to_torch(dequantized_tt)
    check_pcc(input_tr, result_tr, False)
    check_pcc(dequantized_tr, result_tr, False)
    check_match_ratio(input_tr, result_tr, input_dtype)
    check_match_ratio(dequantized_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [32])
@pytest.mark.parametrize("x1", [32])
@pytest.mark.parametrize("x2", [32])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quant_dequant_per_tensor_3d(device, x0, x1, x2, input_dtype):
    """Basic per-tensor quantize/dequantize test for 3D tensors"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, x2, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    quantized_tr = torch.quantize_per_tensor(input_tr, scale, zero_point, dtype=torch.qint32)
    dequantized_tr = torch.dequantize(quantized_tr)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point)
    result_tr = ttnn.to_torch(quantized_tt)
    check_pcc(quantized_tr.int_repr(), result_tr, False)
    check_match_ratio(quantized_tr, result_tr, ttnn.int32)

    dequantized_tt = ttnn.dequantize(quantized_tt, scale, zero_point, dtype=input_dtype)
    result_tr = ttnn.to_torch(dequantized_tt)
    check_pcc(input_tr, result_tr, False)
    check_pcc(dequantized_tr, result_tr, False)
    check_match_ratio(input_tr, result_tr, input_dtype)
    check_match_ratio(dequantized_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [32])
@pytest.mark.parametrize("x1", [32])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
def test_requant_per_tensor_2d(device, x0, x1, input_dtype):
    """Basic per-tensor requantize test for 2D tensors"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)
    scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, -37, 73)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point)
    requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r)
    derequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=input_dtype)
    result_tr = ttnn.to_torch(derequantized_tt)
    check_pcc(input_tr, result_tr, False)
    check_match_ratio(input_tr, result_tr, input_dtype)


def calculate_scale_zero_point_per_channel(input_tensor, axis, q_min, q_max):
    axis_size = input_tensor.shape[axis]
    i_min = [0.0] * axis_size
    i_max = [0.0] * axis_size
    # Slice the input along the axis, get min & max of the slice
    for i in range(axis_size):
        i_min[i] = torch.min(torch.select(input_tensor, axis, i)).item()
        i_max[i] = torch.max(torch.select(input_tensor, axis, i)).item()
    i_min = torch.tensor(i_min, dtype=torch.float32)
    i_max = torch.tensor(i_max, dtype=torch.float32)

    scale = torch.div(torch.sub(i_max, i_min), q_max - q_min)
    zero_point = torch.sub(q_min, torch.div(i_min, scale)).int()

    return (scale, zero_point)


@pytest.mark.parametrize("x0", [32])
@pytest.mark.parametrize("x1", [32])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("axis", [0, 1])
def test_quantization_per_channel_2d(device, x0, x1, input_dtype, axis):
    """Basic per-channel quantization test for 2D tensors"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    scale, zero_point = calculate_scale_zero_point_per_channel(input_tr, axis, -128, 127)
    scale_r, zero_point_r = calculate_scale_zero_point_per_channel(input_tr, axis, -37, 73)

    axis_normalized = (axis + len(input_tr.shape)) % len(input_tr.shape)
    quantized_tr = torch.quantize_per_channel(input_tr, scale, zero_point, axis=axis_normalized, dtype=torch.qint32)
    dequantized_tr = torch.dequantize(quantized_tr)

    scale_tt = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    zero_point_tt = ttnn.from_torch(zero_point, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    scale_r_tt = ttnn.from_torch(scale_r, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    zero_point_r_tt = ttnn.from_torch(zero_point_r, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    quantized_tt = ttnn.quantize(input_tt, scale_tt, zero_point_tt, axis=axis)
    result_tr = ttnn.to_torch(quantized_tt)
    check_pcc(quantized_tr.int_repr(), result_tr, False)
    check_match_ratio(quantized_tr, result_tr, ttnn.int32)

    dequantized_tt = ttnn.dequantize(quantized_tt, scale_tt, zero_point_tt, axis=axis, dtype=input_dtype)
    result_tr = ttnn.to_torch(dequantized_tt)
    check_pcc(input_tr, result_tr, False)
    check_pcc(dequantized_tr, result_tr, False)
    check_match_ratio(input_tr, result_tr, input_dtype)
    check_match_ratio(dequantized_tr, result_tr, input_dtype)

    requantized_tt = ttnn.requantize(quantized_tt, scale_tt, zero_point_tt, scale_r_tt, zero_point_r_tt, axis=axis)
    derequantized_tt = ttnn.dequantize(requantized_tt, scale_r_tt, zero_point_r_tt, axis=axis, dtype=input_dtype)
    result_tr = ttnn.to_torch(derequantized_tt)
    check_pcc(input_tr, result_tr, True)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quantization_per_tensor_program_cache(device, input_dtype):
    """Test program cache functionality"""
    torch.manual_seed(0)

    num_program_cache_entries_list = []

    for dim in [2, 3]:
        for i in range(3):
            # Each iteration gets completely different input tensors, quant ranges, etc.
            input_tr = torch.rand([30 + i] * dim, dtype=torch.float32)

            scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -120 + i, 121 - i)
            scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, -50 - i, 42 + i)

            input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
            quantized_tt = ttnn.quantize(input_tt, scale, zero_point)
            requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r)
            derequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=input_dtype)
            result_tr = ttnn.to_torch(derequantized_tt)

            check_pcc(input_tr, result_tr, False)
            check_match_ratio(input_tr, result_tr, input_dtype)

            num_program_cache_entries_list.append(device.num_program_cache_entries())

    assert num_program_cache_entries_list[0] > 0
    assert max(num_program_cache_entries_list) == min(num_program_cache_entries_list)
