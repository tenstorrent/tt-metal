# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.nightly.unit_tests.operations.eltwise.test_quantization2 import (
    convert_scalar_to_ttnn_tensor,
    calculate_scale_zero_point_per_tensor,
    calculate_scale_zero_point_per_channel,
    check_match_ratio,
    check_pcc,
)


@pytest.mark.parametrize("x0", [16, 31, 63, 128, 65536])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("scale_dim", [0, 1])
@pytest.mark.parametrize("zero_point_dim", [0, 1])
def test_quant_dequant_per_tensor_1d(device, x0, input_dtype, scale_dim, zero_point_dim):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    quantized_tr = torch.quantize_per_tensor(input_tr, scale, zero_point, dtype=torch.qint32)
    dequantized_tr = torch.dequantize(quantized_tr)

    scale = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point = convert_scalar_to_ttnn_tensor(device, zero_point, zero_point_dim, ttnn.int32)

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


# Per-tensor requant tests has a lot more parameter combinations, extract it to avoid repetitive quant & dequant
@pytest.mark.parametrize("x0", [16, 31, 63, 128, 65536])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("scale_dim", [0, 1])
@pytest.mark.parametrize("zero_point_dim", [0, 1])
@pytest.mark.parametrize("scale_r_dim", [0, 1])
@pytest.mark.parametrize("zero_point_r_dim", [0, 1])
def test_requant_per_tensor_1d(device, x0, input_dtype, scale_dim, zero_point_dim, scale_r_dim, zero_point_r_dim):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)
    scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, -37, 73)

    scale = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point = convert_scalar_to_ttnn_tensor(device, zero_point, zero_point_dim, ttnn.int32)
    scale_r = convert_scalar_to_ttnn_tensor(device, scale_r, scale_r_dim, ttnn.float32)
    zero_point_r = convert_scalar_to_ttnn_tensor(device, zero_point_r, zero_point_r_dim, ttnn.int32)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point)
    requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r)
    derequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=input_dtype)
    result_tr = ttnn.to_torch(derequantized_tt)
    relax_pcc = max(scale_dim, zero_point_dim, scale_r_dim, zero_point_r_dim) > 0
    check_pcc(input_tr, result_tr, relax_pcc)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [16, 41, 37, 128])
@pytest.mark.parametrize("x1", [16, 31, 63, 128])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("scale_dim", [0, 1, 2])
@pytest.mark.parametrize("zero_point_dim", [0, 1, 2])
def test_quant_dequant_per_tensor_2d(device, x0, x1, input_dtype, scale_dim, zero_point_dim):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    quantized_tr = torch.quantize_per_tensor(input_tr, scale, zero_point, dtype=torch.qint32)
    dequantized_tr = torch.dequantize(quantized_tr)

    scale = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point = convert_scalar_to_ttnn_tensor(device, zero_point, zero_point_dim, ttnn.int32)

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


@pytest.mark.parametrize("x0", [16, 41, 37, 128])
@pytest.mark.parametrize("x1", [16, 31, 63, 128])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("scale_dim", [0, 1, 2])
@pytest.mark.parametrize("zero_point_dim", [0, 1, 2])
@pytest.mark.parametrize("scale_r_dim", [0, 1, 2])
@pytest.mark.parametrize("zero_point_r_dim", [0, 1, 2])
def test_requant_per_tensor_2d(device, x0, x1, input_dtype, scale_dim, zero_point_dim, scale_r_dim, zero_point_r_dim):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)
    scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, -37, 73)

    scale = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point = convert_scalar_to_ttnn_tensor(device, zero_point, zero_point_dim, ttnn.int32)
    scale_r = convert_scalar_to_ttnn_tensor(device, scale_r, scale_r_dim, ttnn.float32)
    zero_point_r = convert_scalar_to_ttnn_tensor(device, zero_point_r, zero_point_r_dim, ttnn.int32)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point)
    requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r)
    derequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=input_dtype)
    result_tr = ttnn.to_torch(derequantized_tt)
    relax_pcc = max(scale_dim, zero_point_dim, scale_r_dim, zero_point_r_dim) > 0
    check_pcc(input_tr, result_tr, relax_pcc)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [5, 131])
@pytest.mark.parametrize("x1", [7, 127])
@pytest.mark.parametrize("x2", [11, 113])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("scale_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("zero_point_dim", [0, 1, 2, 3])
def test_quant_dequant_per_tensor_3d(device, x0, x1, x2, input_dtype, scale_dim, zero_point_dim):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, x2, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    quantized_tr = torch.quantize_per_tensor(input_tr, scale, zero_point, dtype=torch.qint32)
    dequantized_tr = torch.dequantize(quantized_tr)

    scale = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point = convert_scalar_to_ttnn_tensor(device, zero_point, zero_point_dim, ttnn.int32)

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


@pytest.mark.parametrize("x0", [5, 131])
@pytest.mark.parametrize("x1", [7, 127])
@pytest.mark.parametrize("x2", [11, 113])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("scale_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("zero_point_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("scale_r_dim", [0, 1, 2, 3])
@pytest.mark.parametrize("zero_point_r_dim", [0, 1, 2, 3])
def test_requant_per_tensor_3d(
    device, x0, x1, x2, input_dtype, scale_dim, zero_point_dim, scale_r_dim, zero_point_r_dim
):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, x2, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)
    scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, -37, 73)

    scale = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point = convert_scalar_to_ttnn_tensor(device, zero_point, zero_point_dim, ttnn.int32)
    scale_r = convert_scalar_to_ttnn_tensor(device, scale_r, scale_r_dim, ttnn.float32)
    zero_point_r = convert_scalar_to_ttnn_tensor(device, zero_point_r, zero_point_r_dim, ttnn.int32)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point)
    requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r)
    derequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=input_dtype)
    result_tr = ttnn.to_torch(derequantized_tt)
    relax_pcc = max(scale_dim, zero_point_dim, scale_r_dim, zero_point_r_dim) > 0
    check_pcc(input_tr, result_tr, relax_pcc)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("scale_dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("zero_point_dim", [0, 1, 2, 3, 4])
def test_quant_dequant_per_tensor_4d(device, x0, x1, x2, x3, input_dtype, scale_dim, zero_point_dim):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    quantized_tr = torch.quantize_per_tensor(input_tr, scale, zero_point, dtype=torch.qint32)
    dequantized_tr = torch.dequantize(quantized_tr)

    scale = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point = convert_scalar_to_ttnn_tensor(device, zero_point, zero_point_dim, ttnn.int32)

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


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("scale_dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("zero_point_dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("scale_r_dim", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("zero_point_r_dim", [0, 1, 2, 3, 4])
def test_requant_per_tensor_4d(
    device, x0, x1, x2, x3, input_dtype, scale_dim, zero_point_dim, scale_r_dim, zero_point_r_dim
):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)
    scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, -37, 73)

    scale = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point = convert_scalar_to_ttnn_tensor(device, zero_point, zero_point_dim, ttnn.int32)
    scale_r = convert_scalar_to_ttnn_tensor(device, scale_r, scale_r_dim, ttnn.float32)
    zero_point_r = convert_scalar_to_ttnn_tensor(device, zero_point_r, zero_point_r_dim, ttnn.int32)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point)
    requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r)
    derequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=input_dtype)
    result_tr = ttnn.to_torch(derequantized_tt)
    relax_pcc = max(scale_dim, zero_point_dim, scale_r_dim, zero_point_r_dim) > 0
    check_pcc(input_tr, result_tr, relax_pcc)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [16, 31, 63, 128, 65536])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quantization_per_channel_1d(device, x0, input_dtype):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(input_tr.shape)
    for axis in range(-rank, rank):
        # Each "channel" in a 1D tensor is just a single value, so we can't compute scale & zero-point
        # Calculate scale & zero-point based on the whole input, and apply them to each channel
        scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)
        scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, -37, 73)
        scale = torch.tensor([scale] * x0)
        zero_point = torch.tensor([zero_point] * x0).int()
        scale_r = torch.tensor([scale_r] * x0)
        zero_point_r = torch.tensor([zero_point_r] * x0).int()

        axis_normalized = (axis + rank) % rank
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


@pytest.mark.parametrize("x0", [16, 41, 37, 128])
@pytest.mark.parametrize("x1", [16, 31, 63, 128])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quantization_per_channel_2d(device, x0, x1, input_dtype):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(input_tr.shape)
    for axis in range(-rank, rank):
        scale, zero_point = calculate_scale_zero_point_per_channel(input_tr, axis, -128, 127)
        scale_r, zero_point_r = calculate_scale_zero_point_per_channel(input_tr, axis, -37, 73)

        axis_normalized = (axis + rank) % rank
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


@pytest.mark.parametrize("x0", [5, 131])
@pytest.mark.parametrize("x1", [7, 127])
@pytest.mark.parametrize("x2", [11, 113])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quantization_per_channel_3d(device, x0, x1, x2, input_dtype):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, x2, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(input_tr.shape)
    for axis in range(-rank, rank):
        scale, zero_point = calculate_scale_zero_point_per_channel(input_tr, axis, -128, 127)
        scale_r, zero_point_r = calculate_scale_zero_point_per_channel(input_tr, axis, -37, 73)

        axis_normalized = (axis + rank) % rank
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


@pytest.mark.parametrize("x0", [128])
@pytest.mark.parametrize("x1", [17])
@pytest.mark.parametrize("x2", [3])
@pytest.mark.parametrize("x3", [64])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quantization_per_channel_4d(device, x0, x1, x2, x3, input_dtype):
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, x2, x3, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(input_tr.shape)
    for axis in range(-rank, rank):
        scale, zero_point = calculate_scale_zero_point_per_channel(input_tr, axis, -128, 127)
        scale_r, zero_point_r = calculate_scale_zero_point_per_channel(input_tr, axis, -37, 73)

        axis_normalized = (axis + rank) % rank
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


# TODO:
# Add tests for tensor scales/zero-points once the composite op fallbacks stop creating different
# kernels for the same op when input dimensions and sizes change
# Add tests for per-channel once changing the quantizaiton axis no longer affects the number of
# kernels used
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quantization_per_tensor_program_cache(device, input_dtype):
    torch.manual_seed(0)

    num_program_cache_entries_list = []

    for dim in [1, 2, 3, 4]:
        for i in range(5):
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


@pytest.mark.parametrize("x0", [32])
@pytest.mark.parametrize("x1", [32])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("axis", [0])
def test_requant_per_tensor_to_per_channel_2d(device, x0, x1, input_dtype, axis):
    """Test requantization (per-tensor -> per-channel)"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    in_scale, in_zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    rank = len(input_tr.shape)
    axis_normalized = (axis + rank) % rank
    out_scale, out_zero_point = calculate_scale_zero_point_per_channel(input_tr, axis_normalized, -64, 63)

    out_scale_tt = ttnn.from_torch(out_scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out_zero_point_tt = ttnn.from_torch(out_zero_point, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    quantized_tt = ttnn.quantize(input_tt, in_scale, in_zero_point)
    requantized_tt = ttnn.requantize(quantized_tt, in_scale, in_zero_point, out_scale_tt, out_zero_point_tt, axis=axis)
    derequantized_tt = ttnn.dequantize(requantized_tt, out_scale_tt, out_zero_point_tt, axis=axis, dtype=input_dtype)

    result_tr = ttnn.to_torch(derequantized_tt)
    check_pcc(input_tr, result_tr, True)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [32])
@pytest.mark.parametrize("x1", [32])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("axis", [0])
def test_requant_per_channel_to_per_tensor_2d(device, x0, x1, input_dtype, axis):
    """Test requantization (per-channel -> per-tensor)"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(input_tr.shape)
    axis_normalized = (axis + rank) % rank
    in_scale, in_zero_point = calculate_scale_zero_point_per_channel(input_tr, axis_normalized, -128, 127)

    out_scale, out_zero_point = calculate_scale_zero_point_per_tensor(input_tr, -64, 63)

    in_scale_tt = ttnn.from_torch(in_scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    in_zero_point_tt = ttnn.from_torch(in_zero_point, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    quantized_tt = ttnn.quantize(input_tt, in_scale_tt, in_zero_point_tt, axis=axis)
    requantized_tt = ttnn.requantize(quantized_tt, in_scale_tt, in_zero_point_tt, out_scale, out_zero_point, axis=axis)
    # For per-tensor output (scalar tensors), don't pass axis to dequantize.
    derequantized_tt = ttnn.dequantize(requantized_tt, out_scale, out_zero_point, dtype=input_dtype)

    result_tr = ttnn.to_torch(derequantized_tt)
    check_pcc(input_tr, result_tr, True)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [32])
@pytest.mark.parametrize("x1", [32])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("axis", [0, 1])
def test_requant_all_tensors_per_tensor_to_per_channel_2d(device, x0, x1, input_dtype, axis):
    """Test requantization with all parameters as tensors (per-tensor -> per-channel)"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(input_tr.shape)
    axis_normalized = (axis + rank) % rank
    in_scale, in_zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    out_scale, out_zero_point = calculate_scale_zero_point_per_channel(input_tr, axis_normalized, -64, 63)

    # Convert all parameters to tensors.
    in_scale_tt = ttnn.from_torch(torch.tensor(in_scale), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    in_zero_point_tt = ttnn.from_torch(
        torch.tensor(in_zero_point), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )
    out_scale_tt = ttnn.from_torch(out_scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out_zero_point_tt = ttnn.from_torch(out_zero_point, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    quantized_tt = ttnn.quantize(input_tt, in_scale_tt, in_zero_point_tt)
    requantized_tt = ttnn.requantize(
        quantized_tt, in_scale_tt, in_zero_point_tt, out_scale_tt, out_zero_point_tt, axis=axis
    )
    derequantized_tt = ttnn.dequantize(requantized_tt, out_scale_tt, out_zero_point_tt, axis=axis, dtype=input_dtype)

    result_tr = ttnn.to_torch(derequantized_tt)
    check_pcc(input_tr, result_tr, True)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [32])
@pytest.mark.parametrize("x1", [32])
@pytest.mark.parametrize("input_dtype", [ttnn.float32])
@pytest.mark.parametrize("axis", [0, 1])
def test_requant_all_tensors_per_channel_to_per_tensor_2d(device, x0, x1, input_dtype, axis):
    """Test requantization with all parameters as tensors (per-channel -> per-tensor)"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(input_tr.shape)
    axis_normalized = (axis + rank) % rank
    in_scale, in_zero_point = calculate_scale_zero_point_per_channel(input_tr, axis_normalized, -128, 127)

    out_scale, out_zero_point = calculate_scale_zero_point_per_tensor(input_tr, -64, 63)

    # Convert all parameters to tensors.
    in_scale_tt = ttnn.from_torch(in_scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    in_zero_point_tt = ttnn.from_torch(in_zero_point, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    out_scale_tt = ttnn.from_torch(torch.tensor(out_scale), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out_zero_point_tt = ttnn.from_torch(
        torch.tensor(out_zero_point), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )

    quantized_tt = ttnn.quantize(input_tt, in_scale_tt, in_zero_point_tt, axis=axis)
    requantized_tt = ttnn.requantize(
        quantized_tt, in_scale_tt, in_zero_point_tt, out_scale_tt, out_zero_point_tt, axis=axis
    )
    # For per-tensor output (scalar tensors), don't pass axis to dequantize.
    derequantized_tt = ttnn.dequantize(requantized_tt, out_scale_tt, out_zero_point_tt, dtype=input_dtype)

    result_tr = ttnn.to_torch(derequantized_tt)
    check_pcc(input_tr, result_tr, True)
    check_match_ratio(input_tr, result_tr, input_dtype)


@pytest.mark.parametrize("x0", [32, 128])
@pytest.mark.parametrize("x1", [32, 128])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("q_max", [127, 255])
def test_quant_dequant_requant_uint8_per_tensor_2d(device, x0, x1, input_dtype, q_max):
    """Test quantize, dequantize and requantize (per-tensor) for uint8"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, 0, q_max)

    quantized_tr = torch.quantize_per_tensor(input_tr, scale, zero_point, dtype=torch.quint8)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point, dtype=ttnn.uint8)
    assert quantized_tt.dtype == ttnn.uint8
    result_q = ttnn.to_torch(quantized_tt)
    check_pcc(quantized_tr.int_repr(), result_q, False)
    check_match_ratio(quantized_tr, result_q, ttnn.uint8)

    dequantized_tr = torch.dequantize(quantized_tr)
    dequantized_tt = ttnn.dequantize(quantized_tt, scale, zero_point, dtype=input_dtype)
    result_dq = ttnn.to_torch(dequantized_tt)
    check_pcc(dequantized_tr, result_dq, False)
    check_match_ratio(dequantized_tr, result_dq, input_dtype)

    scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, 0, 200)
    requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r, dtype=ttnn.uint8)
    assert requantized_tt.dtype == ttnn.uint8
    rederequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=input_dtype)
    result_rq = ttnn.to_torch(rederequantized_tt)
    check_pcc(input_tr, result_rq, True)
    check_match_ratio(input_tr, result_rq, input_dtype)


def test_quantize_uint8_upper_saturation(device):
    """Test quantize uint8 upper saturation to 255"""
    row = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0]
    input_tr = torch.tensor([row], dtype=torch.float32)
    scale, zero_point = 1.0 / 255.0, 0
    expected = torch.clamp(torch.round(input_tr / scale + zero_point), 0, 255).to(torch.uint8)

    input_tt = ttnn.from_torch(input_tr, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.quantize(input_tt, scale, zero_point, dtype=ttnn.uint8)
    assert out_tt.dtype == ttnn.uint8
    result = ttnn.to_torch(out_tt)
    assert torch.equal(result, expected), f"got {result.tolist()} expected {expected.tolist()}"


def test_requantize_uint8_upper_saturation(device):
    """Test requantize uint8 upper saturation to 255 on the output side"""
    q_in = torch.tensor([[0, 50, 100, 200, 255, 300, 1000]], dtype=torch.int32)
    in_scale, in_zp, out_scale, out_zp = 1.0, 0, 1.0, 0
    expected = torch.clamp(torch.round((q_in - in_zp) * in_scale / out_scale + out_zp), 0, 255).to(torch.uint8)

    q_in_tt = ttnn.from_torch(q_in, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.requantize(q_in_tt, in_scale, in_zp, out_scale, out_zp, dtype=ttnn.uint8)
    assert out_tt.dtype == ttnn.uint8
    result = ttnn.to_torch(out_tt)
    assert torch.equal(result, expected), f"got {result.tolist()} expected {expected.tolist()}"


@pytest.mark.parametrize("x0", [32, 128])
@pytest.mark.parametrize("x1", [32, 128])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quant_dequant_requant_int8_per_tensor_2d(device, x0, x1, input_dtype):
    """Test quantize, dequantize and requantize (per-tensor) for int8"""
    torch.manual_seed(0)
    input_tr = torch.rand(x0, x1, dtype=torch.float32)
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)

    quantized_tr = torch.quantize_per_tensor(input_tr, scale, zero_point, dtype=torch.qint8)

    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point, dtype=ttnn.int8)
    assert quantized_tt.dtype == ttnn.int8
    result_q = ttnn.to_torch(quantized_tt)
    check_pcc(quantized_tr.int_repr(), result_q, False)
    check_match_ratio(quantized_tr, result_q, ttnn.int8)

    dequantized_tr = torch.dequantize(quantized_tr)
    dequantized_tt = ttnn.dequantize(quantized_tt, scale, zero_point, dtype=input_dtype)
    result_dq = ttnn.to_torch(dequantized_tt)
    check_pcc(dequantized_tr, result_dq, False)
    check_match_ratio(dequantized_tr, result_dq, input_dtype)

    scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, -100, 100)
    requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r, dtype=ttnn.int8)
    assert requantized_tt.dtype == ttnn.int8
    rederequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=input_dtype)
    result_rq = ttnn.to_torch(rederequantized_tt)
    check_pcc(input_tr, result_rq, True)
    check_match_ratio(input_tr, result_rq, input_dtype)


def test_quantize_int8_saturation(device):
    """Test quantize int8 saturation at both ends, to -128 and 127"""
    input_tr = torch.tensor(
        [
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0],
            [0.0, -0.2, -0.4, -0.6, -0.8, -1.0, -2.0, -5.0],
        ],
        dtype=torch.float32,
    )
    scale, zero_point = 1.0 / 127.0, 0
    expected = torch.clamp(torch.round(input_tr / scale + zero_point), -128, 127).to(torch.int8)

    input_tt = ttnn.from_torch(input_tr, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.quantize(input_tt, scale, zero_point, dtype=ttnn.int8)
    assert out_tt.dtype == ttnn.int8
    result = ttnn.to_torch(out_tt)
    assert torch.equal(result, expected), f"got {result.tolist()} expected {expected.tolist()}"


@pytest.mark.parametrize(
    "in_dtype,q_values,in_scale",
    [
        (ttnn.int32, [0, 50, 100, 120, 127, 200, 1000, -50, -100, -120, -127, -200, -1000], 1.0),
        (ttnn.int8, [-128, -100, -32, -1, 0, 1, 32, 100, 127], 4.0),
    ],
)
def test_requantize_int8_output_saturation(device, in_dtype, q_values, in_scale):
    """Test requantize saturating both ends of an int8 output, from an int32 and an int8 input"""
    q_in = torch.tensor([q_values], dtype=torch.int32 if in_dtype == ttnn.int32 else torch.int8)
    in_zp, out_scale, out_zp = 0, 1.0, 0
    expected = torch.clamp(torch.round((q_in.to(torch.float32) - in_zp) * in_scale / out_scale + out_zp), -128, 127).to(
        torch.int8
    )

    q_in_tt = ttnn.from_torch(q_in, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.requantize(q_in_tt, in_scale, in_zp, out_scale, out_zp, dtype=ttnn.int8)
    assert out_tt.dtype == ttnn.int8
    result = ttnn.to_torch(out_tt)
    assert torch.equal(result, expected), f"got {result.tolist()} expected {expected.tolist()}"


def test_dequantize_int8_edge_cases(device):
    """Test dequantize on fixed int8 boundary values against the exact float output"""
    q_in = torch.tensor([[-128, -100, -1, 0, 1, 100, 127]], dtype=torch.int8)
    scale, zero_point = 0.5, -10
    expected = (q_in.to(torch.float32) - zero_point) * scale

    q_in_tt = ttnn.from_torch(q_in, dtype=ttnn.int8, layout=ttnn.TILE_LAYOUT, device=device)
    out_tt = ttnn.dequantize(q_in_tt, scale, zero_point, dtype=ttnn.float32)
    assert out_tt.dtype == ttnn.float32
    result = ttnn.to_torch(out_tt)
    assert torch.equal(result, expected), f"got {result.tolist()} expected {expected.tolist()}"


@pytest.mark.parametrize(
    "in_dtype,in_q_max",
    [(ttnn.int32, 127), (ttnn.int8, 127), (ttnn.uint8, 255)],
)
@pytest.mark.parametrize(
    "out_dtype,out_q_max",
    [(ttnn.int32, 127), (ttnn.int8, 127), (ttnn.uint8, 255)],
)
def test_requant_mixed_dtype_per_tensor_2d(device, in_dtype, in_q_max, out_dtype, out_q_max):
    """Test requant across int32/int8/uint8 input and output dtype combinations (per-tensor)"""
    torch.manual_seed(0)
    input_tr = torch.rand(64, 64, dtype=torch.float32)

    in_q_min = 0 if in_dtype == ttnn.uint8 else -128
    scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, in_q_min, in_q_max)

    input_tt = ttnn.from_torch(input_tr, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    quantized_tt = ttnn.quantize(input_tt, scale, zero_point, dtype=in_dtype)
    assert quantized_tt.dtype == in_dtype

    out_q_min = 0 if out_dtype == ttnn.uint8 else -100
    scale_r, zero_point_r = calculate_scale_zero_point_per_tensor(input_tr, out_q_min, out_q_max)
    requantized_tt = ttnn.requantize(quantized_tt, scale, zero_point, scale_r, zero_point_r, dtype=out_dtype)
    assert requantized_tt.dtype == out_dtype

    dequantized_tt = ttnn.dequantize(requantized_tt, scale_r, zero_point_r, dtype=ttnn.float32)
    result_tr = ttnn.to_torch(dequantized_tt)
    check_pcc(input_tr, result_tr, True)
    check_match_ratio(input_tr, result_tr, ttnn.float32)


# The fused and the composite path compute the same expression by different routes, so compare
# them elementwise as well as by PCC. PCC on its own is blind to a uniform 1-LSB shift: the
# quantize and requantize pairs below differ on roughly half of their elements and still score
# well above the 0.9999 threshold.
def check_dequant_matches_composite(composite, fused, out_dtype):
    """Dequantize: the two routes agree bit-for-bit in float32. In bfloat16 they can land on
    adjacent representable values, so allow a distance of one step. Counting steps rather than
    using a relative tolerance keeps the bound tight, since one bfloat16 step is anywhere between
    2**-8 and 2**-7 in relative terms depending on where the value sits in its binade."""
    if out_dtype != ttnn.bfloat16:
        assert torch.equal(composite, fused)
        return
    # Map sign-magnitude to a monotonic ordinal so that adjacent values differ by 1 across zero.
    lhs = composite.view(torch.int16).to(torch.int32)
    rhs = fused.view(torch.int16).to(torch.int32)
    lhs = torch.where(lhs < 0, -32768 - lhs, lhs)
    rhs = torch.where(rhs < 0, -32768 - rhs, rhs)
    steps = (lhs - rhs).abs().max().item()
    assert steps <= 1, f"fused and composite are {steps} bfloat16 steps apart, expected at most 1"


def check_within_one_lsb(composite, fused):
    """Quantize/requantize: the composite fallback narrows with a truncating typecast where the
    fused LLK rounds to nearest even, so the two disagree by at most 1 LSB."""
    diff = (composite.to(torch.int64) - fused.to(torch.int64)).abs().max().item()
    assert diff <= 1, f"fused and composite differ by {diff} LSB, expected at most 1"


# Per-channel scale + scalar zero-point takes the fused single-pass QUANT/DEQUANT path (vs the
# slower composite). Each test checks fused (scalar zp) == composite (tensor zp) == torch golden.
def _per_channel_amax_scale(input_tr, axis):
    """Symmetric per-channel scale = amax/127 (values land in int8 range, zp=0)."""
    dims = [d for d in range(input_tr.dim()) if d != axis]
    amax = input_tr.abs().amax(dim=dims)
    return (amax / 127.0).clamp_min(1e-8).to(torch.float32)


@pytest.mark.parametrize("shape", [(64, 96), (128, 128), (3, 64, 160)])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
def test_quantize_dequantize_per_channel_symmetric(device, shape, input_dtype):
    """Symmetric (zp=0) per-channel round-trip: fused vs composite vs torch golden."""
    torch.manual_seed(0)
    input_tr = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 4.0  # centered, both signs
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(shape)
    for axis in range(-rank, rank):
        axis_n = (axis + rank) % rank
        scale_vec = _per_channel_amax_scale(input_tr, axis_n)
        zp_vec = torch.zeros(shape[axis_n], dtype=torch.int32)

        scale_tt = ttnn.from_torch(scale_vec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        zp_vec_tt = ttnn.from_torch(zp_vec, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        quantized_golden = torch.quantize_per_channel(input_tr, scale_vec, zp_vec, axis=axis_n, dtype=torch.qint32)
        dequantized_golden = torch.dequantize(quantized_golden)

        # quantize: fused (scalar zp=0) vs composite (tensor zp) vs golden
        q_fused = ttnn.quantize(input_tt, scale_tt, 0, axis=axis)  # NEW fused path
        q_comp = ttnn.quantize(input_tt, scale_tt, zp_vec_tt, axis=axis)
        q_fused_tr = ttnn.to_torch(q_fused)
        check_pcc(ttnn.to_torch(q_comp), q_fused_tr, False)
        check_within_one_lsb(ttnn.to_torch(q_comp), q_fused_tr)
        check_pcc(quantized_golden.int_repr(), q_fused_tr, False)
        check_match_ratio(quantized_golden, q_fused_tr, ttnn.int32)

        # dequantize: fused (scalar zp=0) vs composite (tensor zp) vs golden
        dq_fused = ttnn.dequantize(q_fused, scale_tt, 0, axis=axis, dtype=input_dtype)  # NEW fused path
        dq_comp = ttnn.dequantize(q_fused, scale_tt, zp_vec_tt, axis=axis, dtype=input_dtype)
        dq_fused_tr = ttnn.to_torch(dq_fused)
        check_pcc(ttnn.to_torch(dq_comp), dq_fused_tr, False)
        check_dequant_matches_composite(ttnn.to_torch(dq_comp), dq_fused_tr, input_dtype)
        check_pcc(dequantized_golden, dq_fused_tr, False)
        check_match_ratio(dequantized_golden, dq_fused_tr, input_dtype)


@pytest.mark.parametrize("shape", [(64, 96), (128, 128), (3, 64, 160)])
@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("zero_point", [0, 5, -7])
def test_dequantize_per_channel_scalar_zero_point(device, shape, out_dtype, zero_point):
    """Dequantize with an arbitrary int input, per-channel scale, scalar zp: fused vs composite vs golden."""
    torch.manual_seed(0)
    q_tr = torch.randint(-128, 127, shape, dtype=torch.int32)
    q_tt = ttnn.from_torch(q_tr, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(shape)
    for axis in range(-rank, rank):
        axis_n = (axis + rank) % rank
        axis_size = shape[axis_n]
        scale_vec = torch.rand(axis_size, dtype=torch.float32) * 0.05 + 0.005
        zp_vec = torch.full((axis_size,), zero_point, dtype=torch.int32)

        scale_tt = ttnn.from_torch(scale_vec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        zp_vec_tt = ttnn.from_torch(zp_vec, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        bshape = [1] * rank
        bshape[axis_n] = axis_size
        golden = (q_tr.to(torch.float32) - zero_point) * scale_vec.reshape(bshape)

        dq_fused = ttnn.dequantize(q_tt, scale_tt, zero_point, axis=axis, dtype=out_dtype)  # NEW fused path
        dq_comp = ttnn.dequantize(q_tt, scale_tt, zp_vec_tt, axis=axis, dtype=out_dtype)
        dq_fused_tr = ttnn.to_torch(dq_fused)
        check_pcc(ttnn.to_torch(dq_comp), dq_fused_tr, False)
        check_dequant_matches_composite(ttnn.to_torch(dq_comp), dq_fused_tr, out_dtype)
        check_pcc(golden, dq_fused_tr, False)
        check_match_ratio(golden, dq_fused_tr, out_dtype)


@pytest.mark.parametrize("shape", [(64, 96), (128, 128), (3, 64, 160)])
@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("zero_point", [0, 128, 200])
def test_dequantize_per_channel_scalar_zero_point_uint8(device, shape, out_dtype, zero_point):
    """Per-channel dequantize of a uint8 input with a scalar zp: fused vs composite vs golden."""
    torch.manual_seed(0)
    q_tr = torch.randint(0, 256, shape, dtype=torch.int32)  # uint8 value range
    q_tt = ttnn.from_torch(q_tr.to(torch.uint8), dtype=ttnn.uint8, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(shape)
    for axis in range(-rank, rank):
        axis_n = (axis + rank) % rank
        axis_size = shape[axis_n]
        scale_vec = torch.rand(axis_size, dtype=torch.float32) * 0.05 + 0.005
        zp_vec = torch.full((axis_size,), zero_point, dtype=torch.int32)

        scale_tt = ttnn.from_torch(scale_vec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        zp_vec_tt = ttnn.from_torch(zp_vec, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        bshape = [1] * rank
        bshape[axis_n] = axis_size
        golden = (q_tr.to(torch.float32) - zero_point) * scale_vec.reshape(bshape)

        dq_fused = ttnn.dequantize(q_tt, scale_tt, zero_point, axis=axis, dtype=out_dtype)  # NEW fused path
        dq_comp = ttnn.dequantize(q_tt, scale_tt, zp_vec_tt, axis=axis, dtype=out_dtype)
        dq_fused_tr = ttnn.to_torch(dq_fused)
        check_pcc(ttnn.to_torch(dq_comp), dq_fused_tr, False)
        check_dequant_matches_composite(ttnn.to_torch(dq_comp), dq_fused_tr, out_dtype)
        check_pcc(golden, dq_fused_tr, False)
        check_match_ratio(golden, dq_fused_tr, out_dtype)


@pytest.mark.parametrize("shape", [(64, 96), (128, 128), (3, 64, 160)])
@pytest.mark.parametrize("out_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("zero_point", [0, -128, 127])
def test_dequantize_per_channel_scalar_zero_point_int8(device, shape, out_dtype, zero_point):
    """Per-channel dequantize of an int8 input with a scalar zp: fused vs composite vs golden."""
    torch.manual_seed(0)
    q_tr = torch.randint(-128, 128, shape, dtype=torch.int32)
    q_tt = ttnn.from_torch(q_tr.to(torch.int8), dtype=ttnn.int8, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(shape)
    for axis in range(-rank, rank):
        axis_n = (axis + rank) % rank
        axis_size = shape[axis_n]
        scale_vec = torch.rand(axis_size, dtype=torch.float32) * 0.05 + 0.005
        zp_vec = torch.full((axis_size,), zero_point, dtype=torch.int32)

        scale_tt = ttnn.from_torch(scale_vec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        zp_vec_tt = ttnn.from_torch(zp_vec, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        bshape = [1] * rank
        bshape[axis_n] = axis_size
        golden = (q_tr.to(torch.float32) - zero_point) * scale_vec.reshape(bshape)

        dq_fused = ttnn.dequantize(q_tt, scale_tt, zero_point, axis=axis, dtype=out_dtype)
        dq_comp = ttnn.dequantize(q_tt, scale_tt, zp_vec_tt, axis=axis, dtype=out_dtype)
        dq_fused_tr = ttnn.to_torch(dq_fused)
        check_pcc(ttnn.to_torch(dq_comp), dq_fused_tr, False)
        check_dequant_matches_composite(ttnn.to_torch(dq_comp), dq_fused_tr, out_dtype)
        check_pcc(golden, dq_fused_tr, False)
        check_match_ratio(golden, dq_fused_tr, out_dtype)


@pytest.mark.parametrize("shape", [(32, 128), (64, 96)])
@pytest.mark.parametrize("scale", [0.5, 0.02])
def test_quantize_int8_tensor_zero_point_saturates(device, shape, scale):
    """quantize -> int8 through the composite must saturate, not collapse to the sign."""
    torch.manual_seed(0)
    x = torch.linspace(-300.0, 300.0, shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    xt = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    zp_t = ttnn.from_torch(
        torch.tensor([3], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device
    )

    composite = ttnn.to_torch(ttnn.quantize(xt, scale, zp_t, dtype=ttnn.int8))
    fast = ttnn.to_torch(ttnn.quantize(xt, scale, 3, dtype=ttnn.int8))
    assert torch.equal(composite, fast), "tensor-zero-point quantize diverges from the fast path"

    got = composite.to(torch.float32)
    assert got.min() >= -128 and got.max() <= 127
    # A sign-collapse regression shows up as the output taking only {-1, 0, 127}.
    assert len(torch.unique(got)) > 3, f"output collapsed to {torch.unique(got).tolist()}"


@pytest.mark.parametrize("shape", [(64, 96), (128, 128), (3, 64, 160)])
@pytest.mark.parametrize("input_dtype", [ttnn.float32, ttnn.bfloat16])
@pytest.mark.parametrize("zero_point", [0, 5, -7])
def test_quantize_per_channel_scalar_zero_point(device, shape, input_dtype, zero_point):
    """Quantize with a per-channel scale and a non-zero scalar zp: fused vs composite vs golden.

    The fused path folds the zero-point into the QUANT LLK's scalar arg, so a non-zero zp
    exercises code the symmetric (zp=0) test above cannot reach. Scales are picked so that
    q = x/s + zp stays inside the int8 range, where the fused and composite paths agree (see
    test_quantize_per_channel_scalar_zero_point_saturation for what happens outside it).
    """
    torch.manual_seed(0)
    input_tr = (torch.rand(*shape, dtype=torch.float32) - 0.5) * 4.0
    input_tt = ttnn.from_torch(input_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(shape)
    for axis in range(-rank, rank):
        axis_n = (axis + rank) % rank
        # Leave |zero_point| of headroom so x/s + zp cannot leave [-127, 127].
        dims = [d for d in range(rank) if d != axis_n]
        amax = input_tr.abs().amax(dim=dims)
        scale_vec = (amax / (127.0 - abs(zero_point))).clamp_min(1e-8).to(torch.float32)
        zp_vec = torch.full((shape[axis_n],), zero_point, dtype=torch.int32)

        scale_tt = ttnn.from_torch(scale_vec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        zp_vec_tt = ttnn.from_torch(zp_vec, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        golden = torch.quantize_per_channel(input_tr, scale_vec, zp_vec, axis=axis_n, dtype=torch.qint32)

        q_fused = ttnn.quantize(input_tt, scale_tt, zero_point, axis=axis)  # fused path
        q_comp = ttnn.quantize(input_tt, scale_tt, zp_vec_tt, axis=axis)  # composite path
        q_fused_tr = ttnn.to_torch(q_fused)
        check_pcc(ttnn.to_torch(q_comp), q_fused_tr, False)
        check_within_one_lsb(ttnn.to_torch(q_comp), q_fused_tr)
        check_pcc(golden.int_repr(), q_fused_tr, False)
        check_match_ratio(golden, q_fused_tr, ttnn.int32)

        # Round-trip back through the fused dequantize to confirm the zp is applied in the
        # right direction (a sign flip would still pass the checks above on symmetric data).
        dq_tr = ttnn.to_torch(ttnn.dequantize(q_fused, scale_tt, zero_point, axis=axis, dtype=input_dtype))
        check_pcc(input_tr, dq_tr, False)
        check_match_ratio(input_tr, dq_tr, input_dtype)


@pytest.mark.parametrize("shape", [(64, 96), (128, 128), (3, 64, 160)])
@pytest.mark.parametrize("in_zero_point", [0, 5])
@pytest.mark.parametrize("out_zero_point", [0, -7])
@pytest.mark.parametrize("input_dtype", [ttnn.int32, ttnn.int8])
def test_requantize_per_channel_scalar_zero_point(device, shape, in_zero_point, out_zero_point, input_dtype):
    """Requantize with per-channel scales and scalar zero-points.

    This combination is not a tensor-only per-channel requantize, so it decomposes into
    dequantize + quantize - both of which now take the fused per-channel path. This test pins
    that the indirect speedup does not cost accuracy, and compares against the all-tensor
    per-channel requantize path that stays composite.
    """
    torch.manual_seed(0)
    # Bounded so that (q - z_in) * s_in / s_out + z_out cannot leave the int8 range.
    q_tr = torch.randint(-90, 91, shape, dtype=torch.int32)
    q_tt = ttnn.from_torch(q_tr, dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)

    rank = len(shape)
    for axis in range(-rank, rank):
        axis_n = (axis + rank) % rank
        axis_size = shape[axis_n]

        in_scale_vec = torch.rand(axis_size, dtype=torch.float32) * 0.05 + 0.005
        # Keep the scale ratio near 1 so the requantized values stay representable.
        out_scale_vec = in_scale_vec * (torch.rand(axis_size, dtype=torch.float32) * 0.45 + 0.8)
        in_zp_vec = torch.full((axis_size,), in_zero_point, dtype=torch.int32)
        out_zp_vec = torch.full((axis_size,), out_zero_point, dtype=torch.int32)

        in_scale_tt = ttnn.from_torch(in_scale_vec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        out_scale_tt = ttnn.from_torch(out_scale_vec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        in_zp_tt = ttnn.from_torch(in_zp_vec, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
        out_zp_tt = ttnn.from_torch(out_zp_vec, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

        bshape = [1] * rank
        bshape[axis_n] = axis_size
        ratio = (in_scale_vec / out_scale_vec).reshape(bshape)
        golden = torch.round((q_tr.to(torch.float32) - in_zero_point) * ratio) + out_zero_point

        rq_fused = ttnn.to_torch(
            ttnn.requantize(q_tt, in_scale_tt, in_zero_point, out_scale_tt, out_zero_point, axis=axis)
        )
        rq_comp = ttnn.to_torch(ttnn.requantize(q_tt, in_scale_tt, in_zp_tt, out_scale_tt, out_zp_tt, axis=axis))
        check_pcc(golden, rq_fused, False)
        check_pcc(rq_comp, rq_fused, True)
        check_within_one_lsb(rq_comp, rq_fused)
        check_match_ratio(golden, rq_fused.to(torch.float32), ttnn.float32)

        # int8 cannot widen with a plain typecast (#50401), so both routes above go through a
        # scale-1 dequantize instead. int32 is the reference: the same values must requantize
        # identically in either input dtype.
        if input_dtype == ttnn.int8:
            q_i32 = ttnn.from_torch(q_tr, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
            rq_i32 = ttnn.to_torch(
                ttnn.requantize(q_i32, in_scale_tt, in_zero_point, out_scale_tt, out_zero_point, axis=axis)
            )
            assert torch.equal(rq_fused, rq_i32), "int8 per-channel requantize diverges from int32"
            rq_comp_i32 = ttnn.to_torch(
                ttnn.requantize(q_i32, in_scale_tt, in_zp_tt, out_scale_tt, out_zp_tt, axis=axis)
            )
            assert torch.equal(rq_comp, rq_comp_i32), "int8 all-tensor per-channel requantize diverges from int32"


def test_quantize_per_channel_scalar_zero_point_saturation(device):
    """Pin the int8 saturation of the fused per-channel quantize path.

    The QUANT LLK rounds through the SFPU's FP32_TO_INT8 stage, so an int32 output holds
    int8-range values. The saturation is symmetric at [-127, 127] rather than the [-128, 127]
    of a two's-complement int8, because the rounding stage produces sign-magnitude. The fused
    per-channel path therefore behaves exactly like per-tensor quantize has always done. The
    composite (per-channel tensor zero-point) path narrows with a plain typecast instead and
    does not saturate, so the two disagree once values leave that range. The divergence
    predates this change on the per-tensor side; it is pinned here so the per-channel
    behaviour is a deliberate, visible choice rather than an accident. Unifying the composite
    path is left to the composite-op cleanup TODO in
    ttnn/cpp/ttnn/operations/eltwise/quantization/quantization.cpp.
    """
    row = [-20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 20.0]
    input_tr = torch.tensor([row], dtype=torch.float32).repeat(32, 1)
    scale = 0.05  # x / scale reaches +-400, well outside the int8 range
    axis_size = input_tr.shape[-1]
    scale_vec = torch.full((axis_size,), scale, dtype=torch.float32)
    zp_vec = torch.zeros(axis_size, dtype=torch.int32)

    input_tt = ttnn.from_torch(input_tr, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    scale_tt = ttnn.from_torch(scale_vec, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    zp_vec_tt = ttnn.from_torch(zp_vec, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    expected = torch.clamp(torch.round(input_tr / scale), -127, 127).to(torch.int32)

    fused_tr = ttnn.to_torch(ttnn.quantize(input_tt, scale_tt, 0, axis=-1)).to(torch.int32)
    assert torch.equal(fused_tr, expected), f"got {fused_tr[0].tolist()} expected {expected[0].tolist()}"

    # Same saturation as the per-tensor fused path, which is the behaviour being matched.
    per_tensor_tr = ttnn.to_torch(ttnn.quantize(input_tt, scale, 0)).to(torch.int32)
    assert torch.equal(fused_tr, per_tensor_tr)

    # Documents (does not endorse) the composite path's lack of saturation.
    comp_tr = ttnn.to_torch(ttnn.quantize(input_tt, scale_tt, zp_vec_tt, axis=-1)).to(torch.int32)
    assert comp_tr.abs().max().item() > 127


NARROW_INPUT_DTYPES = [(ttnn.uint8, torch.uint8, 0, 256), (ttnn.int8, torch.int8, -128, 128)]


@pytest.mark.parametrize("input_dtype,torch_dtype,q_lo,q_hi", NARROW_INPUT_DTYPES)
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("scale_dim", [0, 1])
@pytest.mark.parametrize("zero_point", [0, 7, 255])
def test_dequant_narrow_input_with_tensor_zero_point(
    device, input_dtype, torch_dtype, q_lo, q_hi, output_dtype, scale_dim, zero_point
):
    torch.manual_seed(0)
    scale = 0.02

    q_tr = torch.randint(q_lo, q_hi, (2, 3, 64, 96), dtype=torch.int32)
    q_narrow = ttnn.from_torch(q_tr.to(torch_dtype), dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    q_int32 = ttnn.from_torch(q_tr, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    scale_arg = convert_scalar_to_ttnn_tensor(device, scale, scale_dim, ttnn.float32)
    zero_point_tt = convert_scalar_to_ttnn_tensor(device, zero_point, 1, ttnn.int32)

    from_narrow = ttnn.to_torch(ttnn.dequantize(q_narrow, scale_arg, zero_point_tt, dtype=output_dtype))
    from_int32 = ttnn.to_torch(ttnn.dequantize(q_int32, scale_arg, zero_point_tt, dtype=output_dtype))

    assert torch.equal(from_narrow, from_int32)

    # int32 is only a reference, so also pin the absolute value. -128 is the byte the sign-magnitude
    # misread collapses onto 0, and randint over the full range always covers it at this size.
    if output_dtype == ttnn.float32:
        assert torch.equal(from_narrow, (q_tr.to(torch.float32) - zero_point) * scale)


@pytest.mark.parametrize("input_dtype,torch_dtype,q_lo,q_hi", NARROW_INPUT_DTYPES)
@pytest.mark.parametrize("narrow_output", [False, True])
def test_requant_narrow_input_with_tensor_zero_point(
    device, input_dtype, torch_dtype, q_lo, q_hi, narrow_output, expect_error
):
    torch.manual_seed(0)
    in_scale, in_zero_point = 0.02, 7
    out_scale, out_zero_point = 0.05, 3

    q_tr = torch.randint(q_lo, q_hi, (2, 3, 64, 96), dtype=torch.int32)
    q_narrow = ttnn.from_torch(q_tr.to(torch_dtype), dtype=input_dtype, layout=ttnn.TILE_LAYOUT, device=device)
    q_int32 = ttnn.from_torch(q_tr, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    in_zp_tt = convert_scalar_to_ttnn_tensor(device, in_zero_point, 1, ttnn.int32)
    out_zp_tt = convert_scalar_to_ttnn_tensor(device, out_zero_point, 1, ttnn.int32)

    output_dtype = input_dtype if narrow_output else ttnn.int32
    # An int8 output is rejected here on purpose: reading int8 is handled, but the decomposed
    # composite narrows with a typecast, which is not int8-safe (#50401). Pin the guard so that
    # relaxing it has to come with QUANT-LLK narrowing.
    if output_dtype == ttnn.int8:
        with expect_error(RuntimeError, "only supports int32 output"):
            ttnn.requantize(q_narrow, in_scale, in_zp_tt, out_scale, out_zp_tt, dtype=output_dtype)
        return

    from_narrow = ttnn.to_torch(ttnn.requantize(q_narrow, in_scale, in_zp_tt, out_scale, out_zp_tt, dtype=output_dtype))
    from_int32 = ttnn.to_torch(ttnn.requantize(q_int32, in_scale, in_zp_tt, out_scale, out_zp_tt, dtype=output_dtype))

    assert torch.equal(from_narrow, from_int32)


def test_quantize_tensor_zero_point_honors_memory_config(device):
    """Composite (tensor zero-point) path must honor caller memory_config.

    Regression for the DRAM-input / L1-output case: without forwarding
    memory_config into the final typecast, the output incorrectly stays in DRAM.
    """
    torch.manual_seed(0)
    input_tr = torch.rand(64, 128, dtype=torch.float32)
    axis = 1
    scale, zero_point = calculate_scale_zero_point_per_channel(input_tr, axis, -128, 127)

    input_tt = ttnn.from_torch(
        input_tr,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    scale_tt = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    zero_point_tt = ttnn.from_torch(zero_point, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)

    quantized_tt = ttnn.quantize(
        input_tt,
        scale_tt,
        zero_point_tt,
        axis=axis,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    assert (
        quantized_tt.memory_config() == ttnn.L1_MEMORY_CONFIG
    ), f"expected L1 output, got {quantized_tt.memory_config()}"


@pytest.mark.parametrize(
    "use_tensor_scale,axis",
    [
        (True, 1),  # Tensor+Tensor per-channel composite arm
        (False, None),  # float+Tensor per-tensor composite arm
    ],
)
def test_quantize_tensor_zero_point_honors_output_tensor(device, use_tensor_scale, axis):
    """Composite path must write into a preallocated INT32 output_tensor."""
    torch.manual_seed(0)
    input_tr = torch.rand(64, 128, dtype=torch.float32)

    input_tt = ttnn.from_torch(
        input_tr,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if use_tensor_scale:
        scale, zero_point = calculate_scale_zero_point_per_channel(input_tr, axis, -128, 127)
        scale_arg = ttnn.from_torch(scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        zero_point_tt = ttnn.from_torch(zero_point, dtype=ttnn.int32, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        scale, zero_point = calculate_scale_zero_point_per_tensor(input_tr, -128, 127)
        scale_arg = scale
        zero_point_tt = convert_scalar_to_ttnn_tensor(device, zero_point, 1, ttnn.int32)

    output_tt = ttnn.zeros(
        input_tr.shape,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    quantized_tt = ttnn.quantize(
        input_tt,
        scale_arg,
        zero_point_tt,
        axis=axis,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_tensor=output_tt,
    )
    assert quantized_tt.buffer_address() == output_tt.buffer_address(), (
        f"expected output to alias preallocated tensor "
        f"(got {quantized_tt.buffer_address()} vs {output_tt.buffer_address()})"
    )
    assert (
        quantized_tt.memory_config() == ttnn.L1_MEMORY_CONFIG
    ), f"expected L1 output, got {quantized_tt.memory_config()}"
