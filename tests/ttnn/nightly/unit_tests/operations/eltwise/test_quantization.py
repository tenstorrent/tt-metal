# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.eltwise.test_quantization import (
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
