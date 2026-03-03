# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor


def _reference_dequantize(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: tuple[int, ...]) -> torch.Tensor:
    """Naive reference implementation of dequantization that uses `repeat_interleave`"""
    expanded = inv_scale
    for dim, block_dim in enumerate(block_shape):
        expanded = expanded.repeat_interleave(block_dim, dim=dim)
    slices = tuple(slice(0, size) for size in tensor.shape)
    return tensor.float() * expanded[slices].float()


@pytest.mark.parametrize(
    "shape,block_shape,input_dtype",
    [
        ((130, 257), (128, 128), torch.float8_e4m3fn),  # non-divisible in both dims
        ((512, 1024), (128, 128), torch.float8_e4m3fn),  # exactly divisible
        ((63, 65), (32, 32), torch.float8_e4m3fn),  # small non-divisible
        ((5, 7, 9), (2, 3, 4), torch.float8_e4m3fn),  # rank-3 coverage
    ],
)
def test_dequantize_tensor_matches_reference(
    shape: tuple[int, ...], block_shape: tuple[int, ...], input_dtype: torch.dtype
):
    torch.manual_seed(1234)
    inv_shape = tuple(math.ceil(shape[i] / block_shape[i]) for i in range(len(shape)))

    quantized = torch.rand(shape, dtype=torch.float32).to(input_dtype)
    inv_scale = torch.rand(inv_shape, dtype=torch.float32)

    expected = _reference_dequantize(quantized, inv_scale, block_shape)
    actual = dequantize_tensor(quantized, inv_scale, block_shape)

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_dequantize_tensor_does_not_mutate_float32_input():
    torch.manual_seed(2026)

    tensor = torch.randn((8, 257, 259), dtype=torch.float32)
    tensor_before = tensor.clone()
    scale = torch.rand((8, 3, 3), dtype=torch.float32) + 0.1

    _ = dequantize_tensor(tensor, scale, (1, 128, 128))

    assert torch.equal(tensor, tensor_before)


def test_dequantize_tensor_matches_reference_in_experts_weight_loading_flow():
    torch.manual_seed(1337)

    num_experts = 8
    weight_shape = (257, 259)
    block_shape = (128, 128)
    block_shape_with_expert_dim = (1, *block_shape)
    scale_shape = tuple(math.ceil(weight_shape[i] / block_shape[i]) for i in range(len(weight_shape)))

    expert_weights = [torch.randn(weight_shape, dtype=torch.float32) for _ in range(num_experts)]
    quant_scales = [torch.rand(scale_shape, dtype=torch.float32) + 0.1 for _ in range(num_experts)]

    # This mirrors add_inv_scale_to_state_dict's quantization path in module tests.
    quantized_old = [
        _reference_dequantize(weight.clone(), scale, block_shape).to(torch.float8_e4m3fn)
        for weight, scale in zip(expert_weights, quant_scales, strict=True)
    ]
    quantized_new = [
        dequantize_tensor(weight.clone(), scale, block_shape).to(torch.float8_e4m3fn)
        for weight, scale in zip(expert_weights, quant_scales, strict=True)
    ]
    inv_scales = [1.0 / scale for scale in quant_scales]

    quantized_old_stacked = torch.stack(quantized_old)
    quantized_new_stacked = torch.stack(quantized_new)
    inv_scales_stacked = torch.stack(inv_scales)

    # This mirrors Experts.convert_weights dequantization path.
    loaded_old = _reference_dequantize(quantized_old_stacked, inv_scales_stacked, block_shape_with_expert_dim)
    loaded_new = dequantize_tensor(quantized_new_stacked, inv_scales_stacked, block_shape_with_expert_dim)

    assert torch.equal(quantized_new_stacked, quantized_old_stacked)
    assert torch.allclose(loaded_new, loaded_old, atol=1e-6, rtol=1e-6)


def test_dequantize_tensor_rejects_invalid_rank():
    quantized = torch.randn((2, 3), dtype=torch.float32).to(torch.float8_e4m3fn)
    inv_scale = torch.randn((1,), dtype=torch.float32)

    with pytest.raises(ValueError, match="same ndim"):
        dequantize_tensor(quantized, inv_scale, (2, 2))
