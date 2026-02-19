# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from models.demos.deepseek_v3.utils.dequantize import dequantize_tensor


def _reference_dequantize(tensor: torch.Tensor, inv_scale: torch.Tensor, block_shape: tuple[int, ...]) -> torch.Tensor:
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
    torch.manual_seed(sum(shape) + sum(block_shape) + int(input_dtype == torch.bfloat16))
    inv_shape = tuple(math.ceil(shape[i] / block_shape[i]) for i in range(len(shape)))

    quantized = torch.randn(shape, dtype=torch.float32).to(input_dtype)
    inv_scale = torch.randn(inv_shape, dtype=torch.float32)

    expected = _reference_dequantize(quantized, inv_scale, block_shape)
    actual = dequantize_tensor(quantized, inv_scale, block_shape)

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_dequantize_tensor_rejects_invalid_rank():
    quantized = torch.randn((2, 3), dtype=torch.float32).to(torch.float8_e4m3fn)
    inv_scale = torch.randn((1,), dtype=torch.float32)

    with pytest.raises(ValueError, match="same ndim"):
        dequantize_tensor(quantized, inv_scale, (2, 2))
