# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

EXPERT_FP4_BLOCK_SIZE = 32
EXPERT_WEIGHT_ABI = "deepseek_v4_flash.fp4_e2m1fn_x2.block32.v1"


def pack_fp4_indices(indices: torch.Tensor) -> torch.Tensor:
    if indices.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError(f"Expected integer FP4 indices, got {indices.dtype}")
    if indices.shape[-1] % 2 != 0:
        raise ValueError(f"Last dimension must be even for FP4 packing, got {indices.shape[-1]}")
    if torch.any((indices < 0) | (indices > 15)):
        raise ValueError("FP4 indices must be in [0, 15]")

    values = indices.to(torch.uint8)
    low = values[..., 0::2]
    high = values[..., 1::2]
    return torch.bitwise_or(low, torch.bitwise_left_shift(high, 4)).contiguous()


def unpack_fp4_indices(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype not in (torch.uint8, torch.int8):
        raise TypeError(f"Expected uint8/int8 packed FP4 tensor, got {packed.dtype}")
    values = packed.to(torch.uint8)
    low = torch.bitwise_and(values, 0x0F)
    high = torch.bitwise_and(torch.bitwise_right_shift(values, 4), 0x0F)
    unpacked = torch.stack((low, high), dim=-1)
    return unpacked.flatten(-2).contiguous()


def dequantize_fp4_packed(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_size: int = EXPERT_FP4_BLOCK_SIZE,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    indices = unpack_fp4_indices(packed)
    if indices.shape[-1] % block_size != 0:
        raise ValueError(
            f"Unpacked last dimension {indices.shape[-1]} must be divisible by block_size {block_size}"
        )
    expected_scale_shape = indices.shape[:-1] + (indices.shape[-1] // block_size,)
    if tuple(scale.shape) != tuple(expected_scale_shape):
        raise ValueError(f"Expected scale shape {expected_scale_shape}, got {tuple(scale.shape)}")

    values = FP4_E2M1_TABLE.to(device=indices.device)[indices.long()]
    scale_values = scale.float().repeat_interleave(block_size, dim=-1)
    return (values * scale_values).to(dtype).contiguous()
