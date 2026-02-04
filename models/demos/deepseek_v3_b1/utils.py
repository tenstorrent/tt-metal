# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import struct

import torch


def shuffle_weights_for_interleaved_qnope_qrope(
    weights: torch.Tensor,
    num_qnope_heads: int = 64,
    num_qrope_heads: int = 64,
    qnope_head_dim: int = 128,
    qrope_head_dim: int = 64,
    heads_per_row: int = 8,
) -> torch.Tensor:
    """
    Shuffle matmul2 weight columns for interleaved Qnope/Qrope output layout.

    The matmul2 output needs to be distributed to a grid where each row has:
    - 8 Qnope cores (1 head per core, 128 elements each)
    - 4 Qrope cores (2 heads per core, 64 elements each = 128 total)

    This function reorders the weight columns so the output is interleaved by row groups:
    [QNOPE_0:8 | QROPE_0:8 | QNOPE_8:16 | QROPE_8:16 | ...]

    Args:
        weights: Input weight matrix [K, N] where N = num_qnope_heads*qnope_head_dim + num_qrope_heads*qrope_head_dim
        num_qnope_heads: Number of Qnope heads (default 64)
        num_qrope_heads: Number of Qrope heads (default 64)
        qnope_head_dim: Dimension per Qnope head (default 128)
        qrope_head_dim: Dimension per Qrope head (default 64)
        heads_per_row: Number of heads per grid row for both Qnope and Qrope (default 8)

    Returns:
        Shuffled weight matrix [K, N] with interleaved column order
    """
    K = weights.shape[0]
    qnope_total = num_qnope_heads * qnope_head_dim  # 64 * 128 = 8192
    qrope_total = num_qrope_heads * qrope_head_dim  # 64 * 64 = 4096

    # Split Qnope and Qrope columns
    qnope_weights = weights[:, :qnope_total]  # [K, 8192]
    qrope_weights = weights[:, qnope_total : qnope_total + qrope_total]  # [K, 4096]

    # Reshape to per-head: [K, num_heads, head_dim]
    qnope_heads = qnope_weights.reshape(K, num_qnope_heads, qnope_head_dim)
    qrope_heads = qrope_weights.reshape(K, num_qrope_heads, qrope_head_dim)

    # Calculate number of rows
    num_rows = num_qnope_heads // heads_per_row  # 64 / 8 = 8 rows

    # Interleave by row groups
    shuffled_cols = []
    for row in range(num_rows):
        # Qnope heads for this row: heads [row*8 : row*8+8]
        qnope_start = row * heads_per_row
        qnope_row = qnope_heads[:, qnope_start : qnope_start + heads_per_row, :]
        shuffled_cols.append(qnope_row.reshape(K, -1))  # [K, 8*128=1024]

        # Qrope heads for this row: heads [row*8 : row*8+8]
        qrope_start = row * heads_per_row
        qrope_row = qrope_heads[:, qrope_start : qrope_start + heads_per_row, :]
        shuffled_cols.append(qrope_row.reshape(K, -1))  # [K, 8*64=512]

    return torch.cat(shuffled_cols, dim=1)  # [K, 12288]


def unshuffle_output_from_interleaved_qnope_qrope(
    output: torch.Tensor,
    num_qnope_heads: int = 64,
    num_qrope_heads: int = 64,
    qnope_head_dim: int = 128,
    qrope_head_dim: int = 64,
    heads_per_row: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unshuffle interleaved Qnope/Qrope output back to separate contiguous tensors.

    Inverse of shuffle_weights_for_interleaved_qnope_qrope for the output.

    Args:
        output: Interleaved output [1, N] where N = num_qnope_heads*qnope_head_dim + num_qrope_heads*qrope_head_dim
        num_qnope_heads: Number of Qnope heads (default 64)
        num_qrope_heads: Number of Qrope heads (default 64)
        qnope_head_dim: Dimension per Qnope head (default 128)
        qrope_head_dim: Dimension per Qrope head (default 64)
        heads_per_row: Number of heads per grid row (default 8)

    Returns:
        Tuple of (qnope_output, qrope_output):
        - qnope_output: [num_qnope_heads, 1, qnope_head_dim] = [64, 1, 128]
        - qrope_output: [num_qrope_heads, 1, qrope_head_dim] = [64, 1, 64]
    """
    num_rows = num_qnope_heads // heads_per_row  # 8 rows
    qnope_per_row = heads_per_row * qnope_head_dim  # 8 * 128 = 1024
    qrope_per_row = heads_per_row * qrope_head_dim  # 8 * 64 = 512
    row_size = qnope_per_row + qrope_per_row  # 1536

    qnope_chunks = []
    qrope_chunks = []

    for row in range(num_rows):
        row_start = row * row_size
        qnope_chunk = output[:, row_start : row_start + qnope_per_row]
        qrope_chunk = output[:, row_start + qnope_per_row : row_start + row_size]
        qnope_chunks.append(qnope_chunk)
        qrope_chunks.append(qrope_chunk)

    # Concatenate and reshape to [num_heads, 1, head_dim]
    qnope_output = torch.cat(qnope_chunks, dim=1).reshape(num_qnope_heads, 1, qnope_head_dim)
    qrope_output = torch.cat(qrope_chunks, dim=1).reshape(num_qrope_heads, 1, qrope_head_dim)

    return qnope_output, qrope_output


def float_to_bfloat16_packed(value):
    """Convert float to packed bfloat16 (two copies in uint32)"""
    # Convert float32 to bytes
    float_bytes = struct.pack("f", value)
    # Extract upper 16 bits (bfloat16 is truncated float32)
    bf16_bytes = float_bytes[2:4]  # upper 16 bits in little-endian layout
    # Pack two copies into uint32 (little endian)
    packed = int.from_bytes(bf16_bytes + bf16_bytes, byteorder="little")
    return packed


def float_to_uint32(value):
    """Convert float to uint32"""
    return int.from_bytes(struct.pack("f", value), byteorder="little")
