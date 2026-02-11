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


def tile_reshape(
    tensor: torch.Tensor,
    src_shape: tuple[int, int],
    dst_shape: tuple[int, int],
    tile_h: int = 32,
    tile_w: int = 32,
) -> torch.Tensor:
    """
    Reshape a 2D tensor while preserving the row-major tile ordering.

    In tile layout, data is stored as a grid of (tile_h x tile_w) tiles scanned
    in row-major order.  A naive ``torch.reshape`` changes which values land in
    each tile.  This helper keeps the *contents* of every tile unchanged by:

    1. Splitting the tensor into its tile grid.
    2. Flattening the grid to a 1-D sequence (row-major).
    3. Re-gridding the sequence into the destination tile dimensions.

    The total number of tiles must be the same for source and destination shapes.

    Args:
        tensor: Input tensor of shape ``src_shape``.
        src_shape: ``(H, W)`` – current height and width (must be tile-aligned).
        dst_shape: ``(H', W')`` – desired height and width (must be tile-aligned
            and have the same total tile count as ``src_shape``).
        tile_h: Tile height (default 32).
        tile_w: Tile width (default 32).

    Returns:
        Tensor of shape ``dst_shape`` whose row-major tile sequence is identical
        to the source's.
    """
    src_h, src_w = src_shape
    dst_h, dst_w = dst_shape
    src_tr, src_tc = src_h // tile_h, src_w // tile_w
    dst_tr, dst_tc = dst_h // tile_h, dst_w // tile_w
    assert src_tr * src_tc == dst_tr * dst_tc, f"Tile count mismatch: {src_tr * src_tc} vs {dst_tr * dst_tc}"
    # (H, W) -> (tile_rows, tile_h, tile_cols, tile_w) -> (tile_rows, tile_cols, tile_h, tile_w)
    tiles = tensor.reshape(src_tr, tile_h, src_tc, tile_w).permute(0, 2, 1, 3)
    # Flatten to linear tile sequence, then re-grid to destination layout
    tiles = tiles.reshape(-1, tile_h, tile_w).reshape(dst_tr, dst_tc, tile_h, tile_w)
    # (dst_tile_rows, dst_tile_cols, tile_h, tile_w) -> (dst_H, dst_W)
    return tiles.permute(0, 2, 1, 3).reshape(dst_h, dst_w)


def merge_width_sharded_weights(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    tensor1_grid_x: int,
    tensor2_grid_x: int,
    grid_y: int,
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Merge two width-sharded weight tensors into a single contiguous buffer per bank.

    ``tensor1`` lives on the *smaller* grid (``tensor1_grid_x × grid_y`` cores)
    and ``tensor2`` lives on the *larger* grid (``tensor2_grid_x × grid_y``
    cores).  Both grids share the same rows; ``tensor1`` covers a prefix of the
    columns (``0 .. tensor1_grid_x - 1``).

    The merged tensor uses ``tensor2``'s grid.  Each bank stores::

        [tensor1_shard | tensor2_shard_reshaped]   (cores with both tensors)
        [zero_padding  | tensor2_shard_reshaped]   (cores without tensor1)

    ``tensor2``'s per-core shard is tile-reshaped from
    ``(H2, shard_w2)`` to ``(H2 * shard_w2 / shard_w1, shard_w1)`` so that
    every shard in the merged tensor has a uniform width equal to
    ``tensor1``'s shard width.

    Args:
        tensor1: First weight tensor ``(H1, W1)``, width-sharded on the smaller
            grid.
        tensor2: Second weight tensor ``(H2, W2)``, width-sharded on the larger
            grid.
        tensor1_grid_x: Number of columns in ``tensor1``'s core grid.
        tensor2_grid_x: Number of columns in the merged / ``tensor2`` core grid.
        grid_y: Number of rows (identical for both grids).

    Returns:
        ``(merged_tensor, merged_shard_shape)`` where *merged_tensor* has shape
        ``(merged_H, merged_W)`` and *merged_shard_shape* is
        ``(shard_height, shard_width)``.
    """
    tensor1_num_cores = tensor1_grid_x * grid_y
    tensor2_num_cores = tensor2_grid_x * grid_y

    H1, W1 = tensor1.shape
    H2, W2 = tensor2.shape

    tensor1_shard_w = W1 // tensor1_num_cores
    tensor2_shard_w = W2 // tensor2_num_cores

    # After tile-reshape, each tensor2 shard becomes (reshaped_h, tensor1_shard_w)
    tensor2_reshaped_h = H2 * tensor2_shard_w // tensor1_shard_w

    merged_shard_h = H1 + tensor2_reshaped_h
    merged_shard_w = tensor1_shard_w

    merged = torch.zeros((merged_shard_h, merged_shard_w * tensor2_num_cores), dtype=tensor1.dtype)

    for shard_idx in range(tensor2_num_cores):
        core_x = shard_idx % tensor2_grid_x
        core_y = shard_idx // tensor2_grid_x

        col_start = shard_idx * merged_shard_w
        col_end = col_start + merged_shard_w

        # First region: tensor1 data (or zero-padding for cores outside tensor1's grid)
        if core_x < tensor1_grid_x:
            t1_shard_idx = core_y * tensor1_grid_x + core_x
            t1_start = t1_shard_idx * tensor1_shard_w
            t1_end = t1_start + tensor1_shard_w
            merged[:H1, col_start:col_end] = tensor1[:, t1_start:t1_end]

        # Second region: tensor2 data, tile-reshaped to uniform shard width
        t2_start = shard_idx * tensor2_shard_w
        t2_end = t2_start + tensor2_shard_w
        t2_shard = tensor2[:, t2_start:t2_end]
        t2_reshaped = tile_reshape(t2_shard, (H2, tensor2_shard_w), (tensor2_reshaped_h, merged_shard_w))
        merged[H1:, col_start:col_end] = t2_reshaped

    return merged, (merged_shard_h, merged_shard_w)


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
