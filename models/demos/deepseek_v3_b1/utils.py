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


def stitch_weight_tensors(
    matmul1_weights: torch.Tensor,
    matmul2_weights: torch.Tensor,
    n1_per_core: int,
    n2_per_core: int,
    tile_size: int = 32,
) -> torch.Tensor:
    """
    Stitch two width-sharded weight matrices into a single flat tile-packed tensor.

    Creates a tensor with shard shape (tile_size, total_tiles * tile_size) where
    matmul1 tiles are stored contiguously followed by matmul2 tiles. This allows
    custom_mm_block to read each matmul's weights using the in1_tile_index offset.

    The data is rearranged so that each (tile_size, tile_size) block in the output
    corresponds to one tile of the weight matrix, with tile(j) containing weight
    data for K-range [j*tile_size : (j+1)*tile_size].

    Args:
        matmul1_weights: Weight matrix [K1, N1] for first matmul (width-sharded)
        matmul2_weights: Weight matrix [K2, N2] for second matmul (width-sharded)
        n1_per_core: Shard width for matmul1 (columns per core)
        n2_per_core: Shard width for matmul2 (columns per core)
        tile_size: Tile dimension (default 32)

    Returns:
        Stitched tensor of shape (tile_size, stitched_shard_width * num_cores),
        suitable for width sharding with shard shape (tile_size, stitched_shard_width).
    """
    K1, N1 = matmul1_weights.shape
    K2, N2 = matmul2_weights.shape

    num_cores = N1 // n1_per_core
    assert N2 // n2_per_core == num_cores, (
        f"Both weight tensors must shard across the same number of cores: "
        f"matmul1 has {N1}//{n1_per_core}={N1 // n1_per_core}, "
        f"matmul2 has {N2}//{n2_per_core}={N2 // n2_per_core}"
    )

    k1_tiles = K1 // tile_size
    k2_tiles = K2 // tile_size
    n1_tiles = n1_per_core // tile_size
    n2_tiles = n2_per_core // tile_size

    tiles_per_core_m1 = k1_tiles * n1_tiles
    tiles_per_core_m2 = k2_tiles * n2_tiles
    total_tiles_per_core = tiles_per_core_m1 + tiles_per_core_m2
    stitched_shard_width = total_tiles_per_core * tile_size

    stitched_shards = []
    for core in range(num_cores):
        # Extract per-core shards from original weight matrices
        m1_shard = matmul1_weights[:, core * n1_per_core : (core + 1) * n1_per_core]  # (K1, n1_per_core)
        m2_shard = matmul2_weights[:, core * n2_per_core : (core + 1) * n2_per_core]  # (K2, n2_per_core)

        # Rearrange matmul1 shard: (K1, n1_per_core) -> (tile_size, k1_tiles * n1_tiles * tile_size)
        # Each (tile_size, tile_size) block becomes one tile read by custom_mm_block
        # tile(j) contains weight data for K-range [j*tile_size : (j+1)*tile_size]
        m1 = m1_shard.reshape(k1_tiles, tile_size, n1_tiles, tile_size)  # (K_tiles, tile_r, N_tiles, tile_c)
        m1 = m1.permute(1, 0, 2, 3).reshape(tile_size, tiles_per_core_m1 * tile_size)

        # Rearrange matmul2 shard: (K2, n2_per_core) -> (tile_size, k2_tiles * n2_tiles * tile_size)
        m2 = m2_shard.reshape(k2_tiles, tile_size, n2_tiles, tile_size)  # (K_tiles, tile_r, N_tiles, tile_c)
        m2 = m2.permute(1, 0, 2, 3).reshape(tile_size, tiles_per_core_m2 * tile_size)

        # Concatenate: matmul1 tiles first, then matmul2 tiles
        stitched_shards.append(torch.cat([m1, m2], dim=1))  # (tile_size, stitched_shard_width)

    # Concatenate all core shards along width for width-sharded tensor
    return torch.cat(stitched_shards, dim=1)  # (tile_size, stitched_shard_width * num_cores)


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
