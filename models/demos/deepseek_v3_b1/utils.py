# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import struct

import torch

import ttnn


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


def stitch_weight_tensors(
    weight_specs: list[dict],
    device: ttnn.Device,
    unified_grid: ttnn.CoreRangeSet,
    memory_layout: ttnn.TensorMemoryLayout = ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    tile: ttnn.Tile = None,
) -> tuple[ttnn.Tensor, dict]:
    """
    Stitch multiple matmul weight tensors into a unified 2D tensor for efficient CB usage.

    Concatenates weights along the width (N) dimension, allowing cores to read their
    designated column ranges. This works well when:
    - All weights have compatible tile sizes
    - All weights use the same data type (e.g., bfp8)
    - Core grids may overlap (cores can participate in multiple matmuls)

    Args:
        weight_specs: List of dicts with:
            - 'tensor': torch.Tensor [K, N] (on host)
            - 'grid': ttnn.CoreRangeSet (which cores should process this weight)
            - 'name': str (identifier for compile-time args)
        device: ttnn.Device
        unified_grid: Common grid that covers all participating cores (union of all grids)
        memory_layout: Sharding strategy (WIDTH_SHARDED recommended for matmul weights)
        dtype: Data type for the output tensor (default: bfloat8_b for weights)
        tile: Tile descriptor (must be same for all weights)

    Returns:
        - stitched_tensor: ttnn.Tensor with shape [max_K, sum_of_Ns]
        - metadata: Dict with per-tensor column offsets and shapes for kernel args

    Example:
        weight_specs = [
            {'tensor': matmul1_weights, 'grid': matmul1_grid, 'name': 'matmul1'},  # [7168, 1536]
            {'tensor': matmul2_weights, 'grid': matmul2_grid, 'name': 'matmul2'},  # [1536, 12288]
            {'tensor': matmul3_weights, 'grid': matmul3_grid, 'name': 'matmul3'},  # [8192, 512]
        ]

        stitched, metadata = stitch_weight_tensors(
            weight_specs, device, unified_grid,
            memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            dtype=ttnn.bfloat8_b,
        )

        # Result: [8192, 14336] tensor with column layout:
        # columns [0:1536]     -> matmul1 weights
        # columns [1536:13824] -> matmul2 weights
        # columns [13824:14336] -> matmul3 weights

        # Metadata for kernel compile-time args:
        metadata = {
            'unified_shape': (8192, 14336),
            'weights': {
                'matmul1': {
                    'col_start': 0, 'col_end': 1536,
                    'original_shape': (7168, 1536),
                    'col_start_tiles': 0, 'width_tiles': 48,  # 1536/32
                },
                'matmul2': {...},
                'matmul3': {...},
            }
        }
    """
    if tile is None:
        tile = ttnn.Tile([1, 32])

    if len(weight_specs) == 0:
        raise ValueError("weight_specs cannot be empty")

    # Validate all tensors have same tile and dtype
    TILE_HEIGHT = tile.get_tile_shape()[0]
    TILE_WIDTH = tile.get_tile_shape()[1]

    # Find max height (K dimension) across all tensors
    max_height = max(spec["tensor"].shape[0] for spec in weight_specs)

    # Align height to tile boundaries
    max_height = ((max_height + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT

    # Pad tensors to max_height and concatenate along width (N dimension)
    padded_tensors = []
    metadata = {
        "unified_shape": None,  # Will be set after concatenation
        "weights": {},
        "tile": tile,
        "dtype": dtype,
    }

    current_col_offset = 0

    for spec in weight_specs:
        tensor = spec["tensor"]
        orig_height, orig_width = tensor.shape

        # Validate dimensions are tile-aligned
        if orig_width % TILE_WIDTH != 0:
            raise ValueError(f"Tensor '{spec['name']}' width {orig_width} must be divisible by tile width {TILE_WIDTH}")

        # Pad height to max_height if needed
        if orig_height < max_height:
            padding = torch.zeros(max_height - orig_height, orig_width, dtype=tensor.dtype, device=tensor.device)
            padded_tensor = torch.cat([tensor, padding], dim=0)
        else:
            padded_tensor = tensor

        padded_tensors.append(padded_tensor)

        # Calculate column range in elements
        col_start = current_col_offset
        col_end = current_col_offset + orig_width

        # Calculate column range in tiles
        col_start_tiles = col_start // TILE_WIDTH
        width_tiles = orig_width // TILE_WIDTH

        # Store metadata for this weight tensor
        metadata["weights"][spec["name"]] = {
            "col_start": col_start,
            "col_end": col_end,
            "original_shape": (orig_height, orig_width),
            "padded_shape": (max_height, orig_width),
            "col_start_tiles": col_start_tiles,
            "width_tiles": width_tiles,
            "grid": spec["grid"],
        }

        current_col_offset = col_end

    # Concatenate all padded tensors along width dimension
    stitched_torch = torch.cat(padded_tensors, dim=1)  # [max_height, total_width]
    total_width = stitched_torch.shape[1]

    # Validate total width is tile-aligned
    if total_width % TILE_WIDTH != 0:
        raise ValueError(f"Total width {total_width} must be divisible by tile width {TILE_WIDTH}")

    metadata["unified_shape"] = (max_height, total_width)
    metadata["total_width_tiles"] = total_width // TILE_WIDTH
    metadata["total_height_tiles"] = max_height // TILE_HEIGHT

    # Calculate shard shape based on unified grid and memory layout
    grid_ranges = list(unified_grid.ranges())
    if len(grid_ranges) == 0:
        raise ValueError("Unified grid must have at least one range")

    # Get grid dimensions
    grid_size = grid_ranges[0].grid_size()
    grid_rows = grid_size.y
    grid_cols = grid_size.x

    # Calculate shard shape
    if memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        # Distribute width across cores
        shard_width = total_width // grid_cols
        shard_height = max_height

        # Ensure tile alignment
        if shard_width % TILE_WIDTH != 0:
            shard_width = ((shard_width + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH

    elif memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        # Distribute height across cores
        shard_height = max_height // grid_rows
        shard_width = total_width

        # Ensure tile alignment
        if shard_height % TILE_HEIGHT != 0:
            shard_height = ((shard_height + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT

    elif memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        # Distribute both dimensions
        shard_height = max_height // grid_rows
        shard_width = total_width // grid_cols

        # Ensure tile alignment
        shard_height = ((shard_height + TILE_HEIGHT - 1) // TILE_HEIGHT) * TILE_HEIGHT
        shard_width = ((shard_width + TILE_WIDTH - 1) // TILE_WIDTH) * TILE_WIDTH
    else:
        raise ValueError(f"Unsupported memory layout: {memory_layout}")

    # Create shard spec
    shard_spec = ttnn.ShardSpec(
        grid=unified_grid,
        shape=(shard_height, shard_width),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )

    memory_config = ttnn.MemoryConfig(
        memory_layout=memory_layout,
        buffer_type=ttnn.BufferType.L1,
        shard_spec=shard_spec,
    )

    # Create ttnn tensor
    stitched_tensor = ttnn.from_torch(
        stitched_torch,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        tile=tile,
        device=device,
        memory_config=memory_config,
    )

    metadata["shard_shape"] = (shard_height, shard_width)
    metadata["memory_layout"] = memory_layout
    metadata["unified_grid"] = unified_grid

    return stitched_tensor, metadata
