# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Utilities for sanitizing traced model configurations in sweep tests.

Traced configs capture exact memory layouts from real model runs, but some
of these configs are incompatible when replayed in isolation (e.g. sharded
memory configs that clash with kernel circular buffers, or shard grids that
exceed the available cores on the test hardware).

These utilities validate and sanitize configs *before* use, avoiding the
need for try/catch recovery patterns in individual test files.
"""

import ttnn
import logging

logger = logging.getLogger(__name__)


def sanitize_memory_config(memory_config, device, tensor_shape=None, layout=None):
    """
    Validate a traced memory config against the device and return a safe config.

    Returns the original memory_config if it's compatible, or DRAM_MEMORY_CONFIG
    if the traced config would cause a runtime error.

    Checks performed:
    - Shard grid fits within device compute grid
    - Shard spec dimensions are valid for the tensor shape and layout
    - For TILE_LAYOUT, shard dimensions are multiples of 32

    Args:
        memory_config: The traced ttnn.MemoryConfig to validate
        device: The ttnn device (or mesh device)
        tensor_shape: Optional tensor shape for additional validation
        layout: Optional ttnn layout (TILE_LAYOUT / ROW_MAJOR_LAYOUT)

    Returns:
        The original memory_config if valid, or ttnn.DRAM_MEMORY_CONFIG as fallback
    """
    if memory_config is None:
        return ttnn.DRAM_MEMORY_CONFIG

    # Non-sharded configs are always safe
    if not _is_sharded(memory_config):
        return memory_config

    # Get device compute grid
    actual_device = device
    if hasattr(device, "get_devices"):
        # Mesh device — use first sub-device
        devices = device.get_devices()
        if devices:
            actual_device = devices[0]

    try:
        grid_size = actual_device.compute_with_storage_grid_size()
        max_cores = grid_size.x * grid_size.y
    except Exception:
        # Can't query device grid — fall back to safe default
        return ttnn.DRAM_MEMORY_CONFIG

    shard_spec = memory_config.shard_spec
    if shard_spec is None:
        return memory_config

    # Check: shard grid cores must not exceed device cores
    shard_grid = shard_spec.grid
    num_shard_cores = shard_grid.num_cores()
    if num_shard_cores > max_cores:
        logger.debug(
            f"Shard grid requires {num_shard_cores} cores but device has {max_cores}. "
            f"Falling back to DRAM."
        )
        return ttnn.DRAM_MEMORY_CONFIG

    # Check: shard grid coordinates must fit within device grid
    bounding_box = shard_grid.bounding_box()
    if bounding_box.end_coord.x >= grid_size.x or bounding_box.end_coord.y >= grid_size.y:
        logger.debug(
            f"Shard grid bounding box ({bounding_box.end_coord.x}, {bounding_box.end_coord.y}) "
            f"exceeds device grid ({grid_size.x}, {grid_size.y}). Falling back to DRAM."
        )
        return ttnn.DRAM_MEMORY_CONFIG

    # Check: for TILE_LAYOUT, shard shape must be tile-aligned
    if layout == ttnn.TILE_LAYOUT:
        shard_shape = shard_spec.shape
        if len(shard_shape) >= 2:
            if shard_shape[-2] % 32 != 0 or shard_shape[-1] % 32 != 0:
                logger.debug(
                    f"Shard shape {shard_shape} not tile-aligned. Falling back to DRAM."
                )
                return ttnn.DRAM_MEMORY_CONFIG

    return memory_config


def sanitize_output_memory_config(memory_config, device, output_shape=None, layout=None):
    """
    Validate a traced output memory config for the *output* tensor.

    This is particularly important for ops like transpose where the output
    shard spec may become invalid after dimension permutation.

    Same interface as sanitize_memory_config but with naming clarity.
    """
    return sanitize_memory_config(memory_config, device, output_shape, layout)


def tile_align_matmul_shapes(shape_a, shape_b, layout_a, layout_b):
    """
    Align the inner matmul dimension when either input uses TILE_LAYOUT.

    Tile layout pads last two dims to multiples of 32. If the inner dimension
    (A.width == B.height) is not tile-aligned, one side will be padded and the
    other won't, causing a dimension mismatch assertion.

    Args:
        shape_a: Tuple shape of tensor A
        shape_b: Tuple shape of tensor B
        layout_a: Layout of tensor A
        layout_b: Layout of tensor B

    Returns:
        (shape_a, shape_b) with aligned inner dimensions
    """
    if len(shape_a) < 2 or len(shape_b) < 2:
        return shape_a, shape_b

    a_is_tile = layout_a == ttnn.TILE_LAYOUT
    b_is_tile = layout_b == ttnn.TILE_LAYOUT

    if not (a_is_tile or b_is_tile):
        return shape_a, shape_b

    inner_a = shape_a[-1]   # A's width
    inner_b = shape_b[-2]   # B's height
    aligned = _tile_align(max(inner_a, inner_b))

    if inner_a != aligned:
        shape_a = tuple(list(shape_a[:-1]) + [aligned])
    if inner_b != aligned:
        shape_b = tuple(list(shape_b[:-2]) + [aligned, shape_b[-1]])

    return shape_a, shape_b


def _is_sharded(memory_config):
    """Check if a memory config uses sharded memory layout."""
    if hasattr(memory_config, "is_sharded"):
        return memory_config.is_sharded()
    if hasattr(memory_config, "memory_layout"):
        return memory_config.memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED
    return False


def _tile_align(dim):
    """Round up to the nearest multiple of 32."""
    return ((dim + 31) // 32) * 32
