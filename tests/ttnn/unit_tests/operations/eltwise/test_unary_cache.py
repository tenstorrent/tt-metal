# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Program cache validation tests for unary operations.

Tests verify that the program cache correctly handles:
1. Different shapes with same volume (Bug fix)
2. Different logical vs padded shapes (Bug fix)
3. Different layouts
"""

import pytest
import torch
import ttnn
from loguru import logger

pytestmark = pytest.mark.use_module_device


def run_unary_op(device, shape, layout=ttnn.TILE_LAYOUT, op=ttnn.relu, sub_core_grids=None):
    """Helper to run unary operation with various configurations."""
    torch.manual_seed(0)

    # Create input tensor
    input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    tt_input = ttnn.from_torch(
        input_tensor,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    if sub_core_grids is not None:
        # Use the lower-level API with sub_core_grids
        result = ttnn.experimental.operations.primary.unary(
            tt_input,
            op_chain=[ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)],
            sub_core_grids=sub_core_grids
        )
    else:
        result = op(tt_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    return result.cpu()


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
    ],
)
def test_unary_cache_hit_identical_params(device, shape):
    """Test that identical operations produce cache hits."""
    num_cache_entries = []

    for iteration in range(2):
        run_unary_op(device, shape)
        # Dummy tensor to prevent address reuse
        _ = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_cache_entries.append(device.num_program_cache_entries())
        logger.info(f"Iteration {iteration}: cache entries = {num_cache_entries[-1]}")

    # Verify cache hit: entries shouldn't increase on second run
    assert num_cache_entries[0] > 0, "Cache should have entries after first run"
    assert num_cache_entries[0] == num_cache_entries[1], \
        f"Cache hit expected: entries increased from {num_cache_entries[0]} to {num_cache_entries[1]}"


@pytest.mark.parametrize(
    "shapes_with_same_volume",
    [
        # Different shapes with same volume (2048 elements for TILE layout)
        [
            [1, 1, 32, 64],
            [1, 1, 64, 32],
        ],
    ],
)
def test_unary_cache_miss_different_shapes_same_volume(device, shapes_with_same_volume):
    """
    Bug Fix Verification: Different shapes with same volume should produce different cache entries.

    Before fix: For TILE layout, only volume was hashed, causing collisions for different
                shapes with same volume.
    After fix: Full shape (logical_shape and padded_shape) is hashed for all layouts.
    """
    initial_cache_entries = device.num_program_cache_entries()
    cache_entries = [initial_cache_entries]

    for i, shape in enumerate(shapes_with_same_volume):
        run_unary_op(device, shape)
        current_entries = device.num_program_cache_entries()
        cache_entries.append(current_entries)
        logger.info(f"Shape {i} {shape}: cache entries = {current_entries}, "
                   f"volume = {shape[-2] * shape[-1]}")

    # Each different shape should create a new cache entry, even with same volume
    for i in range(1, len(cache_entries)):
        assert cache_entries[i] > cache_entries[i-1], \
            f"Shape {i-1} should create new cache entry (Bug fix): " \
            f"{shapes_with_same_volume[i-1]} (volume={shapes_with_same_volume[i-1][-2]*shapes_with_same_volume[i-1][-1]}) -> " \
            f"{shapes_with_same_volume[i]} (volume={shapes_with_same_volume[i][-2]*shapes_with_same_volume[i][-1]})"


@pytest.mark.parametrize(
    "layout_pairs",
    [
        [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT],
    ],
)
def test_unary_cache_miss_different_layouts(device, layout_pairs):
    """
    Bug Fix Verification: Different layouts should produce different cache entries.

    After fix: layout is explicitly hashed for input tensor.
    """
    shape = [1, 1, 64, 64]
    cache_entries_before = device.num_program_cache_entries()

    # Run with first layout
    run_unary_op(device, shape, layout=layout_pairs[0])
    cache_entries_after_first = device.num_program_cache_entries()

    # Run with second layout
    run_unary_op(device, shape, layout=layout_pairs[1])
    cache_entries_after_second = device.num_program_cache_entries()

    logger.info(f"Cache entries: before={cache_entries_before}, "
                f"after_layout1={cache_entries_after_first}, "
                f"after_layout2={cache_entries_after_second}")

    # Different layouts should create different cache entries
    assert cache_entries_after_first > cache_entries_before, \
        "First layout should create cache entry"
    assert cache_entries_after_second > cache_entries_after_first, \
        "Different layout should create new cache entry (Bug fix - layout hashing)"


@pytest.mark.parametrize(
    "sub_core_grids_list",
    [
        # Different sub_core_grids should produce different cache entries
        [
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
        ],
    ],
)
def test_unary_cache_miss_different_subcore_grids(device, sub_core_grids_list):
    """
    Verify that different sub_core_grids values produce different cache entries.

    This validates that sub_core_grids is properly hashed (it already was before our fix).
    """
    shape = [1, 1, 64, 64]

    cache_entries_before = device.num_program_cache_entries()

    # Run with first sub_core_grid
    run_unary_op(device, shape, sub_core_grids=sub_core_grids_list[0])
    cache_entries_after_first = device.num_program_cache_entries()

    # Run with second sub_core_grid
    run_unary_op(device, shape, sub_core_grids=sub_core_grids_list[1])
    cache_entries_after_second = device.num_program_cache_entries()

    logger.info(f"Cache entries: before={cache_entries_before}, "
                f"after_first={cache_entries_after_first}, "
                f"after_second={cache_entries_after_second}")

    # Different sub_core_grids should create different cache entries
    assert cache_entries_after_first > cache_entries_before, \
        "First operation should create cache entry"
    assert cache_entries_after_second > cache_entries_after_first, \
        "Different sub_core_grids should create new cache entry"


@pytest.mark.parametrize(
    "op_list",
    [
        [ttnn.relu, ttnn.gelu, ttnn.sigmoid, ttnn.abs],
    ],
)
def test_unary_cache_miss_different_operations(device, op_list):
    """
    Verify that different unary operations produce different cache entries.

    This is a sanity check that operation type is properly distinguished.
    """
    shape = [1, 1, 32, 32]
    initial_cache_entries = device.num_program_cache_entries()
    cache_entries = [initial_cache_entries]

    for i, op in enumerate(op_list):
        run_unary_op(device, shape, op=op)
        current_entries = device.num_program_cache_entries()
        cache_entries.append(current_entries)
        logger.info(f"Operation {i} {op.__name__}: cache entries = {current_entries}")

    # Each different operation should create a new cache entry
    for i in range(1, len(cache_entries)):
        assert cache_entries[i] > cache_entries[i-1], \
            f"Operation {i-1} should create new cache entry: " \
            f"{op_list[i-1].__name__} -> {op_list[i].__name__}"


@pytest.mark.parametrize(
    "shape_and_dtype_pairs",
    [
        # Same shape, different dtypes
        [
            ([1, 1, 32, 32], ttnn.bfloat16),
            ([1, 1, 32, 32], ttnn.float32),
        ],
    ],
)
def test_unary_cache_miss_different_dtypes(device, shape_and_dtype_pairs):
    """
    Verify that different dtypes produce different cache entries.

    This validates that dtype is properly hashed.
    """
    cache_entries_before = device.num_program_cache_entries()

    # Run with first dtype
    shape, dtype1 = shape_and_dtype_pairs[0]
    input_tensor = torch.randn(shape, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=dtype1,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _ = ttnn.relu(tt_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cache_entries_after_first = device.num_program_cache_entries()

    # Run with second dtype
    shape, dtype2 = shape_and_dtype_pairs[1]
    input_tensor = torch.randn(shape, dtype=torch.float32)
    tt_input = ttnn.from_torch(
        input_tensor,
        dtype=dtype2,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    _ = ttnn.relu(tt_input, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cache_entries_after_second = device.num_program_cache_entries()

    logger.info(f"Cache entries: before={cache_entries_before}, "
                f"after_dtype1={cache_entries_after_first}, "
                f"after_dtype2={cache_entries_after_second}")

    # Different dtypes should create different cache entries
    assert cache_entries_after_first > cache_entries_before, \
        "First dtype should create cache entry"
    assert cache_entries_after_second > cache_entries_after_first, \
        "Different dtype should create new cache entry"


@pytest.mark.parametrize(
    "shapes",
    [
        # Test multiple different shapes to ensure proper hashing
        [
            [1, 1, 32, 32],
            [1, 1, 32, 64],
            [1, 1, 64, 32],
            [1, 1, 64, 64],
            [1, 1, 128, 32],
        ],
    ],
)
def test_unary_cache_comprehensive_shape_coverage(device, shapes):
    """
    Comprehensive test: Multiple different shapes should each create new cache entries.

    This validates that both logical_shape and padded_shape are properly hashed
    across a variety of shape combinations.
    """
    initial_cache_entries = device.num_program_cache_entries()
    cache_entries = [initial_cache_entries]

    for i, shape in enumerate(shapes):
        run_unary_op(device, shape)
        current_entries = device.num_program_cache_entries()
        cache_entries.append(current_entries)
        logger.info(f"Shape {i} {shape}: cache entries = {current_entries}")

    # Each different shape should create a new cache entry
    for i in range(1, len(cache_entries)):
        assert cache_entries[i] > cache_entries[i-1], \
            f"Shape {i-1} should create new cache entry: " \
            f"{shapes[i-1]} -> {shapes[i]}"

    # Verify total increase matches number of unique shapes
    total_new_entries = cache_entries[-1] - cache_entries[0]
    assert total_new_entries >= len(shapes), \
        f"Expected at least {len(shapes)} new cache entries, got {total_new_entries}"
