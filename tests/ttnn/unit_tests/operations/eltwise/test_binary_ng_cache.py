# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Program cache validation tests for binary_ng operations.

Tests verify that the program cache correctly handles:
1. Different worker_grid values (Bug #1 fix)
2. Scalar presence/absence (Bug #2 fix)
3. Different tensor shapes and broadcast patterns (Bug #3 fix)
"""

import pytest
import torch
import ttnn
from loguru import logger

pytestmark = pytest.mark.use_module_device


def run_binary_ng(device, shape_a, shape_b=None, scalar=None, sub_core_grids=None, op=ttnn.add):
    """Helper to run binary_ng operation with various configurations."""
    torch.manual_seed(0)

    # Create input tensors
    input_a = torch.randn(shape_a, dtype=torch.bfloat16)
    tt_input_a = ttnn.from_torch(input_a, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if scalar is not None:
        # Scalar operation
        result = op(tt_input_a, scalar, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    elif shape_b is not None:
        # Tensor-tensor operation
        input_b = torch.randn(shape_b, dtype=torch.bfloat16)
        tt_input_b = ttnn.from_torch(input_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if sub_core_grids is not None:
            result = ttnn.experimental.operations.primary.binary_ng(
                tt_input_a, tt_input_b, ttnn.BinaryOpType.ADD, sub_core_grids=sub_core_grids
            )
        else:
            result = op(tt_input_a, tt_input_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    else:
        raise ValueError("Must provide either shape_b or scalar")

    return result.cpu()


@pytest.mark.parametrize(
    "shape_a,shape_b",
    [
        ([1, 1, 32, 32], [1, 1, 32, 32]),  # No broadcast
        ([1, 1, 64, 64], [1, 1, 64, 64]),  # Larger shape, no broadcast
    ],
)
def test_binary_ng_cache_hit_identical_params(device, shape_a, shape_b):
    """Test that identical operations produce cache hits."""
    num_cache_entries = []

    for iteration in range(2):
        run_binary_ng(device, shape_a, shape_b)
        # Dummy tensor to prevent address reuse
        _ = ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_cache_entries.append(device.num_program_cache_entries())
        logger.info(f"Iteration {iteration}: cache entries = {num_cache_entries[-1]}")

    # Verify cache hit: entries shouldn't increase on second run
    assert num_cache_entries[0] > 0, "Cache should have entries after first run"
    assert num_cache_entries[0] == num_cache_entries[1], \
        f"Cache hit expected: entries increased from {num_cache_entries[0]} to {num_cache_entries[1]}"


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
def test_binary_ng_cache_miss_different_worker_grids(device, sub_core_grids_list):
    """
    Bug #1 Fix Verification: Different worker_grid values should produce different cache entries.

    Before fix: Different worker grids would hash to same value, causing incorrect core distribution.
    After fix: worker_grid is included in hash, ensuring different programs for different grids.
    """
    shape_a = [1, 1, 64, 64]
    shape_b = [1, 1, 64, 64]

    cache_entries_before = device.num_program_cache_entries()

    # Run with first sub_core_grid
    run_binary_ng(device, shape_a, shape_b, sub_core_grids=sub_core_grids_list[0])
    cache_entries_after_first = device.num_program_cache_entries()

    # Run with second sub_core_grid
    run_binary_ng(device, shape_a, shape_b, sub_core_grids=sub_core_grids_list[1])
    cache_entries_after_second = device.num_program_cache_entries()

    logger.info(f"Cache entries: before={cache_entries_before}, "
                f"after_first={cache_entries_after_first}, "
                f"after_second={cache_entries_after_second}")

    # Different worker grids should create different cache entries
    assert cache_entries_after_first > cache_entries_before, \
        "First operation should create cache entry"
    assert cache_entries_after_second > cache_entries_after_first, \
        "Different worker_grid should create new cache entry (Bug #1 fix)"


@pytest.mark.parametrize(
    "shape",
    [
        [1, 1, 32, 32],
        [1, 1, 64, 64],
    ],
)
def test_binary_ng_cache_miss_scalar_vs_tensor(device, shape):
    """
    Bug #2 Fix Verification: Operations with scalar vs tensor should have different cache entries.

    Before fix: scalar.has_value() was not hashed, could cause collisions.
    After fix: scalar.has_value() is included in hash.
    """
    cache_entries_before = device.num_program_cache_entries()

    # Run tensor-tensor operation
    run_binary_ng(device, shape, shape_b=shape)
    cache_entries_after_tensor = device.num_program_cache_entries()

    # Run tensor-scalar operation
    run_binary_ng(device, shape, scalar=2.5)
    cache_entries_after_scalar = device.num_program_cache_entries()

    logger.info(f"Cache entries: before={cache_entries_before}, "
                f"after_tensor={cache_entries_after_tensor}, "
                f"after_scalar={cache_entries_after_scalar}")

    # Scalar vs tensor should create different cache entries
    assert cache_entries_after_tensor > cache_entries_before, \
        "Tensor operation should create cache entry"
    assert cache_entries_after_scalar > cache_entries_after_tensor, \
        "Scalar operation should create different cache entry (Bug #2 fix)"


@pytest.mark.parametrize(
    "shape_pairs",
    [
        # Different broadcast patterns with same output shape
        [
            ([1, 1, 32, 32], [1, 1, 32, 32]),  # No broadcast
            ([1, 1, 32, 32], [1, 1, 1, 32]),   # Row broadcast
            ([1, 1, 32, 32], [1, 1, 32, 1]),   # Column broadcast
            ([1, 1, 32, 32], [1, 1, 1, 1]),    # Scalar broadcast
        ],
    ],
)
def test_binary_ng_cache_miss_different_broadcast_patterns(device, shape_pairs):
    """
    Bug #3 Fix Verification: Different broadcast patterns should produce different cache entries.

    Before fix: Only shard_volumes was hashed, not tensor shapes, causing collisions
                for different broadcast patterns.
    After fix: logical_shape and padded_shape are explicitly hashed for both inputs.
    """
    initial_cache_entries = device.num_program_cache_entries()
    cache_entries = [initial_cache_entries]

    for i, (shape_a, shape_b) in enumerate(shape_pairs):
        run_binary_ng(device, shape_a, shape_b)
        current_entries = device.num_program_cache_entries()
        cache_entries.append(current_entries)
        logger.info(f"Shape pair {i} {shape_a} + {shape_b}: cache entries = {current_entries}")

    # Each different broadcast pattern should create a new cache entry
    for i in range(1, len(cache_entries)):
        assert cache_entries[i] > cache_entries[i-1], \
            f"Shape pair {i-1} should create new cache entry (Bug #3 fix): " \
            f"{cache_entries[i-1]} -> {cache_entries[i]}"


@pytest.mark.parametrize(
    "shapes_and_layouts",
    [
        # Same shape, different layouts
        [
            ([1, 1, 64, 64], ttnn.TILE_LAYOUT),
            ([1, 1, 64, 64], ttnn.ROW_MAJOR_LAYOUT),
        ],
    ],
)
def test_binary_ng_cache_miss_different_layouts(device, shapes_and_layouts):
    """
    Bug #3 Extended: Different layouts should produce different cache entries.

    After fix: layout() is explicitly hashed for both inputs.
    """
    cache_entries_before = device.num_program_cache_entries()

    # Run with first layout
    shape, layout1 = shapes_and_layouts[0]
    input_a = torch.randn(shape, dtype=torch.bfloat16)
    tt_input_a = ttnn.from_torch(input_a, layout=layout1, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input_b = torch.randn(shape, dtype=torch.bfloat16)
    tt_input_b = ttnn.from_torch(input_b, layout=layout1, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _ = ttnn.add(tt_input_a, tt_input_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cache_entries_after_first = device.num_program_cache_entries()

    # Run with second layout
    shape, layout2 = shapes_and_layouts[1]
    input_a = torch.randn(shape, dtype=torch.bfloat16)
    tt_input_a = ttnn.from_torch(input_a, layout=layout2, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    input_b = torch.randn(shape, dtype=torch.bfloat16)
    tt_input_b = ttnn.from_torch(input_b, layout=layout2, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    _ = ttnn.add(tt_input_a, tt_input_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    cache_entries_after_second = device.num_program_cache_entries()

    logger.info(f"Cache entries: before={cache_entries_before}, "
                f"after_layout1={cache_entries_after_first}, "
                f"after_layout2={cache_entries_after_second}")

    # Different layouts should create different cache entries
    assert cache_entries_after_first > cache_entries_before, \
        "First layout should create cache entry"
    assert cache_entries_after_second > cache_entries_after_first, \
        "Different layout should create new cache entry (Bug #3 fix - layout hashing)"


@pytest.mark.parametrize(
    "shapes_with_same_volume",
    [
        # Different shapes with same volume (512 elements)
        [
            [1, 1, 32, 64],
            [1, 1, 64, 32],
            [1, 1, 128, 16],
        ],
    ],
)
def test_binary_ng_cache_miss_different_shapes_same_volume(device, shapes_with_same_volume):
    """
    Bug #3 Extended: Different shapes with same volume should produce different cache entries.

    This verifies that we hash actual shapes, not just volumes.
    """
    initial_cache_entries = device.num_program_cache_entries()
    cache_entries = [initial_cache_entries]

    for i, shape in enumerate(shapes_with_same_volume):
        run_binary_ng(device, shape, shape_b=shape)
        current_entries = device.num_program_cache_entries()
        cache_entries.append(current_entries)
        logger.info(f"Shape {i} {shape}: cache entries = {current_entries}")

    # Each different shape should create a new cache entry, even with same volume
    for i in range(1, len(cache_entries)):
        assert cache_entries[i] > cache_entries[i-1], \
            f"Shape {i-1} should create new cache entry: " \
            f"{shapes_with_same_volume[i-1]} -> {shapes_with_same_volume[i]}"
