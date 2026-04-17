# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Benchmark binary ``ttnn.eq`` and ``ttnn.gt`` on interleaved L1, height-sharded, and block-sharded memory layouts.

Tensor shapes are ``(32, 32)`` and ``(1024, 1024)`` (stored as ``[1, 1, H, W]`` for shard compatibility).

**32×32 note:** Tile layout requires shard shapes to align to the 32×32 tile grid. Multi-core height/block
shards of a single 32×32 tile are not generally valid, so height- and block-sharded cases use a **1×1**
core grid with shard shape ``[32, 32]`` (layout enum differs from interleaved; useful for comparing paths).

**1024×1024:** Height sharding uses up to 8 cores in row 0 (``shard = [1024/n, 1024]``). Block sharding
prefers a ``4×4`` core grid when the device grid permits, else ``2×2``, else ``1×1``.

Example:

.. code-block:: bash

   pytest tests/ttnn/benchmark/python/test_eq_gt_binary_benchmark.py -v --benchmark-only
   pytest tests/ttnn/benchmark/python/test_eq_gt_binary_benchmark.py -v --benchmark-json=eq_gt.json

Tracy (``test_eq_gt_binary_tracy_once``: one warmup + one signposted iteration; no pytest-benchmark):

.. code-block:: bash

   python -m tracy -v -r -p -o generated/profiler \\
     -m "pytest tests/ttnn/benchmark/python/test_eq_gt_binary_benchmark.py -v -k tracy_once"
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from tracy import signpost

TILE_HW = 32


def _pick_height_core_count(grid_x: int, height: int, width: int) -> int:
    """Largest core count on row 0 such that the physical shard is tile-sized.

    For HEIGHT_SHARDED tile tensors, each shard shape is ``[height // n, width]``; both dimensions must be
    multiples of ``TILE_HW`` (see tensor_layout shard alignment).
    """
    for n in (8, 4, 2, 1):
        if n > grid_x or height % n != 0:
            continue
        shard_h = height // n
        if shard_h % TILE_HW == 0 and width % TILE_HW == 0:
            return n
    return 1


def _pick_block_grid_rows_cols(grid: ttnn.CoreCoord, height: int, width: int) -> tuple[int, int]:
    """Pick a block grid (rows, cols) that fits the device and divides (height, width)."""
    if height == 32 and width == 32:
        return (1, 1)
    for rows, cols in ((4, 4), (2, 2), (1, 1)):
        if rows <= grid.y and cols <= grid.x and height % rows == 0 and width % cols == 0:
            return (rows, cols)
    return (1, 1)


def _memory_config_spatial(
    device: ttnn.Device,
    spatial_layout: str,
    height: int,
    width: int,
) -> ttnn.MemoryConfig:
    if spatial_layout == "interleaved_l1":
        return ttnn.L1_MEMORY_CONFIG

    grid = device.compute_with_storage_grid_size()

    if spatial_layout == "height_sharded":
        num_cores = _pick_height_core_count(grid.x, height, width)
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
        shard_shape = [height // num_cores, width]
        shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)

    if spatial_layout == "block_sharded":
        rows, cols = _pick_block_grid_rows_cols(grid, height, width)
        core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(cols - 1, rows - 1))})
        shard_shape = [height // rows, width // cols]
        shard_spec = ttnn.ShardSpec(core_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)

    raise ValueError(f"Unknown spatial_layout: {spatial_layout}")


def _setup_eq_gt_binary(device, binary_op, spatial_layout, dim):
    """Build inputs and return ``(op_fn, input_a, input_b)`` after device sync."""
    torch.manual_seed(0)
    shape = (1, 1, dim, dim)
    torch_a = torch.randn(shape, dtype=torch.bfloat16)
    torch_b = torch.randn(shape, dtype=torch.bfloat16)

    mem_config = _memory_config_spatial(device, spatial_layout, dim, dim)

    input_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    input_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    ttnn.synchronize_device(device)

    if binary_op == "eq":
        op_fn = ttnn.eq
    elif binary_op == "gt":
        op_fn = ttnn.gt
    else:
        raise AssertionError(binary_op)

    return op_fn, input_a, input_b


@pytest.mark.parametrize("binary_op", ["eq", "gt"])
@pytest.mark.parametrize("spatial_layout", ["interleaved_l1", "height_sharded", "block_sharded"])
@pytest.mark.parametrize("dim", [32, 1024])
def test_benchmark_eq_gt_binary(
    benchmark,
    device,
    binary_op,
    spatial_layout,
    dim,
):
    op_fn, input_a, input_b = _setup_eq_gt_binary(device, binary_op, spatial_layout, dim)

    def run_binary():
        op_fn(input_a, input_b)
        ttnn.synchronize_device(device)

    benchmark.pedantic(run_binary, iterations=10, rounds=2, warmup_rounds=1)


@pytest.mark.parametrize("binary_op", ["eq", "gt"])
@pytest.mark.parametrize("spatial_layout", ["interleaved_l1", "height_sharded", "block_sharded"])
@pytest.mark.parametrize("dim", [32, 1024])
def test_eq_gt_binary_tracy_once(device, binary_op, spatial_layout, dim):
    """Same cases as ``test_benchmark_eq_gt_binary``: one program-cache warmup, then one measured op under Tracy signposts."""
    op_fn, input_a, input_b = _setup_eq_gt_binary(device, binary_op, spatial_layout, dim)

    op_fn(input_a, input_b)
    ttnn.synchronize_device(device)

    signpost("start")
    op_fn(input_a, input_b)
    ttnn.synchronize_device(device)
    signpost("stop")

    ttnn.synchronize_device(device)
