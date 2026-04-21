"""Benchmark binary ``ttnn.lt`` / ``ttnn.gt`` / ``ttnn.le`` / ``ttnn.ge`` for ``int32`` tensors.

Mirrors ``test_eq_gt_binary_benchmark.py`` but exercises the ``calculate_binary_comp_int32``
SFPU path on Wormhole, which shares an 8-instruction "core" between ``lt``/``ge`` and
between ``gt``/``le`` via the replay buffer.

Tensor shapes are ``(32, 32)`` and ``(1024, 1024)`` (stored as ``[1, 1, H, W]``) on
interleaved-L1, height-sharded, and block-sharded layouts.

Example (tracy, one warmup + one signposted iteration):

.. code-block:: bash

   TT_METAL_DEVICE_PROFILER_NOC_EVENTS=1 TT_METAL_DEVICE_PROFILER=1 \\
     python -m tracy -r -v \\
       -m pytest tests/ttnn/benchmark/python/test_lt_gt_le_ge_int32_binary_benchmark.py::test_lt_gt_le_ge_int32_tracy_once -v
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

import ttnn
from tracy import signpost

TILE_HW = 32


def _pick_height_core_count(grid_x: int, height: int, width: int) -> int:
    for n in (8, 4, 2, 1):
        if n > grid_x or height % n != 0:
            continue
        shard_h = height // n
        if shard_h % TILE_HW == 0 and width % TILE_HW == 0:
            return n
    return 1


def _pick_block_grid_rows_cols(grid: ttnn.CoreCoord, height: int, width: int) -> tuple[int, int]:
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


_OP_FNS = {
    "lt": ttnn.lt,
    "gt": ttnn.gt,
    "le": ttnn.le,
    "ge": ttnn.ge,
}


def _setup_lt_gt_le_ge_int32(device, binary_op, spatial_layout, dim):
    """Build int32 inputs and return ``(op_fn, input_a, input_b)`` after device sync."""
    torch.manual_seed(0)
    shape = (1, 1, dim, dim)

    # Pick ranges that exercise both the "same-sign subtract" and the
    # "opposite-sign fallback" code paths inside the replay-buffered core:
    # spanning both signs guarantees a mix of both cases.
    torch_a = torch.randint(-(2**30), 2**30, shape, dtype=torch.int32)
    torch_b = torch.randint(-(2**30), 2**30, shape, dtype=torch.int32)

    mem_config = _memory_config_spatial(device, spatial_layout, dim, dim)

    input_a = ttnn.from_torch(
        torch_a,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    input_b = ttnn.from_torch(
        torch_b,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mem_config,
    )
    ttnn.synchronize_device(device)

    if binary_op not in _OP_FNS:
        raise AssertionError(binary_op)
    return _OP_FNS[binary_op], input_a, input_b


@pytest.mark.parametrize("binary_op", ["lt", "gt", "le", "ge"])
@pytest.mark.parametrize("spatial_layout", ["interleaved_l1", "height_sharded", "block_sharded"])
@pytest.mark.parametrize("dim", [32, 1024])
def test_benchmark_lt_gt_le_ge_int32(
    benchmark,
    device,
    binary_op,
    spatial_layout,
    dim,
):
    op_fn, input_a, input_b = _setup_lt_gt_le_ge_int32(device, binary_op, spatial_layout, dim)

    def run_binary():
        op_fn(input_a, input_b)
        ttnn.synchronize_device(device)

    benchmark.pedantic(run_binary, iterations=10, rounds=2, warmup_rounds=1)


@pytest.mark.parametrize("binary_op", ["lt", "gt", "le", "ge"])
@pytest.mark.parametrize("spatial_layout", ["interleaved_l1", "height_sharded", "block_sharded"])
@pytest.mark.parametrize("dim", [32, 1024])
def test_lt_gt_le_ge_int32_tracy_once(device, binary_op, spatial_layout, dim):
    """One program-cache warmup + one signposted measured iteration per (op, layout, dim)."""
    op_fn, input_a, input_b = _setup_lt_gt_le_ge_int32(device, binary_op, spatial_layout, dim)

    op_fn(input_a, input_b)
    ttnn.synchronize_device(device)

    signpost("start")
    op_fn(input_a, input_b)
    ttnn.synchronize_device(device)
    signpost("stop")

    ttnn.synchronize_device(device)
