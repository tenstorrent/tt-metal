# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

pytestmark = pytest.mark.use_module_device

import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics


# ---------------------------------------------------------------------------
# Helpers for sharded RM reduce tests
# ---------------------------------------------------------------------------


def _height_sharded_mem_config(grid, shard_H, shard_W, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """Return a HEIGHT_SHARDED L1 MemoryConfig with the given shard shape."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_H, shard_W], orientation),
    )


def _width_sharded_mem_config(grid, shard_H, shard_W, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """Return a WIDTH_SHARDED L1 MemoryConfig with the given shard shape."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_H, shard_W], orientation),
    )


def _block_sharded_mem_config(grid, shard_H, shard_W, orientation=ttnn.ShardOrientation.ROW_MAJOR):
    """Return a BLOCK_SHARDED L1 MemoryConfig with the given shard shape."""
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, [shard_H, shard_W], orientation),
    )


def _line_grid(num_cores, horizontal=False):
    """Return a CoreRangeSet that is a line of `num_cores` cores.

    horizontal=False → column strip x=0, y=0..num_cores-1 (good for HEIGHT_SHARDED)
    horizontal=True  → row strip x=0..num_cores-1, y=0   (good for WIDTH_SHARDED)
    """
    if horizontal:
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, num_cores - 1))})


def _block_grid(num_cores_x, num_cores_y):
    """Return a CoreRangeSet that is a 2D block of (num_cores_x × num_cores_y) cores (good for BLOCK_SHARDED)."""
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores_x - 1, num_cores_y - 1))})


_SHARDING_COMBOS = [
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.HEIGHT_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.TensorMemoryLayout.BLOCK_SHARDED),
]


def _factor_block_grid(num_cores):
    """Factor num_cores into (cores_x, cores_y) for a 2D block-sharded grid; prefers square-ish."""
    import math

    for cx in range(int(math.isqrt(num_cores)), 0, -1):
        if num_cores % cx == 0:
            return cx, num_cores // cx
    return 1, num_cores


def _sharded_mem_config(sharding, num_cores, total_rows, total_cols):
    """Build a sharded MemoryConfig sized to (total_rows, total_cols) across `num_cores`.

    BLOCK_SHARDED factors num_cores into a roughly-square 2D grid (cores_x x cores_y);
    both dimensions must divide evenly (cores_y | total_rows and cores_x | total_cols).
    """
    if sharding == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        grid = _line_grid(num_cores, horizontal=False)
        return _height_sharded_mem_config(grid, total_rows // num_cores, total_cols)
    if sharding == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        grid = _line_grid(num_cores, horizontal=True)
        return _width_sharded_mem_config(grid, total_rows, total_cols // num_cores)
    # BLOCK_SHARDED
    cores_x, cores_y = _factor_block_grid(num_cores)
    grid = _block_grid(cores_x, cores_y)
    return _block_sharded_mem_config(grid, total_rows // cores_y, total_cols // cores_x)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("in_sharding, out_sharding", _SHARDING_COMBOS)
@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((1, 1, 512, 512), 4),
        ((2, 1, 128, 128), 4),
        ((2, 2, 32, 128), 4),
    ],
)
def test_mean_rm_h_sharded(device, dtype, keepdim, in_sharding, out_sharding, shape, num_cores):
    N, C, H, W = shape
    total_rows = N * C * H
    output_rows = N * C

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch_dtype)
    torch_ref = torch.mean(torch_input, dim=-2, keepdim=keepdim)

    in_mem = _sharded_mem_config(in_sharding, num_cores, total_rows, W)
    out_mem = _sharded_mem_config(out_sharding, num_cores, output_rows, W)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem
        # torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    output_tensor = ttnn.mean(input_tensor, dim=-2, keepdim=keepdim, memory_config=out_mem)

    output = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16])
@pytest.mark.parametrize("keepdim", [True, False])
@pytest.mark.parametrize("in_sharding, out_sharding", _SHARDING_COMBOS)
@pytest.mark.parametrize(
    "shape, num_cores",
    [
        ((1, 1, 256, 128), 4),
        ((1, 2, 256, 128), 4),
        ((2, 2, 64, 128), 4),
    ],
)
def test_mean_sharded_W_reduce(device, dtype, keepdim, in_sharding, out_sharding, shape, num_cores):
    N, C, H, W = shape
    total_rows = N * C * H
    output_cols = 1

    torch_dtype = torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32
    torch.manual_seed(0)
    torch_input = torch.rand(shape, dtype=torch_dtype)
    torch_ref = torch.mean(torch_input, dim=-1, keepdim=keepdim)

    in_mem = _sharded_mem_config(in_sharding, num_cores, total_rows, W)
    out_mem = _sharded_mem_config(out_sharding, num_cores, total_rows, output_cols)

    input_tensor = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=in_mem
        # torch_input, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mem
    )
    output_tensor = ttnn.mean(input_tensor, dim=-1, keepdim=keepdim, memory_config=out_mem)

    output = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_ref,
        output,
        pcc_threshold=0.999,
        rtol=0.008,
        atol=0.004,
        frobenius_threshold=0.01,
        check_ulp=False,
    )
