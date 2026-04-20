# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for transpose universal input/output support.

Each test covers one distinct code path — no duplicate shapes or redundant combos.
"""

import torch

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def run_transpose_test(
    shape,
    dim0,
    dim1,
    device,
    input_layout=ttnn.TILE_LAYOUT,
    input_mem_config=None,
    output_mem_config=None,
    dtype=ttnn.bfloat16,
):
    """Helper to run a single transpose test with the given configs."""
    torch.manual_seed(12345)
    x = torch.rand(shape).bfloat16().float()

    if input_mem_config is None:
        input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    ttnn_input = ttnn.from_torch(x, layout=input_layout, dtype=dtype, device=device, memory_config=input_mem_config)
    result = ttnn.transpose(ttnn_input, dim0, dim1, memory_config=output_mem_config)

    if output_mem_config is not None and output_mem_config.shard_spec is not None:
        actual = result.memory_config()
        assert (
            actual.memory_layout == output_mem_config.memory_layout
        ), f"Expected output memory layout {output_mem_config.memory_layout}, got {actual.memory_layout}"

    ref = x.transpose(dim0, dim1)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    passing, output = comp_equal(ref, got)
    logger.info(output)
    assert passing, f"Transpose mismatch for shape={shape}, dims=({dim0},{dim1})"


def _block_shard_config(shape, device):
    """Create a 2x2 block-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    grid_x = min(2, compute_grid.x)
    grid_y = min(2, compute_grid.y)
    total_height = shape[0] * shape[1] * shape[2]
    shard_shape = (total_height // grid_y, shape[3] // grid_x)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


def _height_shard_config(shape, device, num_cores=4):
    """Create a height-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_height = shape[0] * shape[1] * shape[2]
    shard_shape = (total_height // num_cores, shape[3])
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _width_shard_config(shape, device, num_cores=4):
    """Create a width-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_height = shape[0] * shape[1] * shape[2]
    shard_shape = (total_height, shape[3] // num_cores)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


L1_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
DRAM_INTERLEAVED = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)


# ---------------------------------------------------------------------------
# 1. Block-sharded input → interleaved output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_block_sharded_to_interleaved_wh(device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 2. Block-sharded input → interleaved output  (CN, needs N,C > 1)
# ---------------------------------------------------------------------------
def test_transpose_block_sharded_to_interleaved_cn(device):
    shape = (2, 4, 32, 64)
    run_transpose_test(
        shape,
        0,
        1,
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 3. Width-sharded input → interleaved output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_width_sharded_to_interleaved_wh(device):
    shape = (1, 1, 64, 128)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_width_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 4. Interleaved input → height-sharded output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_interleaved_to_height_sharded(device):
    shape = (1, 1, 64, 128)
    out_shape = (1, 1, 128, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=_height_shard_config(out_shape, device),
    )


# ---------------------------------------------------------------------------
# 5. Interleaved input → block-sharded output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_interleaved_to_block_sharded(device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=_block_shard_config(shape, device),
    )


# ---------------------------------------------------------------------------
# 6. Block-sharded input → height-sharded output  (WH)
# ---------------------------------------------------------------------------
def test_transpose_block_to_height_sharded(device):
    shape = (1, 1, 64, 128)
    out_shape = (1, 1, 128, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_block_shard_config(shape, device),
        output_mem_config=_height_shard_config(out_shape, device, num_cores=2),
    )


# ---------------------------------------------------------------------------
# 7. Width-sharded input → height-sharded output  (WH, cross shard-type)
# ---------------------------------------------------------------------------
def test_transpose_width_to_height_sharded(device):
    shape = (1, 1, 64, 128)
    out_shape = (1, 1, 128, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_width_shard_config(shape, device),
        output_mem_config=_height_shard_config(out_shape, device),
    )


# ---------------------------------------------------------------------------
# 8. HC transpose: height-sharded input → interleaved output
# ---------------------------------------------------------------------------
def test_transpose_hc_height_sharded(device):
    shape = (1, 4, 32, 64)
    run_transpose_test(
        shape,
        1,
        2,
        device,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 9. CN transpose: height-sharded input → interleaved output
# ---------------------------------------------------------------------------
def test_transpose_cn_height_sharded(device):
    shape = (2, 4, 32, 64)
    run_transpose_test(
        shape,
        0,
        1,
        device,
        input_mem_config=_height_shard_config(shape, device, num_cores=8),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 10. Default output derivation: block-sharded input, no explicit output
# ---------------------------------------------------------------------------
def test_transpose_block_sharded_default_output(device):
    shape = (1, 1, 64, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_block_shard_config(shape, device),
    )


# ---------------------------------------------------------------------------
# 11. Default output derivation: height-sharded input, no explicit output
# ---------------------------------------------------------------------------
def test_transpose_height_sharded_default_output(device):
    shape = (2, 4, 32, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_height_shard_config(shape, device),
    )


# ---------------------------------------------------------------------------
# 12. Sharded output without shard_spec (system derives it)
# ---------------------------------------------------------------------------
def test_transpose_sharded_output_no_shard_spec(device):
    shape = (1, 1, 128, 64)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=output_mem_config,
    )


# ---------------------------------------------------------------------------
# 13. Native L1 height-sharded (verifies native path still selected)
# ---------------------------------------------------------------------------
def test_transpose_native_height_sharded_wh(device):
    shape = (1, 1, 32, 64)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=_height_shard_config(shape, device, num_cores=1),
    )


# ---------------------------------------------------------------------------
# 14. Interleaved baseline (sanity)
# ---------------------------------------------------------------------------
def test_transpose_interleaved_baseline(device):
    shape = (2, 3, 64, 96)
    run_transpose_test(shape, 2, 3, device)


# ---------------------------------------------------------------------------
# 15. ROW_MAJOR height-sharded input (exercises ROW_MAJOR branch in
# get_transpose_shard_specs / is_shard_tile_aligned)
# ---------------------------------------------------------------------------
def test_transpose_row_major_height_sharded_hc(device):
    shape = (1, 4, 32, 64)
    run_transpose_test(
        shape,
        1,
        2,
        device,
        input_layout=ttnn.ROW_MAJOR_LAYOUT,
        input_mem_config=_height_shard_config(shape, device),
        output_mem_config=L1_INTERLEAVED,
    )


# ---------------------------------------------------------------------------
# 16. DRAM-sharded input falls back to interleaved
# (is_native_transpose_sharding rejects BufferType::DRAM)
# ---------------------------------------------------------------------------
def test_transpose_dram_sharded_fallback(device):
    shape = (1, 1, 128, 64)
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(4, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_height = shape[0] * shape[1] * shape[2]
    shard_spec = ttnn.ShardSpec(shard_grid, (total_height // num_cores, shape[3]), ttnn.ShardOrientation.ROW_MAJOR)
    dram_sharded = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch.manual_seed(12345)
    x = torch.rand(shape).bfloat16().float()
    ttnn_input = ttnn.from_torch(
        x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device, memory_config=dram_sharded
    )
    result = ttnn.transpose(ttnn_input, 2, 3, memory_config=L1_INTERLEAVED)
    actual = result.memory_config()
    assert (
        actual.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    ), f"Expected fallback to INTERLEAVED, got {actual.memory_layout}"

    ref = x.transpose(2, 3)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))
    passing, output = comp_equal(ref, got)
    logger.info(output)
    assert passing, f"Transpose mismatch for DRAM-sharded fallback, shape={shape}"


# ---------------------------------------------------------------------------
# 17. Block-sharded output without shard_spec — exercises the block-shard
# branch of generate_transpose_shard_spec (interleaved input, no spec)
# ---------------------------------------------------------------------------
def test_transpose_block_sharded_output_no_shard_spec(device):
    shape = (1, 1, 128, 128)
    output_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1)
    run_transpose_test(
        shape,
        2,
        3,
        device,
        input_mem_config=L1_INTERLEAVED,
        output_mem_config=output_mem_config,
    )
