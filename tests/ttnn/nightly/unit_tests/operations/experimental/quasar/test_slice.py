# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Quasar-native ttnn.experimental.quasar.slice: bit-exact against torch slicing."""

import itertools

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

QUASAR_ONLY = "Quasar-native kernel (multi-threaded data-movement kernel)"

RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT
L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG

_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.int64,
    ttnn.uint16: torch.int32,
    ttnn.uint8: torch.uint8,
}


def _quasar_only(device):
    if device.arch() != ttnn.device.Arch.QUASAR:
        pytest.skip(QUASAR_ONLY)


_seeds = itertools.count(1)


def _random(shape, dtype, seed=None):
    """Random data that differs on every call: a device buffer is often reallocated at the address of an
    earlier test's output, so repeating data would let a page the kernel never wrote pass as correct."""
    g = torch.Generator().manual_seed(next(_seeds) if seed is None else seed)
    if dtype in (ttnn.bfloat16, ttnn.float32):
        return torch.randn(shape, generator=g, dtype=torch.float32).to(_TORCH_DTYPE[dtype])
    high = {ttnn.int32: 2**31 - 1, ttnn.uint32: 2**32 - 1, ttnn.uint16: 2**16 - 1, ttnn.uint8: 2**8 - 1}[dtype]
    low = -(2**31) if dtype == ttnn.int32 else 0
    return torch.randint(low, high, shape, generator=g, dtype=torch.int64).to(_TORCH_DTYPE[dtype])


def _torch_slice(x, begins, ends, steps):
    return x[tuple(slice(b, e, s) for b, e, s in zip(begins, ends, steps))]


def _assert_exact(expected, actual):
    assert tuple(actual.shape) == tuple(expected.shape), f"shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    if expected.dtype == torch.bfloat16:
        expected, actual = expected.float(), actual.float()
    else:
        actual = actual.to(expected.dtype)
    mismatches = int((expected != actual).sum())
    assert mismatches == 0, f"{mismatches} of {expected.numel()} elements differ"


def _run(device, shape, begins, ends, steps, layout, dtype=ttnn.bfloat16, in_mc=DRAM, out_mc=None, **kwargs):
    x = _random(shape, dtype)
    tt_in = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=in_mc)
    tt_out = ttnn.experimental.quasar.slice(
        tt_in, list(begins), list(ends), list(steps), memory_config=out_mc, **kwargs
    )
    assert tt_out.layout == layout
    assert tt_out.dtype == dtype
    assert tt_out.memory_config().buffer_type == (out_mc or in_mc).buffer_type
    _assert_exact(_torch_slice(x, begins, ends, steps), ttnn.to_torch(tt_out))
    return tt_out


# ---------------------------------------------------------------------------
# ROW_MAJOR: a page is a row; the last dim is a byte range (or a strided gather) inside it.
# ---------------------------------------------------------------------------
_RM_CASES = [
    # rank 1
    pytest.param((64,), (0,), (64,), (1,), id="r1_full"),
    pytest.param((64,), (5,), (41,), (1,), id="r1_mid"),
    pytest.param((64,), (63,), (64,), (1,), id="r1_single_element"),
    pytest.param((67,), (3,), (66,), (4,), id="r1_strided"),
    # rank 2
    pytest.param((40, 48), (0, 0), (40, 48), (1, 1), id="r2_full"),
    pytest.param((40, 48), (7, 5), (33, 46), (1, 1), id="r2_unaligned"),
    pytest.param((40, 48), (39, 47), (40, 48), (1, 1), id="r2_single_element"),
    pytest.param((40, 48), (1, 0), (40, 48), (3, 1), id="r2_row_step"),
    pytest.param((40, 48), (0, 1), (40, 47), (1, 2), id="r2_col_step"),
    pytest.param((40, 48), (2, 3), (39, 44), (5, 7), id="r2_both_steps"),
    # rank 3
    pytest.param((3, 20, 33), (1, 2, 3), (3, 19, 30), (1, 1, 1), id="r3_unaligned"),
    pytest.param((5, 20, 33), (0, 1, 0), (5, 20, 33), (2, 3, 4), id="r3_all_steps"),
    pytest.param((5, 20, 33), (4, 19, 32), (5, 20, 33), (1, 1, 1), id="r3_single_element"),
    # rank 4
    pytest.param((2, 3, 34, 36), (0, 0, 0, 0), (2, 3, 34, 36), (1, 1, 1, 1), id="r4_full"),
    pytest.param((2, 3, 34, 36), (1, 1, 5, 7), (2, 3, 31, 29), (1, 1, 1, 1), id="r4_unaligned"),
    pytest.param((2, 3, 34, 36), (0, 0, 1, 2), (2, 3, 33, 35), (1, 2, 2, 3), id="r4_all_steps"),
    pytest.param((2, 3, 34, 36), (0, 2, 0, 0), (2, 3, 34, 36), (1, 1, 1, 1), id="r4_channel"),
    pytest.param((1, 1, 64, 64), (0, 0, 32, 0), (1, 1, 64, 64), (1, 1, 1, 1), id="r4_tile_aligned"),
    # rank 5
    pytest.param((2, 2, 3, 10, 12), (1, 0, 1, 2, 3), (2, 2, 3, 9, 11), (1, 1, 1, 1, 1), id="r5_unaligned"),
    pytest.param((2, 2, 3, 10, 12), (0, 0, 0, 1, 0), (2, 2, 3, 10, 12), (1, 2, 2, 3, 5), id="r5_all_steps"),
]


@pytest.mark.parametrize("shape, begins, ends, steps", _RM_CASES)
def test_slice_row_major(device, shape, begins, ends, steps):
    _quasar_only(device)
    _run(device, shape, begins, ends, steps, RM)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8])
@pytest.mark.parametrize(
    "shape, begins, ends, steps",
    [
        pytest.param((2, 3, 18, 21), (0, 1, 2, 3), (2, 3, 17, 20), (1, 1, 1, 1), id="unit_step"),
        pytest.param((2, 3, 18, 21), (1, 0, 1, 2), (2, 3, 18, 21), (1, 2, 4, 3), id="strided"),
    ],
)
def test_slice_row_major_dtypes(device, shape, begins, ends, steps, dtype):
    _quasar_only(device)
    _run(device, shape, begins, ends, steps, RM, dtype=dtype)


@pytest.mark.parametrize("in_mc, out_mc", [(L1, L1), (L1, DRAM), (DRAM, L1)], ids=["l1_l1", "l1_dram", "dram_l1"])
@pytest.mark.parametrize("steps", [(1, 1, 1, 1), (1, 1, 2, 3)], ids=["unit_step", "strided"])
def test_slice_row_major_buffer_types(device, in_mc, out_mc, steps):
    _quasar_only(device)
    _run(device, (1, 2, 24, 40), (0, 0, 3, 5), (1, 2, 23, 39), steps, RM, in_mc=in_mc, out_mc=out_mc)


# A row wider than one NoC chunk (4096 bytes) moves as several chunks; a strided row shrinks the elements
# per chunk so that the source span of a chunk still fits one staging slot.
@pytest.mark.parametrize(
    "shape, begins, ends, steps, dtype",
    [
        pytest.param((3, 5000), (0, 0), (3, 5000), (2, 1), ttnn.bfloat16, id="bf16_3_chunks"),
        pytest.param((2, 3000), (0, 7), (2, 2990), (1, 1), ttnn.float32, id="fp32_unaligned_3_chunks"),
        pytest.param((2, 1024), (0, 0), (2, 1024), (1, 1), ttnn.float32, id="fp32_exactly_one_chunk"),
        pytest.param((2, 1025), (0, 0), (2, 1025), (1, 1), ttnn.float32, id="fp32_one_chunk_plus_one"),
        pytest.param((2, 6000), (0, 5), (2, 5990), (1, 3), ttnn.bfloat16, id="bf16_strided_chunks"),
        pytest.param((2, 3000), (1, 1), (2, 2999), (1, 7), ttnn.float32, id="fp32_strided_chunks"),
        pytest.param((1, 9000), (0, 0), (1, 9000), (1, 4500), ttnn.int32, id="step_wider_than_slot"),
    ],
)
def test_slice_row_major_wide_rows(device, shape, begins, ends, steps, dtype):
    _quasar_only(device)
    _run(device, shape, begins, ends, steps, RM, dtype=dtype)


# Enough rows that every cluster of the grid gets work and each thread several batches of slots.
@pytest.mark.parametrize("steps", [(1, 1, 1, 1), (1, 1, 1, 3)], ids=["unit_step", "strided"])
def test_slice_row_major_multi_cluster(device, steps):
    _quasar_only(device)
    grid = device.compute_with_storage_grid_size()
    rows = grid.x * grid.y * 6 * 9 + 5  # > clusters x threads x slots, uneven
    _run(device, (1, 2, rows, 20), (0, 0, 3, 1), (1, 2, rows - 2, 19), steps, RM)


# ---------------------------------------------------------------------------
# TILE: tile-aligned, unstrided last two dims move whole tiles; otherwise every output tile is gathered
# element by element from the input tiles it covers.
# ---------------------------------------------------------------------------
_TILE_CASES = [
    # rank 2
    pytest.param((64, 96), (0, 0), (64, 96), (1, 1), id="r2_full"),
    pytest.param((64, 96), (32, 32), (64, 96), (1, 1), id="r2_aligned"),
    pytest.param((64, 96), (32, 64), (50, 81), (1, 1), id="r2_aligned_begin_ragged_end"),
    pytest.param((64, 96), (5, 7), (60, 90), (1, 1), id="r2_unaligned_begin"),
    pytest.param((64, 96), (63, 95), (64, 96), (1, 1), id="r2_single_element"),
    pytest.param((64, 96), (0, 0), (64, 96), (2, 3), id="r2_strided"),
    # rank 3
    pytest.param((3, 64, 64), (1, 32, 0), (3, 64, 64), (1, 1, 1), id="r3_aligned"),
    pytest.param((5, 64, 64), (0, 32, 32), (5, 64, 64), (2, 1, 1), id="r3_upper_step"),
    pytest.param((3, 40, 50), (2, 0, 0), (3, 40, 50), (1, 1, 1), id="r3_padded_input"),
    # rank 4
    pytest.param((2, 3, 64, 96), (0, 0, 0, 0), (2, 3, 64, 96), (1, 1, 1, 1), id="r4_full"),
    pytest.param((2, 3, 64, 96), (1, 1, 32, 32), (2, 3, 64, 96), (1, 1, 1, 1), id="r4_aligned"),
    pytest.param((2, 5, 64, 96), (0, 1, 0, 64), (2, 5, 33, 90), (1, 3, 1, 1), id="r4_upper_step_ragged_end"),
    pytest.param((2, 3, 64, 96), (0, 0, 3, 10), (2, 3, 50, 70), (1, 1, 1, 1), id="r4_unaligned_begin"),
    pytest.param((2, 3, 64, 96), (0, 0, 0, 0), (2, 3, 64, 96), (1, 2, 2, 2), id="r4_strided"),
    pytest.param((1, 1, 128, 64), (0, 0, 96, 0), (1, 1, 128, 64), (1, 1, 1, 1), id="r4_last_tile_row"),
    # rank 5
    pytest.param((2, 2, 3, 32, 64), (1, 0, 1, 0, 32), (2, 2, 3, 32, 64), (1, 1, 1, 1, 1), id="r5_aligned"),
    pytest.param((2, 2, 3, 32, 64), (0, 0, 0, 0, 0), (2, 2, 3, 32, 64), (1, 2, 2, 1, 1), id="r5_upper_steps"),
    # element gather: ragged shapes, 2 x 2 input tiles per output tile, steps wider than a tile
    pytest.param((70, 100), (3, 5), (70, 100), (1, 1), id="r2_ragged_in_and_out"),
    pytest.param((2, 3, 64, 96), (1, 0, 31, 31), (2, 3, 33, 33), (1, 1, 1, 1), id="r4_straddles_four_tiles"),
    pytest.param((1, 200, 100), (0, 1, 2), (1, 200, 99), (1, 40, 33), id="r3_steps_over_a_tile"),
    pytest.param((1, 1, 96, 160), (0, 0, 0, 1), (1, 1, 96, 160), (1, 1, 1, 1), id="r4_width_shift_by_one"),
    pytest.param((1, 1, 96, 160), (0, 0, 1, 0), (1, 1, 96, 160), (1, 1, 1, 1), id="r4_height_shift_by_one"),
    pytest.param((2, 2, 3, 40, 50), (0, 1, 0, 7, 9), (2, 2, 3, 39, 48), (1, 1, 2, 2, 3), id="r5_unaligned_strided"),
]


@pytest.mark.parametrize("shape, begins, ends, steps", _TILE_CASES)
def test_slice_tile(device, shape, begins, ends, steps):
    _quasar_only(device)
    _run(device, shape, begins, ends, steps, TILE)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16])
@pytest.mark.parametrize(
    "begins, ends, steps",
    [
        pytest.param((0, 1, 32, 32), (2, 3, 64, 90), (1, 2, 1, 1), id="whole_tiles"),
        pytest.param((0, 1, 3, 10), (2, 3, 50, 70), (1, 1, 1, 1), id="gather_unaligned"),
        pytest.param((1, 0, 1, 0), (2, 3, 64, 95), (1, 2, 2, 3), id="gather_strided"),
    ],
)
def test_slice_tile_dtypes(device, dtype, begins, ends, steps):
    _quasar_only(device)
    _run(device, (2, 3, 64, 96), begins, ends, steps, TILE, dtype=dtype)


@pytest.mark.parametrize("in_mc, out_mc", [(L1, L1), (L1, DRAM), (DRAM, L1)], ids=["l1_l1", "l1_dram", "dram_l1"])
@pytest.mark.parametrize("begins", [(0, 0, 32, 32), (0, 0, 5, 9)], ids=["whole_tiles", "gather"])
def test_slice_tile_buffer_types(device, in_mc, out_mc, begins):
    _quasar_only(device)
    _run(device, (1, 2, 96, 64), begins, (1, 2, 96, 64), (1, 1, 1, 1), TILE, in_mc=in_mc, out_mc=out_mc)


@pytest.mark.parametrize("begin_w", [64, 70], ids=["whole_tiles", "gather"])
def test_slice_tile_multi_cluster(device, begin_w):
    _quasar_only(device)
    grid = device.compute_with_storage_grid_size()
    height = 32 * (grid.x * grid.y + 4)  # x 8 tiles per row: every cluster gets several tiles per thread
    _run(device, (1, 1, height, 320), (0, 0, 32, begin_w), (1, 1, height, 320), (1, 1, 1, 1), TILE)


# A full-range slice into another memory config is a copy done by the same kernel; with the same memory
# config it returns the input.
@pytest.mark.parametrize("layout", [RM, TILE], ids=["rm", "tile"])
@pytest.mark.parametrize("shape", [(2, 3, 64, 96), (3, 40, 50)], ids=["aligned", "ragged"])
def test_slice_full_range(device, layout, shape):
    _quasar_only(device)
    full = [(0,) * len(shape), shape, (1,) * len(shape)]
    _run(device, shape, *full, layout, in_mc=DRAM, out_mc=L1)
    x = _random(shape, ttnn.bfloat16)
    tt_in = ttnn.from_torch(x, layout=layout, device=device, memory_config=L1)
    same = ttnn.experimental.quasar.slice(tt_in, *[list(v) for v in full])
    assert same.buffer_address() == tt_in.buffer_address()


# ---------------------------------------------------------------------------
# Thread plan, work split and program cache
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("num_threads", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize(
    "shape, begins, ends, steps, layout",
    [
        pytest.param((1, 2, 70, 40), (0, 0, 3, 5), (1, 2, 69, 39), (1, 1, 1, 1), RM, id="rm"),
        pytest.param((1, 2, 70, 40), (0, 0, 3, 5), (1, 2, 69, 39), (1, 1, 2, 3), RM, id="rm_strided"),
        pytest.param((1, 2, 96, 160), (0, 0, 32, 32), (1, 2, 96, 160), (1, 1, 1, 1), TILE, id="tile"),
        pytest.param((1, 2, 96, 160), (0, 0, 5, 7), (1, 2, 90, 150), (1, 1, 1, 1), TILE, id="tile_gather"),
    ],
)
def test_slice_thread_counts(device, monkeypatch, num_threads, shape, begins, ends, steps, layout):
    _quasar_only(device)
    monkeypatch.setenv("TTNN_QSR_SLICE_THREADS", str(num_threads))
    # One cluster, so the threads really share the items of one range.
    one_cluster = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    _run(device, shape, begins, ends, steps, layout, sub_core_grids=one_cluster)


@pytest.mark.parametrize("bad", ["0", "7", "four", "-2", "3x"])
def test_slice_thread_count_override_is_validated(device, monkeypatch, expect_error, bad):
    _quasar_only(device)
    monkeypatch.setenv("TTNN_QSR_SLICE_THREADS", bad)
    x = ttnn.from_torch(torch.zeros((1, 1, 8, 8), dtype=torch.bfloat16), layout=RM, device=device)
    with expect_error(RuntimeError, r"TTNN_QSR_SLICE_THREADS must be an integer"):
        ttnn.experimental.quasar.slice(x, [0, 0, 1, 1], [1, 1, 8, 8], [1, 1, 1, 1])


@pytest.mark.parametrize("layout", [RM, TILE], ids=["rm", "tile"])
def test_slice_sub_core_grids(device, layout):
    _quasar_only(device)
    grid = device.compute_with_storage_grid_size()
    if grid.x < 3 or grid.y < 2:
        pytest.skip(f"needs a 3x2 compute grid, device has {grid.x}x{grid.y}")
    sub = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 0)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 1), ttnn.CoreCoord(0, 1)),
        }
    )
    _run(device, (1, 3, 96, 96), (0, 1, 32, 32), (1, 3, 96, 96), (1, 1, 1, 1), layout, sub_core_grids=sub)


@pytest.mark.parametrize("layout", [RM, TILE], ids=["rm", "tile"])
def test_slice_program_cache(device, layout):
    """Same slice on fresh buffers hits the cache; different begins of the same shape must not alias it."""
    _quasar_only(device)
    shape, steps = (1, 2, 96, 96), (1, 1, 1, 1)
    keep = [_run(device, shape, (0, 0, 32, 32), (1, 2, 64, 64), steps, layout)]
    entries = device.num_program_cache_entries()
    # Hold the first output so the second run lands in different buffers.
    keep.append(_run(device, shape, (0, 0, 32, 32), (1, 2, 64, 64), steps, layout))
    assert device.num_program_cache_entries() == entries
    keep.append(_run(device, shape, (0, 1, 64, 0), (1, 2, 96, 32), steps, layout))
    assert device.num_program_cache_entries() > entries


# ---------------------------------------------------------------------------
# Preallocated output, tensor begins, sharded tensors
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape, begins, ends, steps, layout",
    [
        pytest.param((1, 2, 40, 48), (0, 1, 3, 5), (1, 2, 39, 47), (1, 1, 1, 1), RM, id="rm"),
        pytest.param((1, 2, 40, 48), (0, 0, 1, 0), (1, 2, 40, 48), (1, 1, 2, 3), RM, id="rm_strided"),
        pytest.param((1, 2, 96, 96), (0, 1, 32, 32), (1, 2, 96, 96), (1, 1, 1, 1), TILE, id="tile"),
        pytest.param((1, 2, 96, 96), (0, 1, 3, 5), (1, 2, 90, 96), (1, 1, 1, 2), TILE, id="tile_gather"),
    ],
)
def test_slice_preallocated_output(device, shape, begins, ends, steps, layout):
    _quasar_only(device)
    x = _random(shape, ttnn.bfloat16)
    ref = _torch_slice(x, begins, ends, steps)
    tt_in = ttnn.from_torch(x, layout=layout, device=device)
    out = ttnn.from_torch(torch.zeros(ref.shape, dtype=torch.bfloat16), layout=layout, device=device, memory_config=L1)
    result = ttnn.experimental.quasar.slice(tt_in, list(begins), list(ends), list(steps), output_tensor=out)
    assert result.buffer_address() == out.buffer_address()
    _assert_exact(ref, ttnn.to_torch(out))


@pytest.mark.parametrize(
    "shape, starts, ends, slice_dim, num_devices",
    [
        pytest.param((1, 1, 64, 64), [0, 0, 32, 0], [1, 1, 64, 64], 2, 2, id="r4_h"),
        pytest.param((1, 1, 64, 128), [0, 0, 0, 96], [1, 1, 64, 128], 3, 4, id="r4_w"),
        pytest.param((1, 4, 32, 64), [0, 2, 0, 0], [1, 4, 32, 64], 1, 2, id="r4_channel"),
        pytest.param((6, 32, 64), [4, 0, 0], [6, 32, 64], 0, 3, id="r3_batch"),
    ],
)
def test_slice_tile_tensor_begins(device, shape, starts, ends, slice_dim, num_devices):
    _quasar_only(device)
    x = _random(shape, ttnn.bfloat16)
    tt_in = ttnn.from_torch(x, layout=TILE, device=device)
    start_t = ttnn.from_torch(torch.tensor(starts), device=device)
    end_t = ttnn.from_torch(torch.tensor(ends), device=device)
    result = ttnn.experimental.quasar.slice(tt_in, start_t, end_t, slice_dim=slice_dim, num_devices=num_devices)
    ref = _torch_slice(x, starts, ends, [1] * len(shape))
    _assert_exact(ref, ttnn.to_torch(result))


def _height_sharded(device, num_cores, shard_shape):
    grid = device.compute_with_storage_grid_size()
    if num_cores > grid.x * grid.y:
        pytest.skip(f"needs {num_cores} clusters, device has {grid.x * grid.y}")
    spec = ttnn.ShardSpec(
        ttnn.num_cores_to_corerangeset(num_cores, grid, True), shard_shape, ttnn.ShardOrientation.ROW_MAJOR
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, spec)


# Height-sharded tensors keep whole rows / tiles per page, so the kernel reads and writes them directly.
@pytest.mark.parametrize(
    "shape, begins, ends, steps, layout, in_shard, out_shard",
    [
        pytest.param(
            (1, 1, 52, 64), (0, 0, 13, 0), (1, 1, 39, 64), (1, 1, 1, 1), RM, (4, (13, 64)), (2, (13, 64)), id="rm"
        ),
        pytest.param(
            (1, 1, 52, 64), (0, 0, 1, 3), (1, 1, 51, 61), (1, 1, 1, 1), RM, (4, (13, 64)), None, id="rm_to_l1"
        ),
        pytest.param(
            (1, 1, 48, 40),
            (0, 0, 2, 1),
            (1, 1, 46, 39),
            (1, 1, 2, 2),
            RM,
            None,
            (2, (11, 19)),
            id="rm_strided_from_dram",
        ),
        pytest.param(
            (1, 1, 128, 64), (0, 0, 32, 0), (1, 1, 96, 64), (1, 1, 1, 1), TILE, (4, (32, 64)), (2, (32, 64)), id="tile"
        ),
    ],
)
def test_slice_height_sharded(device, shape, begins, ends, steps, layout, in_shard, out_shard):
    _quasar_only(device)
    in_mc = _height_sharded(device, *in_shard) if in_shard else DRAM
    out_mc = _height_sharded(device, *out_shard) if out_shard else L1
    out = _run(device, shape, begins, ends, steps, layout, in_mc=in_mc, out_mc=out_mc)
    assert out.memory_config().memory_layout == out_mc.memory_layout


def _width_or_block_sharded(device, layout, grid_h, grid_w, shard_shape):
    grid = device.compute_with_storage_grid_size()
    if grid_w > grid.x or grid_h > grid.y:
        pytest.skip(f"needs a {grid_w}x{grid_h} compute grid, device has {grid.x}x{grid.y}")
    spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_w - 1, grid_h - 1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(layout, ttnn.BufferType.L1, spec)


# A TILE page is a whole tile under every sharding, so width / block sharded TILE tensors are read and
# written directly; a ROW_MAJOR row is split over the shards and goes through an interleaved tensor.
@pytest.mark.parametrize(
    "begins, ends, steps",
    [
        pytest.param((0, 0, 32, 64), (1, 1, 64, 192), (1, 1, 1, 1), id="whole_tiles"),
        pytest.param((0, 0, 3, 70), (1, 1, 60, 250), (1, 1, 1, 1), id="gather"),
    ],
)
@pytest.mark.parametrize(
    "mem_layout, grid_h, grid_w, shard_shape",
    [
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, 1, 4, (64, 64), id="width"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, 2, 4, (32, 64), id="block"),
    ],
)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_slice_width_and_block_sharded_input(
    device, layout, mem_layout, grid_h, grid_w, shard_shape, begins, ends, steps
):
    _quasar_only(device)
    in_mc = _width_or_block_sharded(device, mem_layout, grid_h, grid_w, shard_shape)
    _run(device, (1, 1, 64, 256), begins, ends, steps, layout, in_mc=in_mc, out_mc=L1)


@pytest.mark.parametrize(
    "mem_layout, grid_h, grid_w, shard_shape",
    [
        pytest.param(ttnn.TensorMemoryLayout.WIDTH_SHARDED, 1, 4, (32, 32), id="width"),
        pytest.param(ttnn.TensorMemoryLayout.BLOCK_SHARDED, 1, 4, (32, 32), id="block"),
    ],
)
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
def test_slice_width_and_block_sharded_output(device, layout, mem_layout, grid_h, grid_w, shard_shape):
    _quasar_only(device)
    out_mc = _width_or_block_sharded(device, mem_layout, grid_h, grid_w, shard_shape)
    out = _run(
        device, (1, 1, 64, 256), (0, 0, 32, 64), (1, 1, 64, 192), (1, 1, 1, 1), layout, in_mc=DRAM, out_mc=out_mc
    )
    assert out.memory_config().memory_layout == mem_layout
