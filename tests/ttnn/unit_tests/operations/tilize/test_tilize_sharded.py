# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 — sharded I/O (same-spec, zero-copy) tests for tilize.

The supported sharded path is same-spec, zero-copy: an RM-sharded L1 input is
tilized straight into a TILE-sharded L1 output on the IDENTICAL shard spec, with
both circular buffers aliased onto the local L1 shard buffers (no reader, no
writer, no DRAM/NoC). tilize is a pure layout op, so the oracle is identity:
`to_torch(tilize(from_torch(x, ROW_MAJOR, mc), mc)) == x` (exact for the
value-preserving dtypes here).

These tests exercise the four schemes (HEIGHT / WIDTH / BLOCK legacy + nd) that
Refinement 2 adds, plus the clean-refusal contract for the out-of-scope cases
(single-core+sharded, interleaved<->sharded crossover, cross-spec resharding,
multi-shard-per-core) — Refinement 2b. The device is opened once per module via
the dir conftest (@pytest.mark.use_module_device); request `device` by name.
"""

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import ExcludedCell, UnsupportedAxisValue
from ttnn.operations.tilize import tilize


def _crs(*ranges):
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(*s), ttnn.CoreCoord(*e)) for (s, e) in ranges})


_HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
_WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
_BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED
_ROW = ttnn.ShardOrientation.ROW_MAJOR
_COL = ttnn.ShardOrientation.COL_MAJOR
_L1 = ttnn.BufferType.L1


def _legacy_mc(scheme, grid, shard_shape, orient):
    return ttnn.MemoryConfig(scheme, _L1, ttnn.ShardSpec(grid, shard_shape, orient))


def _nd_mc(grid, shard_shape, orient):
    return ttnn.MemoryConfig(_L1, ttnn.NdShardSpec(ttnn.Shape(shard_shape), grid, orient))


def _make_torch(dtype, shape):
    if dtype in (ttnn.uint32, ttnn.uint16):
        return torch.randint(0, 100, shape, dtype=torch.int32)
    if dtype == ttnn.int32:
        return torch.randint(-1000, 1000, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.randn(shape, dtype=torch.float32)
    return torch.randn(shape).bfloat16()


def _run_identity(device, shape, mc, dtype):
    torch_in = _make_torch(dtype, shape)
    tt_in = ttnn.from_torch(torch_in, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    tt_out = tilize(tt_in, memory_config=mc)
    assert tt_out.layout == ttnn.TILE_LAYOUT
    out = ttnn.to_torch(tt_out)
    # Integers may read back as an unsigned torch dtype; compare in int64 to
    # sidestep torch's int/uint promotion rules while keeping exact identity.
    if dtype in (ttnn.uint32, ttnn.uint16, ttnn.int32):
        assert torch.equal(torch_in.to(torch.int64), out.to(torch.int64)), "integer identity mismatch"
    else:
        assert torch.equal(torch_in, out), f"identity mismatch: max_diff={(torch_in.float()-out.float()).abs().max()}"


# ---------------------------------------------------------------------------
# Same-spec zero-copy — the four schemes Refinement 2 adds (bf16 identity).
# ---------------------------------------------------------------------------
_SCHEME_CASES = [
    pytest.param([1, 1, 512, 64], _legacy_mc(_HEIGHT, _crs(((0, 0), (3, 0))), (128, 64), _ROW), id="height-4c"),
    pytest.param([1, 1, 64, 512], _legacy_mc(_WIDTH, _crs(((0, 0), (3, 0))), (64, 128), _ROW), id="width-4c"),
    pytest.param([1, 1, 128, 128], _legacy_mc(_BLOCK, _crs(((0, 0), (1, 1))), (64, 64), _ROW), id="block-row-2x2"),
    pytest.param([1, 1, 128, 128], _legacy_mc(_BLOCK, _crs(((0, 0), (1, 1))), (64, 64), _COL), id="block-col-2x2"),
    pytest.param([1, 1, 128, 128], _nd_mc(_crs(((0, 0), (1, 1))), (1, 1, 64, 64), _ROW), id="nd-r4-2x2"),
    pytest.param([4, 32, 64], _nd_mc(_crs(((0, 0), (1, 0))), (2, 32, 64), _ROW), id="nd-r3-batchfold"),
    pytest.param([2, 64, 64], _nd_mc(_crs(((0, 0), (1, 0))), (1, 64, 64), _ROW), id="nd-r3-simple"),
]


@pytest.mark.parametrize("shape,mc", _SCHEME_CASES)
def test_tilize_sharded_identity_bf16(device, shape, mc):
    _run_identity(device, shape, mc, ttnn.bfloat16)


# fp32 + integer passthrough on the sharded path (value-preserving dtypes).
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.uint32, ttnn.uint16, ttnn.int32])
def test_tilize_sharded_identity_dtypes(device, dtype):
    mc = _legacy_mc(_HEIGHT, _crs(((0, 0), (3, 0))), (128, 64), _ROW)
    _run_identity(device, [1, 1, 512, 64], mc, dtype)


# A larger width-sharded shard (multi-tile-wide shard).
def test_tilize_sharded_wide_width(device):
    mc = _legacy_mc(_WIDTH, _crs(((0, 0), (7, 0))), (32, 256), _ROW)
    _run_identity(device, [1, 1, 32, 2048], mc, ttnn.bfloat16)


# ---------------------------------------------------------------------------
# Refinement 2b — interleaved <-> sharded crossover (split reader / writer).
# HEIGHT-sharded, ROW_MAJOR, one-shard-per-core maps each shard to a CONTIGUOUS
# global tile-row range; the split reader/writer reuse the interleaved kernels.
# ---------------------------------------------------------------------------
def _run_crossover(device, shape, sharded_mc, interleaved_mc, in_sharded, dtype=ttnn.bfloat16):
    torch_in = _make_torch(dtype, shape)
    in_mc = sharded_mc if in_sharded else interleaved_mc
    out_mc = interleaved_mc if in_sharded else sharded_mc
    tt_in = ttnn.from_torch(torch_in, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=in_mc)
    tt_out = tilize(tt_in, memory_config=out_mc)
    assert tt_out.layout == ttnn.TILE_LAYOUT
    out = ttnn.to_torch(tt_out)
    if dtype in (ttnn.uint32, ttnn.uint16, ttnn.int32):
        assert torch.equal(torch_in.to(torch.int64), out.to(torch.int64)), "crossover integer identity mismatch"
    else:
        assert torch.equal(
            torch_in, out
        ), f"crossover identity mismatch: max_diff={(torch_in.float()-out.float()).abs().max()}"


_CROSSOVER_CASES = [
    # (shape, HEIGHT shard shape, grid_end) — 1 shard/core
    pytest.param([1, 1, 512, 64], (128, 64), (3, 0), id="512x64-4c"),
    pytest.param([1, 1, 256, 128], (64, 128), (3, 0), id="256x128-4c"),
    pytest.param([2, 128, 96], (128, 96), (1, 0), id="r3-2c"),
]


@pytest.mark.parametrize("shape,shard_shape,grid_end", _CROSSOVER_CASES)
def test_tilize_crossover_interleaved_to_sharded(device, shape, shard_shape, grid_end):
    """DRAM interleaved RM input -> HEIGHT-sharded TILE output (split reader)."""
    mc = _legacy_mc(_HEIGHT, _crs(((0, 0), grid_end)), shard_shape, _ROW)
    _run_crossover(device, shape, mc, ttnn.DRAM_MEMORY_CONFIG, in_sharded=False)


@pytest.mark.parametrize("shape,shard_shape,grid_end", _CROSSOVER_CASES)
def test_tilize_crossover_sharded_to_interleaved(device, shape, shard_shape, grid_end):
    """HEIGHT-sharded RM input -> DRAM interleaved TILE output (split writer)."""
    mc = _legacy_mc(_HEIGHT, _crs(((0, 0), grid_end)), shard_shape, _ROW)
    _run_crossover(device, shape, mc, ttnn.DRAM_MEMORY_CONFIG, in_sharded=True)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.uint32])
def test_tilize_crossover_dtypes(device, dtype):
    """Crossover works for the value-preserving dtypes too."""
    mc = _legacy_mc(_HEIGHT, _crs(((0, 0), (3, 0))), (128, 64), _ROW)
    _run_crossover(device, [1, 1, 512, 64], mc, ttnn.DRAM_MEMORY_CONFIG, in_sharded=False, dtype=dtype)
    _run_crossover(device, [1, 1, 512, 64], mc, ttnn.DRAM_MEMORY_CONFIG, in_sharded=True, dtype=dtype)


# ---------------------------------------------------------------------------
# Refinement 2b — multi-shard-per-core, same-spec, even, no-padding (nd only).
# A core owns k = n_shards/n_cores contiguous full shards, tilized as one bank.
# (Legacy HEIGHT/WIDTH/BLOCK cannot hold >1 shard/core — from_torch rejects it.)
# ---------------------------------------------------------------------------
_MULTISHARD_CASES = [
    pytest.param([4, 128, 128], (2, 64, 64), (1, 1), id="nd-8sh-4c"),  # 8 shards / 4 cores
    pytest.param([1, 1, 512, 64], (1, 1, 128, 64), (1, 0), id="nd-4sh-2c"),  # 4 shards / 2 cores
]


@pytest.mark.parametrize("shape,shard_shape,grid_end", _MULTISHARD_CASES)
def test_tilize_multishard_same_spec_nd(device, shape, shard_shape, grid_end):
    mc = _nd_mc(_crs(((0, 0), grid_end)), shard_shape, _ROW)
    _run_identity(device, shape, mc, ttnn.bfloat16)


# ---------------------------------------------------------------------------
# Clean-refusal contract for the cases STILL deferred (Refinement 2c).
# These MUST refuse (not hang, not produce wrong output).
# ---------------------------------------------------------------------------
def test_sharded_single_core_excluded(device, expect_error):
    """Sharding is inherently multi-core -> single-core+sharded is an ExcludedCell."""
    mc = _legacy_mc(_HEIGHT, _crs(((0, 0), (3, 0))), (128, 64), _ROW)
    t = ttnn.from_torch(
        torch.randn([1, 1, 512, 64]).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    with expect_error(ExcludedCell, "."):
        tilize(t, memory_config=mc, use_multicore=False)


def test_width_crossover_refused(device, expect_error):
    """WIDTH crossover needs column-chunked reads (Refinement 2c)."""
    out_mc = _legacy_mc(_WIDTH, _crs(((0, 0), (3, 0))), (64, 128), _ROW)
    t = ttnn.from_torch(
        torch.randn([1, 1, 64, 512]).bfloat16(),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with expect_error(UnsupportedAxisValue, "."):
        tilize(t, memory_config=out_mc)


def test_padded_multishard_same_spec_identity(device):
    """Refinement 2c: cliff/padded same-spec nd multi-shard (3 % 2 != 0,
    160 % 64 != 0) now tilizes the whole PHYSICAL bank in place — identity."""
    x = torch.rand([3, 160, 160], dtype=torch.bfloat16)
    mc = _nd_mc(_crs(((0, 0), (1, 1))), (2, 64, 64), _ROW)
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
    res = ttnn.to_torch(tilize(t, memory_config=mc))
    assert torch.equal(x, res)
