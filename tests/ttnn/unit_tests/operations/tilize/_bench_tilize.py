# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf-only bench for tilize — NOT part of the golden suite, NO PCC assert.

Underscore-prefixed so the correctness runs don't collect it. Measurement and
ablation need no correctness, and the golden INPUTS are deliberately tiny (they
cannot be bandwidth-bound, so they cannot measure what Track A optimizes).

    scripts/run_safe_pytest.sh --run-all \\
        tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py

Prints `DEVICE KERNEL DURATION [ns]` per case (in-process device profiler) plus
the achieved DRAM bandwidth (read + write = 2x tensor bytes).

Shape regimes (op_design.md §9.4):
  (a) grid-filling square   [1,1,2048,2048]  — per-core DRAM efficiency
  (b) wide/short  MANDATORY [1,1,32,16384]   — NT_H=1: does the split fill the grid?
  (c) multi-block-per-core  [1,1,8192,1024]  — the only regime where a
                                               next-block overlap lever can show
  (d) smallest regime       [1,1,32,64]      — per-core-overhead levers (master.md B0)

Every case is a cumulative NON-REGRESSION row: the shapes and plans any phase
made fast, re-measured together so a later change cannot quietly slow one of
them down. The per-lever counterfactual arms that produced `lever_ledger.json`'s
numbers were harness scaffolding and were removed with the harness itself; the
measurements they produced are recorded in the ledger.
"""

import os

# In-process on-device profiler — all three, before the device opens.
os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.tilize import tilize
from ttnn.operations.tilize import tilize_program_descriptor as pd


_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

SHAPES = {
    "a_square": [1, 1, 2048, 2048],
    "b_wide_short": [1, 1, 32, 16384],
    "c_multiblock": [1, 1, 8192, 1024],
    "d_smallest": [1, 1, 32, 64],
}

_DTYPES = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}


def _read_kernel_ns(device):
    """On-device kernel ns over the programs dispatched since the last read."""
    ttnn.ReadDeviceProfiler(device)
    per_chip = ttnn.get_latest_programs_perf_data()
    total, found = 0.0, False
    for programs in (per_chip or {}).values():
        for program in programs:
            results = getattr(program, "program_analyses_results", None) or {}
            entry = results.get(_DURATION_KEY)
            if entry is None:
                continue
            total += float(entry.duration)
            found = True
    return total if found else None


def _measure(
    device,
    shape,
    dtype,
    *,
    use_multicore=True,
    use_double_buffer=True,
    label="",
    in_mem_config=None,
    out_mem_config=None,
    pad=None,
    out_dtype=None,
    tile_h=None,
    in_tile_h=None,
):
    """One warm launch (compile + program cache), then ONE measured launch.

    Device kernel duration has no warm-up transient, so a trial loop would just
    re-measure the same number (see /perf-measure "Measurement discipline").
    """
    if dtype == ttnn.uint8:
        torch_input = torch.randint(0, 200, shape, dtype=torch.uint8)
    elif dtype in (ttnn.uint32, ttnn.uint16, ttnn.int32):
        torch_input = torch.randint(0, 100, shape, dtype=torch.int32)
    else:
        torch_input = torch.randn(shape).to(torch.bfloat16 if dtype == ttnn.bfloat16 else torch.float32)
    # `in_tile_h` set => the RETILE path: the SOURCE is already TILE layout at
    # that height (Refinement 5). Otherwise the input is ROW_MAJOR as always.
    source_kwargs = (
        dict(layout=ttnn.TILE_LAYOUT, tile=ttnn.Tile([in_tile_h, 32]))
        if in_tile_h is not None
        else dict(layout=ttnn.ROW_MAJOR_LAYOUT)
    )
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        device=device,
        memory_config=in_mem_config if in_mem_config is not None else ttnn.DRAM_MEMORY_CONFIG,
        **source_kwargs,
    )
    call = dict(memory_config=out_mem_config, use_multicore=use_multicore, use_double_buffer=use_double_buffer)
    if tile_h is not None:
        call["tile"] = ttnn.Tile([tile_h, 32])
    if out_dtype is not None:
        call["dtype"] = out_dtype
    call.update(pad or {})

    tilize(tt_input, **call)
    ttnn.synchronize_device(device)
    _read_kernel_ns(device)  # flush the warm-up window

    out = tilize(tt_input, **call)
    ttnn.synchronize_device(device)
    ns = _read_kernel_ns(device)

    elem = {ttnn.bfloat16: 2, ttnn.float32: 4, ttnn.uint8: 1, ttnn.uint16: 2}.get(dtype, 4)
    tensor_bytes = 1
    for d in shape:
        tensor_bytes *= d
    tensor_bytes *= elem
    gbps = (2 * tensor_bytes) / ns if ns else float("nan")
    logger.info(f"BENCH tilize {label} shape={shape} ns={ns} GB/s={gbps:.1f}")
    assert ns is not None, "profiler produced no data (profiler-enabled build?)"
    return ns


# --- baseline: every regime x dtype ---------------------------------------
@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_baseline(device, regime, dtype_name):
    _measure(device, SHAPES[regime], _DTYPES[dtype_name], label=f"baseline/{regime}/{dtype_name}")


# --- sharded placement (Refinement 1; op_design.md §9.4 case (e)) ----------
# NOTE the RE-TARGET: a local-shard side is L1 loopback, not DRAM, so the
# DRAM-floor target does not describe these rows.
def _height_shard(shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    shard_shape = (shape[-2] // num_cores, shape[-1])
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )


SHARDED_SHAPES = {
    # (e) small same-spec zero-copy case from the design's bench table
    "e_shard_same_small": ([1, 1, 512, 64], 4),
    # a bigger same-spec case: 8 cores, 8 blocks/core — where a per-block lever
    # could show on the sharded path at all
    "e_shard_same_wide": ([1, 1, 2048, 256], 8),
}


@pytest.mark.parametrize("regime", list(SHARDED_SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_sharded_same_spec(device, regime, dtype_name):
    """Zero-copy both sides: no NoC traffic on either side, L1 loopback only."""
    shape, num_cores = SHARDED_SHAPES[regime]
    cfg = _height_shard(shape, num_cores)
    _measure(
        device,
        shape,
        _DTYPES[dtype_name],
        in_mem_config=cfg,
        out_mem_config=cfg,
        label=f"baseline/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("regime", list(SHARDED_SHAPES))
def test_bench_sharded_crossover(device, regime):
    """DRAM interleaved in -> local shard out (the read half is still DRAM)."""
    shape, num_cores = SHARDED_SHAPES[regime]
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=ttnn.DRAM_MEMORY_CONFIG,
        out_mem_config=_height_shard(shape, num_cores),
        label=f"crossover/{regime}",
    )


# --- cross-spec reshard + padded-into-a-shard (Refinement 2) ----------------
# Two rows for the cumulative set:
#
#   (f) cross-spec reshard — a WIDTH-sharded RM source (page = SHARD row, so the
#       gather splits every row span across pages) into a HEIGHT-sharded TILE
#       destination that is packed in place. L1 -> L1, no DRAM leg at all.
#   (g) padded into a local shard — the fill is materialized into the streaming
#       input CB while the destination shard is still written in place.
def _width_shard(shape, num_cores):
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_cores - 1, 0))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (shape[-2], shape[-1] // num_cores), ttnn.ShardOrientation.ROW_MAJOR),
    )


# (shape, source cores, destination cores). The source is WIDTH-sharded on 2 cores,
# so its page is 128 elements = 256 B: narrow enough to need the page-split gather,
# wide enough to clear MIN_STREAM_READ_BYTES so the destination stays local.
RESHARD_SHAPE = ([1, 1, 1024, 256], 2, 8)
# (logical shape, pad target, destination cores)
PAD_SHARD_SHAPE = ([1, 1, 2040, 256], [1, 1, 2048, 256], 8)


def test_bench_reshard_cross_spec(device):
    shape, src_cores, dst_cores = RESHARD_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=_width_shard(shape, src_cores),
        out_mem_config=_height_shard(shape, dst_cores),
        label="reshard_cross_spec",
    )


def test_bench_padded_into_local_shard(device):
    shape, target, dst_cores = PAD_SHARD_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=_height_shard(target, dst_cores),
        pad=dict(output_padded_shape=target, pad_value=0.0),
        label="padded_into_local_shard",
    )


# --- Refinement 3: the interleaved aligned path ------------------------------
# Ceiling re-target (measured on this box with the `dram_saturation` example, a
# pure DRAM->DRAM copy of the SAME tensor with no compute kernel at all):
#   (a) 2048x2048 bf16  87,710 ns @64 cores (191.3 GB/s) / 86,943 @32 (193.0)
#   (b) 32x16384  bf16  12,078 ns @64 cores (173.6 GB/s) / 11,550 @32 (181.6)
#   (c) 8192x1024 bf16 174,772 ns @64 cores (192.0 GB/s)
# An interleaved DRAM->DRAM stream saturates at ~192 GB/s on this box, NOT at the
# 288 GB/s datasheet peak — that is the number the achieved ratio is measured
# against from here on.


@pytest.mark.parametrize("blocks_per_core", [1, 2, 4, 8, 16], ids=lambda v: f"bpc{v}")
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_sweep_pipeline_blocks(device, blocks_per_core, dtype_name, monkeypatch):
    """Sweep PIPELINE_BLOCKS_PER_CORE on the grid-filling square — the one regime
    the knob moves. More blocks = deeper overlap but a smaller read transfer."""
    monkeypatch.setattr(pd, "PIPELINE_BLOCKS_PER_CORE", blocks_per_core)
    _measure(
        device,
        SHAPES["a_square"],
        _DTYPES[dtype_name],
        label=f"sweep_bpc={blocks_per_core}/a_square/{dtype_name}",
    )


@pytest.mark.parametrize("min_read", [128, 256, 512, 1024], ids=lambda v: f"min{v}")
def test_bench_sweep_pipeline_min_read(device, min_read, monkeypatch):
    """Sweep MIN_PIPELINE_READ_BYTES on the wide/short shape — the regime the
    transfer-size cap is protecting (its 512 B read is already at the floor, so
    lowering the cap is the only way to give it a second block per core)."""
    monkeypatch.setattr(pd, "MIN_PIPELINE_READ_BYTES", min_read)
    _measure(
        device,
        SHAPES["b_wide_short"],
        ttnn.bfloat16,
        label=f"sweep_min_read={min_read}/b_wide_short/bf16",
    )


# --- Refinement 4: the integer dtype family and the padded widening cast -------
# The dtype family is a WIDTH change on the same data path, so its bench rows give
# later phases a non-regression baseline on 1-byte and 4-byte integer datums.

_R4_DTYPES = {"uint32": ttnn.uint32, "uint8": ttnn.uint8}


@pytest.mark.parametrize("regime", ["a_square", "d_smallest"])
@pytest.mark.parametrize("dtype_name", list(_R4_DTYPES))
def test_bench_dtype_family(device, regime, dtype_name):
    """Baseline for the integer dtype family (Refinement 4). uint8 additionally
    carries fp32 DEST, which is why its per-byte cost is worth recording."""
    _measure(device, SHAPES[regime], _R4_DTYPES[dtype_name], label=f"dtype/{regime}/{dtype_name}")


# Half the output tile-rows are WHOLE PAD tiles — the worst case for the writer's
# output-format stamp (every element of those tiles is stored individually).
_OUT_FILL_SHAPE = ([1, 1, 1024, 2048], [1, 1, 2048, 2048])


def test_bench_widening_pad(device):
    """The worst-case geometry for the writer's OUTPUT-format pad stamp: half the
    output tile-rows are WHOLE pad tiles."""
    shape, target = _OUT_FILL_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        out_dtype=ttnn.float32,
        pad=dict(output_padded_shape=target, pad_value=10.2),
        label="widening_pad",
    )


@pytest.mark.parametrize("regime", ["a_square", "d_smallest"])
def test_bench_bfloat8_output(device, regime):
    """The block-float output path (master.md F24 ships the FAST, truncating
    packer). `d_smallest` is the B0 per-core-overhead check."""
    _measure(
        device,
        SHAPES[regime],
        ttnn.bfloat16,
        out_dtype=ttnn.bfloat8_b,
        label=f"bfloat8_b/{regime}",
    )


# --- Refinement 5: tile geometry --------------------------------------------
# Two new data paths get cumulative rows here. TILE HEIGHT is a shape-dependent
# code path (it sets the CB page size, and through the L1 cap the W block factor
# and the block count), so it is benched across the RANGE of the axis rather than
# at one point. RETILE is a different reader entirely (whole-page staging + a
# local face permutation), so its cost is recorded separately from the row-major
# path it shares a compute kernel with.


@pytest.mark.parametrize("tile_height", [32, 16, 8, 1])
def test_bench_tile_height(device, tile_height):
    """Tiny tiles on the grid-filling square: same bytes, more/smaller pages."""
    _measure(device, SHAPES["a_square"], ttnn.bfloat16, tile_h=tile_height, label=f"tile_h={tile_height}/a_square")


@pytest.mark.parametrize("tile_height", [32, 8])
def test_bench_tile_height_smallest(device, tile_height):
    """master.md B0: the per-core-overhead regime, where a finer tile could only
    ever cost (there is not enough work to amortize anything)."""
    _measure(device, SHAPES["d_smallest"], ttnn.bfloat16, tile_h=tile_height, label=f"tile_h={tile_height}/d_smallest")


_RETILE_SHAPE = [1, 1, 1024, 1024]


@pytest.mark.parametrize("in_tile_h,tile_height", [(32, 8), (1, 32), (32, 16)], ids=["32to8", "1to32", "32to16"])
def test_bench_retile(device, in_tile_h, tile_height):
    """The retile path (R_RETILE). The face permutation is a CPU-side L1 copy, so
    this row is expected to sit WELL above the row-major path's DRAM-bound number —
    it is recorded as the baseline a later phase would have to beat, not as a
    claim that it is near any ceiling."""
    _measure(
        device,
        _RETILE_SHAPE,
        ttnn.bfloat16,
        tile_h=tile_height,
        in_tile_h=in_tile_h,
        label=f"retile/{in_tile_h}to{tile_height}",
    )
