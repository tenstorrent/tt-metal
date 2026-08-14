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

Lever arms: every `levers=dict(<knob>=0)` case is the measured OFF arm of an
applied lever, so `eval/verify_levers.py` can find the knob and the ledger row
can carry BOTH numbers. `stub=` arms are the classification ablation (payload
removed, synchronization kept).
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
    levers=None,
    ablate=None,
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
    levers = levers or {}
    ablate = ablate or {}
    saved = dict(pd.LEVERS)
    saved_ablate = dict(pd.ABLATE)
    pd.LEVERS.update(levers)
    pd.ABLATE.update(ablate)
    try:
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
    finally:
        pd.LEVERS.update(saved)
        pd.ABLATE.update(saved_ablate)

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


# --- lever OFF arms (counterfactuals) --------------------------------------
@pytest.mark.parametrize(
    "regime",
    [
        r
        if r != "b_wide_short"
        else pytest.param(
            r,
            marks=pytest.mark.xfail(
                reason="w_split OFF on the wide/short shape OOMs L1: the input CB would be "
                "WT=512 tiles (op_design.md §1.3 candidate 2). The counterfactual cannot even "
                "be built — that IS the measurement.",
                strict=True,
                raises=RuntimeError,
            ),
        )
        for r in SHAPES
    ],
)
def test_bench_lever_w_split_off(device, regime):
    """A0/A1 grid fill: pure height split. On (b) this collapses to ONE core."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(w_split=0), label=f"w_split=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_row_wise_off(device, regime):
    """master.md A1: split_work_to_cores column-wise (the binding default trap)."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(row_wise=0), label=f"row_wise=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_block_write_off(device, regime):
    """master.md B7: one write barrier per tile page instead of per block."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(block_write=0), label=f"block_write=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_double_buffer_off(device, regime):
    """master.md C16: depth-1 CBs — no read/compute/write overlap."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(double_buffer=0), label=f"double_buffer=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_multicore_off(device, regime):
    """A0 baseline: the whole op on one core."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(multicore=0), label=f"multicore=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_page_write_off(device, regime):
    """master.md B5: each tile page split into two half-page transactions."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(page_write=0), label=f"page_write=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_noc_split_off(device, regime):
    """master.md B9: reader/writer NoC assignment swapped."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(noc_split=0), label=f"noc_split=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_regime_select_off(device, regime):
    """master.md D20: no compile-time specialization — the pad reader on the aligned path."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(regime_select=0), label=f"regime_select=0/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_fp32_dest_off(device, regime):
    """master.md F25: fp32 DEST + lossless unpack (the exactness gate) turned off."""
    _measure(device, SHAPES[regime], ttnn.float32, levers=dict(fp32_dest=0), label=f"fp32_dest=0/{regime}")


# --- sharded placement (Refinement 1; op_design.md §9.4 case (e)) ----------
# NOTE the RE-TARGET: a local-shard side is L1 loopback, not DRAM, so the
# DRAM-floor target does not describe these rows. The `zero_copy=0` arm is the
# measured OFF arm of master.md C14 (+ A2): it takes the TensorAccessor path over
# the very shard that is already resident, which is the "tolerated, not
# implemented" sharding an aliased CB replaces.
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
def test_bench_lever_zero_copy_off(device, regime):
    """master.md C14 (+A2) OFF: re-read/re-write the resident shard over the NoC."""
    shape, num_cores = SHARDED_SHAPES[regime]
    cfg = _height_shard(shape, num_cores)
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=cfg,
        out_mem_config=cfg,
        levers=dict(zero_copy=0),
        label=f"zero_copy=0/{regime}",
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


@pytest.mark.parametrize("regime", list(SHARDED_SHAPES))
@pytest.mark.parametrize(
    "ablate, name",
    [({"compute": 1}, "no_compute"), ({"compute": 1, "dm": 1}, "sync_only")],
    ids=["no_compute", "sync_only"],
)
def test_bench_sharded_ablation(device, regime, ablate, name):
    """Classification for the zero-copy path: with NO NoC traffic on either side,
    the only payload left is the tilize LLK, so this is what re-targets the
    sharded rows away from the DRAM floor."""
    shape, num_cores = SHARDED_SHAPES[regime]
    cfg = _height_shard(shape, num_cores)
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=cfg,
        out_mem_config=cfg,
        ablate=ablate,
        label=f"ablate:{name}/{regime}",
    )


# --- classification ablation (op_design.md §9.1) ---------------------------
# Payload removed, synchronization intact. Output is WRONG by design — this is a
# measurement, never a correctness check. Peel stages CUMULATIVELY: stages
# overlap, so a single removal is only a lower bound on that stage's cost.
@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize(
    "ablate, name",
    [
        ({"compute": 1}, "no_compute"),
        ({"dm": 1}, "no_dm"),
        ({"compute": 1, "dm": 1}, "sync_only"),
    ],
    ids=["no_compute", "no_dm", "sync_only"],
)
def test_bench_ablation(device, regime, ablate, name):
    _measure(device, SHAPES[regime], ttnn.bfloat16, ablate=ablate, label=f"ablate:{name}/{regime}")


# --- cross-spec reshard + padded-into-a-shard (Refinement 2) ----------------
# Two NEW rows for the cumulative set, and both are measured against the SAME
# `zero_copy` knob whose OFF arm is literally the pre-Refinement-2 behaviour:
#
#   (f) cross-spec reshard — a WIDTH-sharded RM source (page = SHARD row, so the
#       gather splits every row span across pages) into a HEIGHT-sharded TILE
#       destination that is packed in place. L1 -> L1, no DRAM leg at all.
#       zero_copy=0 sends both sides through the accessor over the generic grid
#       split (master.md C14 + C15 + A2 OFF on the reshard path).
#   (g) padded into a local shard — the fill is materialized into the streaming
#       input CB while the destination shard is still written in place. Before
#       Refinement 2 a padded call disqualified BOTH sides from zero-copy, which
#       is exactly what zero_copy=0 reproduces.
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
# Same reshard with a 4-core (128 B page) source — below the knee, which is what
# the `xfer_gate` lever exists to catch.
GATED_RESHARD_SHAPE = ([1, 1, 1024, 256], 4, 8)
# (logical shape, pad target, destination cores)
PAD_SHARD_SHAPE = ([1, 1, 2040, 256], [1, 1, 2048, 256], 8)


@pytest.mark.parametrize("zero_copy", [1, 0], ids=["on", "off"])
def test_bench_reshard_cross_spec(device, zero_copy):
    shape, src_cores, dst_cores = RESHARD_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=_width_shard(shape, src_cores),
        out_mem_config=_height_shard(shape, dst_cores),
        levers=dict(zero_copy=zero_copy),
        label=f"reshard_cross_spec/zero_copy={zero_copy}",
    )


@pytest.mark.parametrize("xfer_gate", [1, 0], ids=["on", "off"])
def test_bench_lever_xfer_gate(device, xfer_gate):
    """The read-transfer gate: OFF aliases a destination whose shard width pins
    the reader to 128 B transfers on 8 cores instead of a coarse WT_CHUNK on 64."""
    shape, src_cores, dst_cores = GATED_RESHARD_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=_width_shard(shape, src_cores),
        out_mem_config=_height_shard(shape, dst_cores),
        levers=dict(xfer_gate=xfer_gate),
        label=f"gated_reshard/xfer_gate={xfer_gate}",
    )


@pytest.mark.parametrize("xfer_gate", [1, 0], ids=["on", "off"])
def test_bench_lever_xfer_gate_narrow_destination(device, xfer_gate):
    """The worst case for an aliased destination: a ONE-tile-wide shard (64 B
    reads) — master.md B5 territory, and the largest measured swing."""
    shape = [1, 1, 1024, 256]
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=_width_shard(shape, 8),
        levers=dict(xfer_gate=xfer_gate),
        label=f"narrow_dest_shard/xfer_gate={xfer_gate}",
    )


@pytest.mark.parametrize(
    "ablate, name",
    [({"compute": 1}, "no_compute"), ({"compute": 1, "dm": 1}, "sync_only")],
    ids=["no_compute", "sync_only"],
)
def test_bench_reshard_ablation(device, ablate, name):
    """Classify the gather: one side is L1 loopback, the other a cross-core L1
    read, so neither the DRAM floor nor the pure-compute bound describes it."""
    shape, src_cores, dst_cores = RESHARD_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=_width_shard(shape, src_cores),
        out_mem_config=_height_shard(shape, dst_cores),
        ablate=ablate,
        label=f"ablate:{name}/reshard_cross_spec",
    )


@pytest.mark.parametrize("zero_copy", [1, 0], ids=["on", "off"])
def test_bench_padded_into_local_shard(device, zero_copy):
    shape, target, dst_cores = PAD_SHARD_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=_height_shard(target, dst_cores),
        pad=dict(output_padded_shape=target, pad_value=0.0),
        levers=dict(zero_copy=zero_copy),
        label=f"padded_into_local_shard/zero_copy={zero_copy}",
    )


# --- Refinement 3: the DM levers on the interleaved aligned path -------------
# Ceiling re-target (measured on this box with the `dram_saturation` example, a
# pure DRAM->DRAM copy of the SAME tensor with no compute kernel at all):
#   (a) 2048x2048 bf16  87,710 ns @64 cores (191.3 GB/s) / 86,943 @32 (193.0)
#   (b) 32x16384  bf16  12,078 ns @64 cores (173.6 GB/s) / 11,550 @32 (181.6)
#   (c) 8192x1024 bf16 174,772 ns @64 cores (192.0 GB/s)
# An interleaved DRAM->DRAM stream saturates at ~192 GB/s on this box, NOT at the
# 288 GB/s datasheet peak — that is the number the achieved ratio is measured
# against from here on.


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_pipeline_off(device, regime, dtype_name):
    """Refinement 3 / master.md C16+B8 enabler: OFF = the Phase-0 blocking rule,
    which lands ONE block per core on a grid-filling shape so read / compute /
    write cannot overlap."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(pipeline=0),
        label=f"pipeline=0/{regime}/{dtype_name}",
    )


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


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_read_trid_off(device, regime, dtype_name):
    """master.md B8 OFF: one plain read barrier per block, so the NoC drains at
    every block boundary instead of block i+1's reads already being in flight.
    Only measurable where a core owns >1 block — (a) (4/core after the pipeline
    knob) and (c) (4/core) — and B0-checked on the smallest regime (d)."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(read_trid=0),
        label=f"read_trid=0/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_read_vc_on(device, regime, dtype_name):
    """master.md B10 ON arm: spread read requests over NUM_READ_VCS unicast VCs.
    B10 ships PARKED OFF (see LEVERS) — this is the arm that measured the loss."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(read_vc=1),
        label=f"read_vc=1/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_read_one_packet_off(device, regime, dtype_name):
    """master.md B6 OFF: the any-length issue path even when the transfer fits
    NOC_MAX_BURST_SIZE (512 B) — (b)'s read is exactly 512 B."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(read_one_packet=0),
        label=f"read_one_packet=0/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_custom_reader_off(device, regime, dtype_name):
    """All three read levers off => the reader compiles to the library-helper
    call verbatim. This is the OFF arm of the helper SUBSTITUTION itself."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(read_trid=0, read_vc=0, read_one_packet=0),  # -> library helper
        label=f"library_reader/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("regime", ["a_square", "b_wide_short", "c_multiblock"])
@pytest.mark.parametrize(
    "ablate, name",
    [({"dm_read": 1}, "no_reads"), ({"dm_write": 1}, "no_writes")],
    ids=["no_reads", "no_writes"],
)
def test_bench_ablation_dm_halves(device, regime, ablate, name):
    """Which NoC HALF binds? Reader and writer are one pipeline, so a reader-side
    lever only has a writer twin worth building if the write half is on the
    critical path. Payload removed one half at a time, all synchronization kept."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, ablate=ablate, label=f"ablate:{name}/{regime}")


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_write_trid_off(device, regime, dtype_name):
    """master.md B8, WRITE side — the reader lever's twin. OFF = one plain write
    barrier per block, so the write NoC drains at every block boundary. The
    split-DM ablation put the WRITE half on the critical path (it is the slower
    of the two halves on every real-work regime), which is why this twin exists."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(write_trid=0),
        label=f"write_trid=0/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_both_trid_off(device, regime, dtype_name):
    """Both halves of B8 off together — reader and writer are one pipeline, so
    the pair is measured together as well as separately."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(read_trid=0, write_trid=0),
        label=f"both_trid=0/{regime}/{dtype_name}",
    )


_PRE_R3 = dict(pipeline=0, read_trid=0, read_vc=0, read_one_packet=0, write_trid=0)


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_pre_refinement_3(device, regime, dtype_name):
    """Every Refinement-3 lever off at once = the Refinement-2 interleaved path
    exactly (Phase-0 blocking + the library reader + barrier-per-block writes).
    The whole-refinement counterfactual, and the non-regression control for the
    wide/short row whose own run-to-run spread exceeds the +-3% band."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(_PRE_R3),
        label=f"pre_r3/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("pipeline", [1, 0], ids=["on", "off"])
@pytest.mark.parametrize(
    "ablate, name",
    [({}, "full"), ({"dm_read": 1}, "no_reads"), ({"dm_write": 1}, "no_writes")],
    ids=["full", "no_reads", "no_writes"],
)
def test_bench_pipeline_overlap_mechanism(device, pipeline, ablate, name):
    """WHY the pipeline knob wins. C16 (per-core depth-2 CB) is flat with or
    without it, so the win is not per-core double-buffering. Hypothesis: with one
    block per core every core reads, then computes, then writes IN LOCKSTEP, so
    the read and write NoCs are busy in disjoint phases; finer blocks desynchronize
    the cores and the two DRAM directions overlap. Measure it: compare
    (read half + write half) against the wall in each arm — the bigger the
    shortfall, the more overlap."""
    _measure(
        device,
        SHAPES["a_square"],
        ttnn.bfloat16,
        levers=dict(pipeline=pipeline),
        ablate=ablate,
        label=f"overlap/pipeline={pipeline}/{name}",
    )


# --- Refinement 4: the integer dtype family, and the two numeric-format knobs --
# The dtype family is a WIDTH change on the same data path, so its bench rows exist
# to (a) give later phases a non-regression baseline on 1-byte and 4-byte integer
# datums, and (b) price the two knobs this refinement turned: fp32 DEST for 8-bit
# datums (a correctness requirement, see test_uint8_requires_fp32_dest) and the
# OUTPUT-format pad stamp.

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


def _measure_widening_pad(device, levers, label):
    """The worst-case geometry for the OUTPUT-format pad stamp: half the output
    tile-rows are WHOLE pad tiles, so every element of them is stored individually."""
    shape, target = _OUT_FILL_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        out_dtype=ttnn.float32,
        pad=dict(output_padded_shape=target, pad_value=10.2),
        levers=levers,
        label=label,
    )


# The two arms spell their knob value as a LITERAL rather than sharing one
# parametrized body: `eval/verify_levers.py` scans the bench for
# `levers=dict(<knob>=<int>)` to prove the counterfactual is re-runnable, and a
# forwarded parameter is invisible to it.
def test_bench_lever_out_fill_on(device):
    """Refinement 4: the writer's OUTPUT-format pad stamp, shipped state."""
    _measure_widening_pad(device, dict(out_fill=1), "out_fill=1/widening_pad")


def test_bench_lever_out_fill_off(device):
    """OFF is the pre-Refinement-4 behaviour (the reader's input-format fill only),
    which on a WIDENING cast with an input-inexact fill is numerically WRONG — so
    this arm prices a correctness fix, it is not a choice of setting."""
    _measure_widening_pad(device, dict(out_fill=0), "out_fill=0/widening_pad")


@pytest.mark.parametrize("regime", ["a_square", "d_smallest"])
def test_bench_lever_pack_fast_on(device, regime):
    """master.md F24 (`bfp8_pack_precise`), shipped state: the FAST (truncating)
    block-float packer. bfloat8_b is the only output format the knob touches, so the
    arm requests it explicitly. `d_smallest` is the B0 per-core-overhead check."""
    _measure(
        device,
        SHAPES[regime],
        ttnn.bfloat16,
        out_dtype=ttnn.bfloat8_b,
        levers=dict(pack_fast=1),
        label=f"pack_fast=1/{regime}",
    )


@pytest.mark.parametrize("regime", ["a_square", "d_smallest"])
def test_bench_lever_pack_fast_off(device, regime):
    """F24 OFF: `bfp8_pack_precise=True`, the precise packer — it rounds instead of
    truncating and costs an extra pack pass."""
    _measure(
        device,
        SHAPES[regime],
        ttnn.bfloat16,
        out_dtype=ttnn.bfloat8_b,
        levers=dict(pack_fast=0),
        label=f"pack_fast=0/{regime}",
    )


@pytest.mark.parametrize(
    "ablate, name",
    [({}, "full"), ({"compute": 1}, "no_compute"), ({"dm": 1}, "no_dm"), ({"compute": 1, "dm": 1}, "sync_only")],
    ids=["full", "no_compute", "no_dm", "sync_only"],
)
def test_bench_widening_pad_ablation(device, ablate, name):
    """Classify the ONE path Refinement 4 adds. Every other path keeps Phase 0's
    DM-bound verdict (an element width does not change what binds), but the
    OUTPUT-format pad stamp is neither compute nor NoC: it is rv32 volatile L1
    stores on the WRITER, so both payload ablations must leave it standing.

    Measured: full 385,227 -> no_compute 393,690 -> no_dm 355,588 -> ALL payloads
    stubbed at once 360,882 ns. 94% of the wall survives with nothing but the CB
    handshake and the stamp left, so this path is STAMP-bound."""
    _measure(
        device,
        _OUT_FILL_SHAPE[0],
        ttnn.bfloat16,
        out_dtype=ttnn.float32,
        pad=dict(output_padded_shape=_OUT_FILL_SHAPE[1], pad_value=10.2),
        ablate=ablate,
        label=f"ablate:{name}/widening_pad",
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


# --- Refinement 6: the completeness audit's own levers ------------------------
# master.md B13 (`read_state` / `write_state`) and D21 (`precomp_index`). All
# three ship PARKED at 0 (their OFF arm is the Refinement-5 kernel byte for
# byte), so every test below that turns one to 1 is the lever's ON arm and the
# matching `=0` test is the shipped baseline. B13 is a per-transaction issue
# lever, so it is priced across TRANSACTION SIZE, not at one point:
#
#   read  — (a) 1024 B per source row, (b) 512 B, (d) 128 B, and the cross-core
#           gather's 256 B page slices (the ONLY path where consecutive
#           transfers share a NoC endpoint, which is the whole premise).
#   write — the output TILE page, which is a TILE-HEIGHT question: 512 B at
#           tile_h=8, 256 B at 4, 64 B at 1. Above 512 B there is no
#           any-length stateful write in the API at all, so the lever is
#           compile-time inert on the default 32-row geometry.
#
# The write arms pin `write_trid=0` on BOTH sides: B8's write-side double-issue
# owns the same command buffer, so leaving it on would measure the pair rather
# than the lever.


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_read_state_on(device, regime, dtype_name):
    """master.md B13 READ side ON (set_state/with_state, endpoint cached)."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(read_state=1),
        label=f"read_state=1/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("regime", list(SHAPES))
@pytest.mark.parametrize("dtype_name", list(_DTYPES))
def test_bench_lever_read_state_off(device, regime, dtype_name):
    """The shipped arm: one plain noc_async_read per source row."""
    _measure(
        device,
        SHAPES[regime],
        _DTYPES[dtype_name],
        levers=dict(read_state=0),
        label=f"read_state=0/{regime}/{dtype_name}",
    )


@pytest.mark.parametrize("read_state", [1, 0], ids=["on", "off"])
def test_bench_lever_read_state_gather(device, read_state):
    """B13 on the ONE path whose consecutive transfers share a source core: the
    cross-core L1 gather (a source shard lives on one core, and a block's TILE_H
    rows all come out of it). If the lever pays anywhere on this op, here."""
    shape, src_cores, dst_cores = RESHARD_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=_width_shard(shape, src_cores),
        out_mem_config=_height_shard(shape, dst_cores),
        levers=dict(read_state=read_state),
        label=f"read_state={read_state}/reshard_gather",
    )


@pytest.mark.parametrize("tile_height", [8, 4, 1])
def test_bench_lever_write_state_on(device, tile_height):
    """master.md B13 WRITE side ON, swept across the output page size."""
    _measure(
        device,
        SHAPES["a_square"],
        ttnn.bfloat16,
        tile_h=tile_height,
        levers=dict(write_state=1, write_trid=0),
        label=f"write_state=1/tile_h={tile_height}",
    )


@pytest.mark.parametrize("tile_height", [8, 4, 1])
def test_bench_lever_write_state_off(device, tile_height):
    """The shipped arm at the same geometry (B8's write twin off on both sides)."""
    _measure(
        device,
        SHAPES["a_square"],
        ttnn.bfloat16,
        tile_h=tile_height,
        levers=dict(write_state=0, write_trid=0),
        label=f"write_state=0/tile_h={tile_height}",
    )


@pytest.mark.parametrize("tile_height", [8, 1])
def test_bench_lever_write_state_smallest_on(device, tile_height):
    """master.md B0: the per-core-overhead regime, where a fixed per-transfer
    setup cost is most likely to show as a LOSS."""
    _measure(
        device,
        SHAPES["d_smallest"],
        ttnn.bfloat16,
        tile_h=tile_height,
        levers=dict(write_state=1, write_trid=0),
        label=f"write_state=1/d_smallest/tile_h={tile_height}",
    )


@pytest.mark.parametrize("tile_height", [8, 1])
def test_bench_lever_write_state_smallest_off(device, tile_height):
    """The B0 check's OFF arm."""
    _measure(
        device,
        SHAPES["d_smallest"],
        ttnn.bfloat16,
        tile_h=tile_height,
        levers=dict(write_state=0, write_trid=0),
        label=f"write_state=0/d_smallest/tile_h={tile_height}",
    )


@pytest.mark.parametrize("regime", ["d_smallest"])
@pytest.mark.parametrize("rep", range(3))
def test_bench_lever_read_state_smallest_repeat(device, regime, rep):
    """master.md B0 for the READ side, repeated: the smallest regime is where a
    per-transfer setup cost dominates, and it is the row whose verdict the
    parked default rests on, so it is measured more than once."""
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(read_state=1), label=f"read_state=1/{regime}/r{rep}")
    _measure(device, SHAPES[regime], ttnn.bfloat16, levers=dict(read_state=0), label=f"read_state=0/{regime}/r{rep}")


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_precomp_index_on(device, regime):
    """master.md D21 ON: the (tile-row, W chunk) origin comes from the host and
    the loop steps it, instead of a div/mod per block."""
    _measure(
        device,
        SHAPES[regime],
        ttnn.bfloat16,
        levers=dict(precomp_index=1),
        label=f"precomp_index=1/{regime}",
    )


@pytest.mark.parametrize("regime", list(SHAPES))
def test_bench_lever_precomp_index_off(device, regime):
    """The shipped arm: the kernel recomputes b % nt_h / b / nt_h per block."""
    _measure(
        device,
        SHAPES[regime],
        ttnn.bfloat16,
        levers=dict(precomp_index=0),
        label=f"precomp_index=0/{regime}",
    )


# --- Refinement 6: pricing the ONE unbuilt lever (design lamp L4) -------------
# `split_reader` on a destination-local plan. The writer issues NO NoC traffic
# there, so BRISC is idle and could take half the reads — the recorded ~1.7x
# candidate. It is a scheme change (a second input CB, so the CB keeps one
# producer each), not a knob, so this phase PRICES it instead of building it:
# the split-DM ablation says how much of that wall is the read half, which is
# the ceiling on what moving half of it to a second RISC could buy.
@pytest.mark.parametrize(
    "ablate,name",
    [({}, "full"), ({"dm_read": 1}, "no_read"), ({"dm_write": 1}, "no_write"), ({"compute": 1}, "no_compute")],
)
def test_bench_l4_split_reader_headroom(device, ablate, name):
    shape, src_cores, dst_cores = RESHARD_SHAPE
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        in_mem_config=_width_shard(shape, src_cores),
        out_mem_config=_height_shard(shape, dst_cores),
        ablate=ablate,
        label=f"l4_headroom/{name}",
    )


@pytest.mark.parametrize(
    "ablate,name",
    [({}, "full"), ({"dm_read": 1}, "no_read"), ({"dm_write": 1}, "no_write")],
)
def test_bench_l4_split_reader_headroom_crossover(device, ablate, name):
    """The same question on the DRAM -> local-shard crossover, where the read leg
    is a DRAM read rather than an L1 gather."""
    shape = SHARDED_SHAPES["e_shard_same_wide"][0]
    cores = SHARDED_SHAPES["e_shard_same_wide"][1]
    _measure(
        device,
        shape,
        ttnn.bfloat16,
        out_mem_config=_height_shard(shape, cores),
        ablate=ablate,
        label=f"l4_headroom_crossover/{name}",
    )
