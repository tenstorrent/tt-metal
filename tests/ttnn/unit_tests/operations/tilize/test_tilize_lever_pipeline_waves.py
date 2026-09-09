# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Lever measurement: `PIPELINE_WAVES_PER_CORE`.

DO NOT DELETE. This is the harness behind Refinement 3. A WAVE is one tile-row
of the block — the reader's push quantum, the compute helper's per-call unit and
the writer's minimum wait. A core whose whole assignment is ONE wave reads, then
computes, then writes strictly in series; with every core in lockstep the device
alternates a read-only phase with a write-only phase instead of sustaining both.

The tensor holds `R * num_w_chunks` tile-rows, so

    waves_per_core = R * num_w_chunks / num_cores

which makes the knob a pure multiplier on the column cut. On an `R == 1`
geometry (the `attention:` LOOSE_CASE `[1,1,32,16384]`) every extra wave has to
be bought from the column axis, HALVING the read transaction each doubling —
so the knob trades read transaction size for read/write overlap and the winner
is a measurement, not a derivation.

Run for numbers (device kernel ns per row, in the parametrize order):
    scripts/run_safe_pytest.sh --profile \
        tests/ttnn/unit_tests/operations/tilize/test_tilize_lever_pipeline_waves.py

Every variant is CORRECTNESS-checked with `torch.equal` — the knob changes the
block width, the block count and the CB capacity, and tilize does no arithmetic
so anything but bit identity is a bug.
"""

import pytest
import torch

import ttnn

import ttnn.operations.tilize.tilize_program_descriptor as pd
from ttnn.operations.tilize import tilize

ATTENTION_SHAPE = (1, 1, 32, 16384)  # R=1 C=512 — the flagged config
GUARD_SHAPES = [
    ((1, 1, 2048, 64), "grid2d_full_width"),
    ((1, 1, 2048, 2048), "square_large"),
    ((1, 1, 16384, 32), "tall_narrow"),
    # The two geometries whose w1 block is 1024 B — the read sizes between the
    # attention shape's 512 B and square_large's 4096 B, i.e. where the floor
    # has to be right or it trades a good read for a wave that cannot pay.
    ((1, 1, 1024, 1024), "square_mid"),
    ((1, 1, 32, 32768), "short_wide_wide"),
]

WAVES = [1, 2, 4, 8]


def _run(device, shape, waves, monkeypatch, label, floor=0):
    # `floor=0` disables MIN_BLOCK_ROW_BYTES so the sweep exercises the RAW wave
    # factor; production takes the deepest wave count clearing the real floor,
    # which is what `test_production_wave_choice` below pins.
    monkeypatch.setattr(pd, "PIPELINE_WAVES_PER_CORE", waves)
    monkeypatch.setattr(pd, "MIN_BLOCK_ROW_BYTES", floor)
    monkeypatch.setattr(pd, "_PLAN_CACHE", {})
    grid = device.compute_with_storage_grid_size()

    torch.manual_seed(11)
    torch_input = torch.randn(shape, dtype=torch.float32).bfloat16()
    tt_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_output = tilize(tt_input)
    plan = pd.derive_plan(tt_input, tt_output, low_l1=False, grid=grid)
    cores = len(plan.assignment)
    print(
        f"\n[lever PIPELINE_WAVES_PER_CORE={waves}] {label} {shape}: "
        f"bw={plan.block_width_tiles} chunks={plan.num_w_chunks} blocks={plan.num_blocks_total} "
        f"cores={cores}/{grid.x * grid.y} "
        f"waves/core={plan.tensor_row_blocks * plan.num_w_chunks / max(cores, 1):.2f} "
        f"read={plan.block_row_bytes}B L1/core={plan.l1_per_core_bytes // 1024}KiB"
    )
    assert torch.equal(ttnn.to_torch(tt_output), torch_input), f"{label}@{waves}: not bit-identical"
    return plan


@pytest.mark.parametrize("waves", WAVES, ids=[f"w{w}" for w in WAVES])
def test_pipeline_waves_attention(device, waves, monkeypatch):
    plan = _run(device, ATTENTION_SHAPE, waves, monkeypatch, "attention")
    # R == 1, so with the floor disabled the knob passes straight through to the
    # column cut and each doubling halves the read transaction.
    assert plan.num_w_chunks == min(512, 64 * waves)
    assert len(plan.assignment) == 64


@pytest.mark.parametrize("waves", WAVES, ids=[f"w{w}" for w in WAVES])
@pytest.mark.parametrize("shape,label", GUARD_SHAPES, ids=[g[1] for g in GUARD_SHAPES])
def test_pipeline_waves_guard(device, shape, label, waves, monkeypatch):
    _run(device, shape, waves, monkeypatch, label)


# --- the WRITER TWIN of the wave lever ------------------------------------
# Raising the wave count NARROWS the block, and the writer's in-flight
# transaction count is `write_rows_per_barrier * block_width_tiles` with
# `write_rows_per_barrier = ceil(WRITE_BATCH_MIN_TILES / block_width_tiles)` —
# so a wave doubling that takes bw from 8 to 4 also HALVES the writes in flight
# unless WRITE_BATCH_MIN_TILES moves with it. Reader and writer are one
# pipeline; this pins the pair rather than the reader lever alone.
TWIN_GRID = [(1, 4), (1, 8), (2, 4), (2, 8), (2, 16), (4, 8), (4, 16)]


@pytest.mark.parametrize("waves,wb", TWIN_GRID, ids=[f"w{w}_wb{b}" for w, b in TWIN_GRID])
def test_pipeline_waves_writer_twin(device, waves, wb, monkeypatch):
    monkeypatch.setattr(pd, "WRITE_BATCH_MIN_TILES", wb)
    plan = _run(device, ATTENTION_SHAPE, waves, monkeypatch, f"attention_wb{wb}")
    print(f"    writes in flight per barrier = " f"{plan.write_rows_per_barrier * plan.block_width_tiles}")


# --- what production actually picks ----------------------------------------
# The shipped rule is "deepest pipe (up to PIPELINE_WAVES_PER_CORE) whose read
# transaction still clears MIN_BLOCK_ROW_BYTES". These are the settings the
# sweeps above measured as the per-shape optimum; pinning them here is what
# stops a later plan edit from silently reverting the tuning.
PRODUCTION_CHOICE = [
    ((1, 1, 32, 16384), 8, 64, "attention: the 2nd wave would cost a 256B read, which does not pay"),
    ((1, 1, 2048, 2048), 16, 4, "square_large: 4 waves at 1024B is the measured optimum"),
    ((1, 1, 32, 32768), 8, 128, "short_wide_wide: 2 waves at 512B is the measured optimum"),
    ((1, 1, 1024, 1024), 8, 4, "square_mid: 512B clears the floor and is inside the noise band"),
    ((1, 1, 2048, 64), 2, 1, "full_width: C==2 cannot buy a wave above the floor"),
    ((1, 1, 16384, 32), 1, 1, "tall_narrow: R already supplies 8 waves; knob inert"),
]


@pytest.mark.parametrize("shape,bw,chunks,why", PRODUCTION_CHOICE, ids=[c[3].split(":")[0] for c in PRODUCTION_CHOICE])
def test_production_wave_choice(device, shape, bw, chunks, why):
    grid = device.compute_with_storage_grid_size()
    torch.manual_seed(11)
    t = torch.randn(shape, dtype=torch.float32).bfloat16()
    x = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y = tilize(x)
    plan = pd.derive_plan(x, y, low_l1=False, grid=grid)
    assert (plan.block_width_tiles, plan.num_w_chunks) == (bw, chunks), why
    # Either the read clears the floor, or the geometry could not buy a wave at
    # all and the cut fell back to the occupancy-only one (C too small).
    assert (
        plan.block_row_bytes >= pd.MIN_BLOCK_ROW_BYTES
        or plan.num_w_chunks == plan.tensor_col_tiles
        or (plan.num_w_chunks * plan.tensor_row_blocks <= 64)
    )
    assert len(plan.assignment) == 64
    assert torch.equal(ttnn.to_torch(y), t)
