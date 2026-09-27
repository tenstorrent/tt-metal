# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — the five per-block-fixed-cost knobs, pinned on the HOST.

DO NOT DELETE.  Every one of these is invisible to a numerical test: the op produces
the same values whichever way the knob goes, so only an assertion on the built
ProgramDescriptor can catch a later phase inlining one to a constant or flattening a
measured default back.  The measurements behind each default are in the changelog's
Refinement 3 entry and next to the constant itself.

  1. PASS_A_SQ_BLOCK -- SHIPPED ON.  Pass A's `square` ran at DEST block_size 1 while
     every pass-B chain took PASS_B_BLK; giving it the same block measured 1.009-1.026x
     on the interleaved prefill.  Pinned ON, and pinned as an APPENDED compute CT arg so
     `test_program_is_structurally_the_seeds` still sees the seed's prefix.
  2. RES_FUSE (Lamp L-RES-FUSE) -- parked at its byte-identical default, kept LIVE.  The
     four-element form the lamp describes is structurally wrong (pack is its own cohort,
     so an intermediate DEST value cannot be published); the correct STREAM-only form
     measured 0.989x.
  3. DM_TXN_ROWS_MAX (lever 3) -- parked at 1, kept LIVE.  The packed CT word MUST be a
     plain `block_rows` at the default, and TXN_ROWS must always DIVIDE BLOCK_ROWS --
     that divisibility is what makes the multi-tile-row reserve straddle-free, so it is
     an invariant, not a preference.
  4. PER_CHANNEL_TRIM_{GAMMA,BIAS} (lever 4 / Lamp L-OPERAND-TRIM) -- D23's derived
     policy WON the re-measurement (coarser reads are 0.76-1.00x), so both stay derived.
     The legality filter is the load-bearing part: a forced granularity may never
     produce a TRUNCATED read on a dtype whose face offset is not 64-byte aligned.
  5. CB_SQ_EXACT -- parked at 0, kept LIVE.  Charging cb_x_squared its real width under
     the D12 fold coarsens BLOCK_ROWS on the 64-core BLOCK shard (20 -> 25) and measured
     0.987x there, so the conservative price ships.

Nothing here dispatches; every assertion is on the host-built ProgramDescriptor.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor import (
    COMPUTE_CT_SCALARS,
    READER_CT_SCALARS,
    TRIM_DERIVED,
    _dm_txn_rows,
    _pack_txn_rows,
    create_program_descriptor,
)

_ML = ttnn.TensorMemoryLayout

# Where each knob lands.  Indices, not slices: an off-by-one here is exactly the drift
# these tests exist to catch.
_CT_BLOCK_ROWS = 4  # reader AND writer: BLOCK_ROWS | (TXN_ROWS - 1) << 16
_CT_GAMMA_TRIM = 19  # reader
_CT_BIAS_TRIM = 23  # reader
_CT_PASS_A_SQ_BLOCK = 24  # compute
_CT_RES_FUSE = 25  # compute


def _config():
    c = ttnn.ComputeConfigDescriptor()
    c.math_fidelity = ttnn.MathFidelity.HiFi2
    c.fp32_dest_acc_en = False
    c.math_approx_mode = False
    return c


def _build(device, shape, *, mode="gamma", dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Build the descriptor for one configuration.  No dispatch."""
    torch.manual_seed(0)
    W = shape[-1]
    mc = ttnn.DRAM_MEMORY_CONFIG
    zeros = torch.zeros(shape, dtype=torch.bfloat16)
    x = ttnn.from_torch(zeros, dtype=dtype, layout=layout, device=device, memory_config=mc)
    vec = torch.zeros(1, 1, 1, W, dtype=torch.bfloat16)
    w = ttnn.from_torch(vec, dtype=dtype, layout=layout, device=device) if "gamma" in mode else None
    b = ttnn.from_torch(vec, dtype=dtype, layout=layout, device=device) if "bias" in mode else None
    r = (
        ttnn.from_torch(zeros, dtype=dtype, layout=layout, device=device, memory_config=mc)
        if "residual" in mode
        else None
    )
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(list(shape)), dtype, layout, device, mc)
    return create_program_descriptor(
        x, out, weight=w, bias=b, residual=r, epsilon=1e-12, compute_kernel_config=_config()
    )


# --------------------------------------------------------------------------------
# 1 + 2.  The two pass-A knobs: shipped values, and that both are still LIVE.
# --------------------------------------------------------------------------------


def test_pass_a_square_takes_the_dest_block_by_default(device):
    """Measured 1.009-1.026x on the interleaved prefill -- it ships ON."""
    assert PD.PASS_A_SQ_BLOCK == 1, "pass A's DEST-lane block is the phase's measured win; it ships on"
    d = _build(device, (1, 1, 256, 1024))
    ct = list(d.kernels[2].compile_time_args)
    assert len(ct) == COMPUTE_CT_SCALARS
    assert ct[_CT_PASS_A_SQ_BLOCK] == 1


def test_res_fuse_is_parked_at_its_byte_identical_default(device):
    """Lamp L-RES-FUSE measured 0.989x where it is even correct -- parked, NOT deleted."""
    assert PD.RES_FUSE == 0
    d = _build(device, (1, 1, 256, 1024), mode="gamma_bias_residual")
    assert list(d.kernels[2].compile_time_args)[_CT_RES_FUSE] == 0


@pytest.mark.parametrize("knob, index", [("PASS_A_SQ_BLOCK", _CT_PASS_A_SQ_BLOCK), ("RES_FUSE", _CT_RES_FUSE)])
def test_the_pass_a_knobs_are_still_live(device, knob, index):
    """Flipping the module constant must MOVE the CT arg -- an inlined knob is a dead knob."""
    saved = getattr(PD, knob)
    try:
        for value in (0, 1):
            setattr(PD, knob, value)
            d = _build(device, (1, 1, 256, 1024), mode="gamma_bias_residual")
            assert list(d.kernels[2].compile_time_args)[index] == value, f"{knob} no longer reaches the kernel"
    finally:
        setattr(PD, knob, saved)


# --------------------------------------------------------------------------------
# 3.  The transaction unit: default byte-identity, and the divisibility INVARIANT.
# --------------------------------------------------------------------------------


def test_the_transaction_word_is_a_plain_block_rows_at_the_default(device):
    assert PD.DM_TXN_ROWS_MAX == 1, "lever 3 measured flat; it ships parked at the seed's per-tile-row barrier"
    for shape in [(1, 1, 8192, 1024), (1, 1, 256, 1024), (1, 1, 1024, 256)]:
        d = _build(device, shape)
        reader = list(d.kernels[0].compile_time_args)[_CT_BLOCK_ROWS]
        writer = list(d.kernels[1].compile_time_args)[_CT_BLOCK_ROWS]
        assert reader == writer, "both dataflow halves must decode the SAME packed word"
        assert reader >> 16 == 0, f"{shape}: the default must encode TXN_ROWS == 1, i.e. a plain block_rows"


@pytest.mark.parametrize("block_rows", list(range(1, 41)))
@pytest.mark.parametrize("cap", [0, 1, 2, 3, 4, 8, 1000])
def test_the_transaction_unit_always_divides_block_rows(block_rows, cap):
    """The straddle-free invariant.  A group of TXN_ROWS * WT_CHUNK pages starting at a
    block-aligned offset can only stay inside a `depth * BLOCK_ROWS * WT_CHUNK` ring if
    TXN_ROWS divides BLOCK_ROWS -- so this is a correctness property, not a preference."""
    saved = PD.DM_TXN_ROWS_MAX
    try:
        PD.DM_TXN_ROWS_MAX = cap
        txn = _dm_txn_rows(block_rows)
        assert 1 <= txn <= block_rows
        assert block_rows % txn == 0
        if cap == 0:
            assert txn == block_rows, "cap 0 means the whole row-block, the design's stated intent"
        else:
            assert txn <= cap
        word = _pack_txn_rows(block_rows, txn)
        assert word & 0xFFFF == block_rows
        assert (word >> 16) + 1 == txn
    finally:
        PD.DM_TXN_ROWS_MAX = saved


def test_a_transaction_unit_that_does_not_divide_the_block_is_refused(expect_error):
    with expect_error(AssertionError, "must divide BLOCK_ROWS"):
        _pack_txn_rows(6, 4)


# --------------------------------------------------------------------------------
# 4.  The per-channel trim: derived by default, and the legality filter.
# --------------------------------------------------------------------------------


def test_per_channel_trim_ships_derived(device):
    """D23's derived policy WON the re-measurement (0.76-1.00x for anything coarser)."""
    assert PD.PER_CHANNEL_TRIM_GAMMA == TRIM_DERIVED
    assert PD.PER_CHANNEL_TRIM_BIAS == TRIM_DERIVED
    d = _build(device, (1, 1, 256, 1024), mode="gamma_bias")
    ct = list(d.kernels[0].compile_time_args)
    assert len(ct) >= READER_CT_SCALARS
    # bf16's 2048-byte tile has a 512-byte face: 64-byte aligned, so both take the
    # two-face-row form, and the bias takes it from its OWN tile size.
    assert ct[_CT_GAMMA_TRIM] == 2
    assert ct[_CT_BIAS_TRIM] == 2


def test_a_forced_trim_can_never_truncate_a_block_float_read(device):
    """bfloat8_b's 1088-byte tile has a 272-byte face, which is NOT 64-byte aligned: a
    face-offset read would be silently truncated DOWN to the alignment.  Forcing the
    face-row form must fall back to the half page, never produce the truncating read."""
    saved = (PD.PER_CHANNEL_TRIM_GAMMA, PD.PER_CHANNEL_TRIM_BIAS)
    try:
        PD.PER_CHANNEL_TRIM_GAMMA = 2
        PD.PER_CHANNEL_TRIM_BIAS = 2
        d = _build(device, (1, 1, 256, 1024), mode="gamma_bias", dtype=ttnn.bfloat8_b)
        ct = list(d.kernels[0].compile_time_args)
        assert ct[_CT_GAMMA_TRIM] == 1
        assert ct[_CT_BIAS_TRIM] == 1
    finally:
        PD.PER_CHANNEL_TRIM_GAMMA, PD.PER_CHANNEL_TRIM_BIAS = saved


@pytest.mark.parametrize("forced", [0, 1, 2])
def test_the_trim_knobs_are_still_live(device, forced):
    saved = PD.PER_CHANNEL_TRIM_GAMMA
    try:
        PD.PER_CHANNEL_TRIM_GAMMA = forced
        d = _build(device, (1, 1, 256, 1024), mode="gamma")
        assert list(d.kernels[0].compile_time_args)[_CT_GAMMA_TRIM] == forced
    finally:
        PD.PER_CHANNEL_TRIM_GAMMA = saved


# --------------------------------------------------------------------------------
# 5.  The cb_x_squared price: the conservative default ships.
# --------------------------------------------------------------------------------


def test_cb_x_squared_keeps_the_seeds_conservative_price(device):
    """CB_SQ_EXACT = 1 is CORRECT (it can only ever shrink the price, never overflow L1)
    but coarsens BLOCK_ROWS 20 -> 25 on the 64-core BLOCK shard and measured 0.987x
    there, so the conservative price ships and the exact one stays a live knob."""
    assert PD.CB_SQ_EXACT == 0
    saved = PD.CB_SQ_EXACT
    try:
        # A shape where the D12 fold is on and L1 binds: the two must DIFFER, or the
        # knob has stopped doing anything.
        shape = (1, 1, 1024, 128)
        blocks = {}
        for value in (0, 1):
            PD.CB_SQ_EXACT = value
            blocks[value] = list(_build(device, shape).kernels[2].compile_time_args)[3]
        assert blocks[1] >= blocks[0], "the exact price can only ever admit a COARSER block"
    finally:
        PD.CB_SQ_EXACT = saved
