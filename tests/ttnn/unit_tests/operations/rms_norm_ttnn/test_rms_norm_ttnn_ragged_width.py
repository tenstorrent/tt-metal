# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 -- the RAGGED (PADDED) WIDTH CHUNK (descriptor D32).

DO NOT DELETE.  What this pins, and why each half is here:

  1. THE KNOB IS LIVE.  `RAGGED_WIDTH_CHUNK` must still move the blocking, and
     `ROW_RESIDENT_MIN_CHUNK_WT` must still move the regime.  An inlined knob is a
     dead knob, and both of these were built as tunables on purpose.
  2. THE CLIFF IS GONE.  At a PRIME `Wt` (4064 = 127 * 32, 2848 = 89 * 32) the only
     divisor below the L1 cap is 1, which repaid every per-phase init / reconfig /
     pipeline fill-and-drain `Wt` times per block.  Every chunked build on those
     widths must now come out at WT_CHUNK > 1.
  3. THE PAD IS ACCOUNTED FOR, EXACTLY.  `NUM_W_CHUNKS * WT_CHUNK` is the PADDED
     width; the held CBs span it, the reader zeroes the pad tiles and the writer
     skips them.  The two dataflow kernels must agree on the count -- the reader
     carries it as its own CT scalar, the writer PACKED into WT_CHUNK's word
     (its arg-list LENGTH is a checked structural property).
  4. THE GATE HOLDS.  A width that is not tile-aligned (PARTIAL_W != 0) keeps D1's
     divisor clamp, because the reduce's partial scaler / 0-1 mask is aimed at the
     last tile of the block and padding would make that a PAD tile.
  5. IT IS STILL CORRECT ON DEVICE, at the prime widths, in both layouts, with and
     without every optional operand -- the pad tiles must contribute exactly 0 to
     sum(t^2), which is the one thing a wrong pad would silently break.

Measured (blackhole p150b 1350 MHz, ragged vs the divisor clamp): 1.34x-10.46x on
the prime-Wt shapes, and 1.05x on (1,1,1024,16384) STREAM where the coarsest
divisor was also not the coarsest FITTING chunk.  See the changelog.
"""

from __future__ import annotations

import pytest
import torch

import ttnn

import ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor as PD
from ttnn.operations.rms_norm_ttnn import rms_norm_ttnn, torch_rms_norm_ttnn
from ttnn.operations.rms_norm_ttnn.rms_norm_ttnn_program_descriptor import (
    READER_CT_SCALARS,
    _largest_divisor_at_most,
    _width_chunk,
    create_program_descriptor,
)

# Where the two halves of the pad count land.  Indices, not slices: an off-by-one
# here is exactly the drift this file exists to catch.
_READER_CT_WT_CHUNK = 2
_READER_CT_NUM_W_CHUNKS = 3
_READER_CT_WT_PAD = 29
_WRITER_CT_WT_CHUNK_PACKED = 2  # WT_CHUNK | WT_PAD << 16
_WRITER_CT_NUM_W_CHUNKS = 3

# 4064 = 127 * 32 and 2848 = 89 * 32 -- both Wt PRIME, so `_largest_divisor_at_most`
# returns 1 for every cap below the whole row.  These are the resilience-group shapes
# op_requirements.md names as the cliff's only reachable region.
PRIME_W = [4064, 2848]


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


def _blocking(descriptor):
    """(WT_CHUNK, NUM_W_CHUNKS, WT_PAD) as the READER sees it, cross-checked against
    the WRITER's packed word -- the two kernels index the same pad and must agree."""
    reader = list(descriptor.kernels[0].compile_time_args)
    writer = list(descriptor.kernels[1].compile_time_args)
    assert len(reader) > READER_CT_SCALARS
    wt_chunk = reader[_READER_CT_WT_CHUNK]
    num_chunks = reader[_READER_CT_NUM_W_CHUNKS]
    wt_pad = reader[_READER_CT_WT_PAD]
    packed = writer[_WRITER_CT_WT_CHUNK_PACKED]
    assert packed & 0xFFFF == wt_chunk, "the writer's WT_CHUNK diverged from the reader's"
    assert packed >> 16 == wt_pad, "the writer's WT_PAD diverged from the reader's"
    assert writer[_WRITER_CT_NUM_W_CHUNKS] == num_chunks
    return wt_chunk, num_chunks, wt_pad


# --------------------------------------------------------------------------------
# 1.  `_width_chunk` -- the ONE source of truth for the chunk-count decision.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("wt_core", [127, 89, 128, 224, 512, 100, 7, 1])
@pytest.mark.parametrize("cap", [1, 2, 3, 5, 8, 16, 32, 63, 64, 109, 1000])
def test_width_chunk_is_a_valid_covering_split(wt_core, cap):
    """Whatever it returns must COVER the row, fit the cap, and be at least as coarse
    as D1's divisor -- those three are the whole contract."""
    wtc, n = _width_chunk(wt_core, cap, ragged_ok=True)
    effective_cap = max(1, min(cap, wt_core))
    assert 1 <= wtc <= effective_cap
    assert n >= 1
    assert n * wtc >= wt_core, "the chunking must cover every real width tile"
    assert (n - 1) * wtc < wt_core, "a whole chunk of pure pad is a chunk that should not exist"
    assert wtc >= _largest_divisor_at_most(wt_core, effective_cap), "never coarser under D1 than under D32"
    # The balanced form is what keeps the pad negligible.
    assert n * wtc - wt_core < n


@pytest.mark.parametrize("wt_core", [127, 89, 128, 224, 512, 100])
@pytest.mark.parametrize("cap", [2, 5, 16, 32, 64])
def test_width_chunk_declines_to_the_divisor_when_asked(wt_core, cap):
    """`ragged_ok=False` and `RAGGED_WIDTH_CHUNK=0` must both give D1 exactly."""
    div = _largest_divisor_at_most(wt_core, max(1, min(cap, wt_core)))
    assert _width_chunk(wt_core, cap, ragged_ok=False) == (div, wt_core // div)
    saved = PD.RAGGED_WIDTH_CHUNK
    try:
        PD.RAGGED_WIDTH_CHUNK = 0
        assert _width_chunk(wt_core, cap, ragged_ok=True) == (div, wt_core // div)
    finally:
        PD.RAGGED_WIDTH_CHUNK = saved


def test_width_chunk_prefers_the_divisor_when_it_is_no_coarser():
    """A power-of-two width has a divisor AT the cap; the pad must not be taken for
    nothing (an unpadded build is byte-identical and strictly cheaper)."""
    for cap in (1, 2, 4, 8, 16, 32, 64, 128):
        wtc, n = _width_chunk(128, cap, ragged_ok=True)
        assert n * wtc == 128, f"cap={cap} padded a width that divides evenly"


# --------------------------------------------------------------------------------
# 2.  The cliff is gone, and the knob that removed it is still live.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("W", PRIME_W)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("mode", ["no_gamma", "gamma", "gamma_bias_residual"])
def test_prime_width_is_off_the_granularity_cliff(device, W, layout, mode):
    """The whole point of the refinement: no chunked build on a prime Wt may sit at
    one tile per chunk."""
    wt_chunk, num_chunks, wt_pad = _blocking(_build(device, (1, 1, 3104, W), mode=mode, layout=layout))
    wt = W // 32
    assert wt_chunk > 1, f"Wt={wt} still collapses to a one-tile chunk ({num_chunks} chunks)"
    assert num_chunks * wt_chunk == wt + wt_pad
    assert wt_pad < wt_chunk


@pytest.mark.parametrize("W", PRIME_W)
@pytest.mark.parametrize("mode", ["gamma", "gamma_bias_residual"])
def test_the_ragged_knob_is_still_live(device, W, mode):
    """Flipping the module constant must MOVE the blocking back onto the cliff.

    A width whose whole row is RESIDENT is not chunked at all (NUM_W_CHUNKS == 1) and
    has no cliff to fall off -- Wt = 89 with only a weight is one -- so the assertion
    is on the CHUNKED builds, which is the population D32 exists for.
    """
    saved = PD.RAGGED_WIDTH_CHUNK
    try:
        PD.RAGGED_WIDTH_CHUNK = 0
        divisor = _blocking(_build(device, (1, 1, 3104, W), mode=mode))
        PD.RAGGED_WIDTH_CHUNK = 1
        ragged = _blocking(_build(device, (1, 1, 3104, W), mode=mode))
    finally:
        PD.RAGGED_WIDTH_CHUNK = saved
    assert divisor[2] == 0, "the divisor clamp never pads"
    if divisor[1] == 1:
        assert ragged == divisor, "a resident row is one chunk either way"
        return
    assert divisor[0] == 1, "a prime Wt has no chunked divisor above 1 -- that IS the cliff"
    assert ragged[0] > divisor[0]


def test_the_l5_chunk_floor_is_still_live(device):
    """`ROW_RESIDENT_MIN_CHUNK_WT` ships at 1 (measured flat once the ragged chunk
    landed -- nothing reaches the floor any more), but it stays a live tunable: at a
    high enough floor the L5 regime must DECLINE and STREAM must take over."""
    saved = PD.ROW_RESIDENT_MIN_CHUNK_WT
    assert saved == 1, "the floor ships at its byte-identical default"
    try:
        PD.ROW_RESIDENT_MIN_CHUNK_WT = 10**9
        wt_chunk, num_chunks, _ = _blocking(_build(device, (1, 1, 3104, 4064), mode="gamma_bias_residual"))
    finally:
        PD.ROW_RESIDENT_MIN_CHUNK_WT = saved
    assert num_chunks > 1 and wt_chunk >= 1


# --------------------------------------------------------------------------------
# 3.  The gate: a non-tile-aligned width keeps D1's divisor clamp.
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("W", [4063, 2847, 4033])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
def test_a_partial_last_tile_refuses_the_pad(device, W, layout):
    """PARTIAL_W != 0 aims the reduce's partial scaler / 0-1 mask at the LAST tile of
    the block; padding would make that a pad tile and silently drop the mask."""
    _, _, wt_pad = _blocking(_build(device, (1, 1, 3104, W), mode="gamma", layout=layout))
    assert wt_pad == 0


# --------------------------------------------------------------------------------
# 4.  On device: the pad tiles must contribute EXACTLY 0 to sum(t^2).
# --------------------------------------------------------------------------------


@pytest.mark.parametrize("W", PRIME_W)
@pytest.mark.parametrize("rows", [32, 3104], ids=["1row", "97row"])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("mode", ["no_gamma", "gamma", "residual", "gamma_bias_residual"])
def test_prime_width_is_correct_on_device(device, W, rows, layout, mode):
    """A pad tile that is not exactly zero inflates sum(t^2) by its own energy, which
    shows up as a uniform under-scale of the whole row -- so PCC is the right probe."""
    if rows == 3104 and W == 2848 and mode != "gamma_bias_residual":
        pytest.skip("one wide row-count per width is enough; the pad geometry is per-core")
    shape = (1, 1, rows, W)
    torch.manual_seed(0)
    t = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
    x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=layout, device=device)
    kwargs, ref = {}, {"input_tensor": t.float()}
    if "gamma" in mode:
        g = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        kwargs["weight"] = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=layout, device=device)
        ref["weight"] = g.float()
    if "bias" in mode:
        b = torch.randn(1, 1, 1, W, dtype=torch.float32).to(torch.bfloat16)
        kwargs["bias"] = ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=layout, device=device)
        ref["bias"] = b.float()
    if "residual" in mode:
        r = torch.randn(shape, dtype=torch.float32).to(torch.bfloat16)
        kwargs["residual_input_tensor"] = ttnn.from_torch(r, dtype=ttnn.bfloat16, layout=layout, device=device)
        ref["residual_input_tensor"] = r.float()

    got = ttnn.to_torch(rms_norm_ttnn(x, epsilon=1e-12, **kwargs)).float().flatten()
    expected = torch_rms_norm_ttnn(**ref, epsilon=1e-12).float().flatten()
    a, b = got - got.mean(), expected - expected.mean()
    pcc = float((a * b).sum() / (a.norm() * b.norm() + 1e-30))
    rel_rms = float((got - expected).pow(2).mean().sqrt() / (expected.pow(2).mean().sqrt() + 1e-30))
    assert pcc > 0.9995, f"pcc={pcc:.6f} rel_rms={rel_rms:.5f}"
    assert rel_rms < 0.04, f"pcc={pcc:.6f} rel_rms={rel_rms:.5f}"
