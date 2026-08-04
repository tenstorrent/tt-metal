# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""On-device perf probes for rms_norm — one op call per test.

Not a correctness gate (the acceptance suite is).  These exist so
`scripts/run_safe_pytest.sh --profile` produces one clean per-op row per
(shape, regime) of interest, and so the same shapes can be re-measured after a
kernel change.

Shape set covers the four structurally different perf situations:
  grid_filled_resident   many tile-rows, narrow W  -> the row split fills the grid
  prefill_resident       Rt >> num_cores           -> BLOCK_ROWS > 1 (coarse block)
  decode_stream          Rt = 1, wide W            -> 1 core, width-chunked
                                                      (this is Lamp L1's target)
  few_rows_resident      Rt < num_cores            -> partial grid
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm

PERF_SHAPES = [
    pytest.param((1, 1, 2048, 256), id="grid_filled_resident"),
    pytest.param((1, 1, 8192, 1024), id="prefill_resident"),
    pytest.param((1, 1, 32, 4096), id="decode_stream"),
    pytest.param((1, 1, 128, 512), id="few_rows_resident"),
]


def _compute_config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def _run(device, shape, layout, gamma_layout):
    torch.manual_seed(42)
    W = shape[-1]
    x = ttnn.from_torch(torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device)
    g = ttnn.from_torch(
        torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=gamma_layout,
        device=device,
    )
    out = rms_norm(x, gamma=g, compute_kernel_config=_compute_config())
    assert tuple(out.shape) == tuple(shape)


@pytest.mark.parametrize("shape", PERF_SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_rms_norm_perf(device, shape, layout):
    _run(device, shape, layout, ttnn.ROW_MAJOR_LAYOUT)


# ---------------------------------------------------------------------------
# CB_DEPTH_CANDIDATES band (descriptor deviation D4).
#
# These widths sit BETWEEN the depth-2 and depth-1 residency thresholds, so the
# depth search is what decides whether x is read once (RESIDENT) or twice
# (STREAM).  Outside the band both depths agree and the knob is inert — which is
# why the first four shapes above cannot measure it.  Rt = 1 on purpose: one
# core, where bytes moved (not overlap) is the binding constraint.
# ---------------------------------------------------------------------------

DEPTH_BAND_CASES = [
    # (shape, gamma_layout) -- band is Wt in [91,126] for TILE gamma,
    #                                 Wt in [80,105] for ROW_MAJOR gamma.
    pytest.param((1, 1, 32, 4032), ttnn.TILE_LAYOUT, id="band_W4032_gamma_tile"),
    pytest.param((1, 1, 32, 3072), ttnn.ROW_MAJOR_LAYOUT, id="band_W3072_gamma_rm"),
]


@pytest.mark.parametrize("shape, gamma_layout", DEPTH_BAND_CASES)
def test_rms_norm_perf_depth_band(device, shape, gamma_layout):
    _run(device, shape, ttnn.TILE_LAYOUT, gamma_layout)


# ---------------------------------------------------------------------------
# Reduce-datapath crossover (descriptor deviation D7, Refinement 1b).
#
# The four probes above run the Phase-0 precision corner (HiFi4 /
# fp32_dest_acc_en=True); the datapath knob is measured at the config the
# `_perf_case` table actually declares -- bfloat16 / HiFi2 /
# fp32_dest_acc_en=False -- because that is a different reduce datapath (16-bit
# DEST) and a stand-in number would be meaningless.
#
# A/B by flipping REDUCE_ACC_VIA_ADD_MIN_WT in the descriptor:
#   4     -> AccumulateViaAdd on every shape here (all have WT_CHUNK >> 4)
#   10**9 -> ReduceTile everywhere (the pre-Refinement-1b datapath)
# ---------------------------------------------------------------------------

DATAPATH_SHAPES = [
    pytest.param((1, 1, 32, 7168), id="decode_w7168"),  # Rt=1, STREAM, the 7x-goal case
    pytest.param((1, 1, 32, 1024), id="decode_w1024"),  # Rt=1, narrow-W control
    pytest.param((1, 1, 8192, 5120), id="prefill_w5120"),  # Rt=256, grid-filling
    pytest.param((1, 1, 224, 3072), id="resilience_w3072"),  # Rt=7, prime tile count
]


def _perf_case_config():
    """The `_perf_case` / `_RESILIENCE_BASE` pinned config."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi2
    cfg.fp32_dest_acc_en = False
    cfg.math_approx_mode = False
    return cfg


@pytest.mark.parametrize("shape", DATAPATH_SHAPES)
def test_rms_norm_perf_reduce_datapath(device, shape):
    torch.manual_seed(42)
    W = shape[-1]
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    g = ttnn.from_torch(
        torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    out = rms_norm(x, gamma=g, compute_kernel_config=_perf_case_config())
    assert tuple(out.shape) == tuple(shape)


# ---------------------------------------------------------------------------
# The interleaved cross-core WIDTH SPLIT (Lamp L1, descriptor D11, Refinement 3).
#
# `Rt = 1` decode profiles: the Phase-0 row split fills exactly ONE core no matter
# how wide W is, so the whole tensor moves through one core's NoC.  GRID_W turns
# the width split on; this A/B sweeps the group size so the crossover (per-core
# bytes fall as 1/gw, the root's gather cost rises with gw) is re-measurable.
#
# GRID_W is the OVERRIDE handle: 1 = no split (the Phase-0 baseline), >= 2 = force
# that many cores per width group, 0 = the shipped AUTO policy.  Identify the
# variants in the profiler CSV by CORE COUNT.
# ---------------------------------------------------------------------------

WIDTH_SPLIT_CASES = [
    # The group-size crossover on the >=7x goal shape (Wt = 224, so every gw here
    # is an exact divisor).  MEASURED 2026-08-04, see descriptor D11:
    #   gw   1 -> 41779 ns    8 -> 13926    16 -> 12876    32 -> 14224    56 -> 19338
    *[pytest.param((1, 1, 32, 7168), gw, id=f"w7168_gw{gw}") for gw in (1, 8, 16, 32, 56)],
    # ... and on the narrow-W control (Wt = 32), where the knob that binds is
    # WIDTH_SPLIT_MIN_WT_PER_CORE rather than the group ceiling:
    #   gw   1 -> 11207 ns    4 ->  7296     8 ->  7149    16 ->  8305
    *[pytest.param((1, 1, 32, 1024), gw, id=f"w1024_gw{gw}") for gw in (1, 4, 8, 16)],
    # The shipped AUTO policy on both targets (must land on the sweep's optimum),
    # and on the two regimes that must NOT split: a mid-Rt shape whose row split
    # already engages 32 cores (WIDTH_SPLIT_MIN_GAIN) and a grid-filling prefill.
    *[
        pytest.param(shape, 0, id=f"auto_{shape[-2]}x{shape[-1]}")
        for shape in ((1, 1, 32, 7168), (1, 1, 32, 1024), (1024, 1024), (1, 1, 8192, 1024))
    ],
    # A one-core minimal program: the ~3.5 us fixed floor every number above sits
    # on top of (kernel launch + dispatch), so the sweep is read against it.
    pytest.param((1, 1, 32, 32), 1, id="floor_1core"),
]


@pytest.mark.parametrize("shape, grid_w", WIDTH_SPLIT_CASES)
def test_rms_norm_perf_width_split(device, shape, grid_w):
    import ttnn.operations.rms_norm.rms_norm_program_descriptor as pdmod

    saved = pdmod.GRID_W
    try:
        pdmod.GRID_W = grid_w
        torch.manual_seed(42)
        W = shape[-1]
        x = ttnn.from_torch(
            torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        g = ttnn.from_torch(
            torch.randn(W, dtype=torch.bfloat16).reshape(1, 1, 1, W),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        out = rms_norm(x, gamma=g, compute_kernel_config=_perf_case_config())
        assert tuple(out.shape) == tuple(shape)
    finally:
        pdmod.GRID_W = saved


# ---------------------------------------------------------------------------
# The ROW_MAJOR BAND scheme (Refinement 2b).
#
# A distinct dataflow from every probe above: x and the output never leave L1
# (each core stages the band it already holds out of its own shard), and the
# per-row stat comes from the cross-core combine rather than a local reduce.  Two
# band geometries, because they take DIFFERENT transaction granularities:
#   band_is_tile_column  shard_w == 32 -> band fills its tile columns exactly, so
#                        a whole tile-row of 32 sticks moves in ONE local read
#                        (and one local write back);
#   sub_tile_band        shard_w == 8  -> the band is a quarter of a tile column,
#                        so the staging is one local read per stick.  This is the
#                        granularity a later perf round would attack.
# Paired with the same shape INTERLEAVED, so the sharded number always has its
# DRAM-fed control measured in the same run.
# ---------------------------------------------------------------------------

_ML = ttnn.TensorMemoryLayout

BAND_PERF_CASES = [
    pytest.param((1, 1, 224, 3072), _ML.WIDTH_SHARDED, id="band_is_tile_column_w3072"),
    pytest.param((1, 1, 224, 3072), _ML.INTERLEAVED, id="control_interleaved_w3072"),
    pytest.param((1, 1, 256, 512), _ML.WIDTH_SHARDED, id="sub_tile_band_w512"),
    pytest.param((1, 1, 256, 512), _ML.INTERLEAVED, id="control_interleaved_w512"),
    pytest.param((1, 1, 224, 3072), _ML.BLOCK_SHARDED, id="band_block_w3072"),
]


@pytest.mark.parametrize("shape, memory_layout", BAND_PERF_CASES)
def test_rms_norm_perf_row_major_band(device, shape, memory_layout):
    from eval.sharding import auto_shard_config

    torch.manual_seed(42)
    W = shape[-1]
    mc = None
    if memory_layout != _ML.INTERLEAVED:
        mc = auto_shard_config(
            list(shape), memory_layout, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16, device=device
        )
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=mc,
    )
    g = ttnn.from_torch(
        torch.randn(W, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    out = rms_norm(
        x, gamma=g, compute_kernel_config=_perf_case_config(), memory_config=(mc if mc is not None else None)
    )
    assert tuple(out.shape) == tuple(shape)
