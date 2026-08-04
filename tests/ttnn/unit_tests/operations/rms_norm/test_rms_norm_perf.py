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
