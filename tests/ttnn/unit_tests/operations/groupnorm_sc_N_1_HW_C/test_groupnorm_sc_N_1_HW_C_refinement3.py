# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 tests: non-tile-aligned group widths (SD / SDXL regime).

groups_alignment=non_aligned — (C / num_groups) % 32 != 0 with G > 1, so
groups straddle tile boundaries. The kernel switches to the cluster path:
work unit = (n, cluster) of lcm(Cg, 32) channels; passes 1/2 run per group
with 0/1 column masks; pass 3 is one Row-broadcast sweep over the cluster
using reader-scattered per-column mean/rstd row vectors.

Coverage: SD/SDXL widths (Cg in {10, 20, 30, 40, 80}), non-SD group counts
(Cg in {16, 24, 36, 56}), the hardest coupling (C tail AND group straddle,
Cg in {24, 20, 36, 25}), HW tails + straddle, all dtypes (incl. bf8b output
mask path), affine variants, multi-batch, multicluster + multicore split.

Tolerances mirror golden TOLERANCES: fp32 (0.999, 0.01) · bf16 (0.995, 0.02)
· bf8b (0.99, 0.10).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

TOLERANCES = {
    ttnn.float32: (0.999, 0.01),
    ttnn.bfloat16: (0.995, 0.02),
    ttnn.bfloat8_b: (0.99, 0.10),
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}


def torch_groupnorm(x, num_groups, gamma=None, beta=None, eps=1e-5):
    N, _, HW, C = x.shape
    x_nchw = x.to(torch.float32).squeeze(1).permute(0, 2, 1)
    w = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    b = beta.to(torch.float32).reshape(C) if beta is not None else None
    y = torch.nn.functional.group_norm(x_nchw, num_groups, weight=w, bias=b, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1)


def run_case(device, shape, num_groups, dtype=ttnn.bfloat16, affine="gamma_beta", affine_dtype=None, layout=None):
    torch.manual_seed(0)
    N, _, HW, C = shape
    layout = layout or ttnn.TILE_LAYOUT
    affine_dtype = affine_dtype or ttnn.bfloat16

    gamma = beta = tt_g = tt_b = None
    if affine in ("gamma_beta", "gamma_only"):
        gamma = torch.randn(1, 1, 1, C, dtype=TORCH_DTYPE[affine_dtype])
        affine_layout = ttnn.TILE_LAYOUT if affine_dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
        tt_g = ttnn.from_torch(gamma, dtype=affine_dtype, layout=affine_layout, device=device)
    if affine == "gamma_beta":
        beta = torch.randn(1, 1, 1, C, dtype=TORCH_DTYPE[affine_dtype])
        tt_b = ttnn.from_torch(beta, dtype=affine_dtype, layout=affine_layout, device=device)

    x = torch.randn(shape, dtype=TORCH_DTYPE[dtype])
    expected = torch_groupnorm(x, num_groups, gamma, beta)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)
    result = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, num_groups, gamma=tt_g, beta=tt_b)).to(torch.float32)

    pcc_thresh, rms_thresh = TOLERANCES[dtype]
    abs_err = (result - expected).abs()
    rel_rms = (abs_err.pow(2).mean().sqrt() / expected.std().clamp(min=1e-10)).item()
    print(f"\nshape={shape} G={num_groups} dtype={dtype} affine={affine}: rel_rms={rel_rms:.5f}")
    assert_with_pcc(expected, result, pcc=pcc_thresh)
    assert rel_rms < rms_thresh, f"relative RMS {rel_rms:.5f} exceeds {rms_thresh}"


# --- SD / SDXL regime: tile-aligned C, straddling groups --------------------


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        ((1, 1, 32, 320), 32),  # Cg=10, cluster=5 tiles x 16 groups
        ((1, 1, 64, 640), 32),  # Cg=20, cluster=5 tiles x 8 groups
        ((1, 1, 32, 960), 32),  # Cg=30, cluster=15 tiles x 16 groups
        ((1, 1, 64, 1280), 32),  # Cg=40, span 3 tiles
        ((1, 1, 32, 2560), 32),  # Cg=80, 16 clusters
        ((1, 1, 64, 192), 8),  # Cg=24
        ((1, 1, 64, 384), 8),  # Cg=48
        ((1, 1, 128, 448), 8),  # Cg=56
        ((1, 1, 64, 160), 8),  # Cg=20, single cluster
        ((1, 1, 128, 48), 3),  # Cg=16, half-tile groups
    ],
)
def test_sd_widths(device, shape, num_groups):
    run_case(device, shape, num_groups)


def test_sd_full_size(device):
    # Real SD1.5 stage: HW=1024 exercises Ht=32 streaming on the cluster path.
    run_case(device, (1, 1, 1024, 640), 32)


# --- affine variants over straddling groups ---------------------------------


@pytest.mark.parametrize("affine", ["gamma_beta", "gamma_only", "no_affine"])
def test_affine_variants(device, affine):
    run_case(device, (1, 1, 64, 320), 32, affine=affine)


@pytest.mark.parametrize("affine_dtype", [ttnn.float32, ttnn.bfloat8_b])
def test_mixed_affine_dtype(device, affine_dtype):
    run_case(device, (1, 1, 64, 160), 8, affine_dtype=affine_dtype)


# --- dtypes ------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b])
def test_dtypes(device, dtype):
    run_case(device, (1, 1, 64, 320), 32, dtype=dtype)


def test_row_major_input(device):
    run_case(device, (1, 1, 64, 192), 8, layout=ttnn.ROW_MAJOR_LAYOUT)


# --- C tail AND group straddle (the 8 hardest golden cells) ------------------


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        ((1, 1, 64, 48), 2),  # Cg=24, c_tail=16
        ((1, 1, 64, 80), 2),  # Cg=40, c_tail=16
        ((1, 1, 128, 48), 3),  # Cg=16, c_tail=16
        ((1, 1, 64, 80), 4),  # Cg=20
        ((1, 1, 128, 144), 4),  # Cg=36, c_tail=16
        ((1, 1, 64, 200), 8),  # Cg=25, whole row one cluster
        ((2, 1, 64, 48), 2),  # multi-batch
    ],
)
def test_c_tail_with_straddle(device, shape, num_groups):
    run_case(device, shape, num_groups)


# --- HW tail + straddle ------------------------------------------------------


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        ((1, 1, 50, 320), 32),  # hw_tail=18
        ((1, 1, 17, 80), 4),  # hw_tail=17, c_tail=16
        ((2, 1, 47, 160), 8),  # multi-batch + hw_tail
    ],
)
def test_hw_tail_with_straddle(device, shape, num_groups):
    run_case(device, shape, num_groups)


def test_hw_tail_bf8b_output_mask(device):
    # bf8b + HW tail exercises the scalar output-mask path (cb_mask_ones/rows).
    run_case(device, (1, 1, 50, 320), 32, dtype=ttnn.bfloat8_b)


# --- multi-batch / multicore split over clusters ------------------------------


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        ((2, 1, 64, 640), 32),  # 8 cluster units
        ((4, 1, 32, 320), 32),  # 8 cluster units across batches
        ((2, 1, 128, 192), 8),  # Cg=24, 4 units
    ],
)
def test_multibatch_multicore(device, shape, num_groups):
    run_case(device, shape, num_groups)
