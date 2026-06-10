# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 2 tests: non-tile-aligned shapes (HW and C tails).

- hw_non_aligned: HW % 32 != 0, C aligned. Tail rows of the last tile row are
  masked in the variance pass (zero-padded input keeps the mean pass exact).
- c_non_aligned: C % 32 != 0, G == 1 (a single group spans C, so groups never
  straddle tiles). Tail cols of the last tile column are masked.
- both tails: corner tile masked on both axes.

C tails with G > 1 (group straddle) remain unsupported until Refinement 3 —
validate must raise NotImplementedError.

Tolerances mirror golden TOLERANCES: fp32 (0.999, 0.01) · bf16 (0.995, 0.02) ·
bf8b (0.99, 0.10).
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


# --- hw_non_aligned (HW % 32 != 0, C aligned) -------------------------------


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        pytest.param((1, 1, 17, 64), 1, id="17x64_g1"),
        pytest.param((1, 1, 50, 128), 1, id="50x128_g1"),
        pytest.param((1, 1, 47, 256), 1, id="47x256_g1"),
        pytest.param((2, 1, 100, 128), 1, id="2x100x128_g1"),
        # hw tail with multiple (tile-aligned) groups — masking shared by groups
        pytest.param((1, 1, 50, 128), 4, id="50x128_g4"),
        pytest.param((2, 1, 100, 256), 8, id="2x100x256_g8"),
    ],
)
def test_hw_non_aligned(device, shape, num_groups):
    run_case(device, shape, num_groups)


# --- c_non_aligned (C % 32 != 0, single group) -------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 64, 17), id="64x17"),
        pytest.param((1, 1, 64, 50), id="64x50"),
        pytest.param((1, 1, 128, 100), id="128x100"),
        pytest.param((2, 1, 64, 47), id="2x64x47"),
    ],
)
def test_c_non_aligned(device, shape):
    run_case(device, shape, 1)


# --- both tails at once -------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 17, 17), id="17x17"),
        pytest.param((1, 1, 50, 100), id="50x100"),
        pytest.param((2, 1, 47, 50), id="2x47x50"),
    ],
)
def test_both_tails(device, shape):
    run_case(device, shape, 1)


# --- affine / dtype coverage on tail shapes ----------------------------------


@pytest.mark.parametrize("affine", ["gamma_beta", "gamma_only", "no_affine"])
def test_affine_variants(device, affine):
    run_case(device, (1, 1, 50, 100), 1, affine=affine)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["fp32", "bf8b"])
def test_tail_dtypes(device, dtype):
    run_case(device, (1, 1, 50, 64), 1, dtype=dtype)
    run_case(device, (1, 1, 64, 50), 1, dtype=dtype)


def test_rm_input_layout(device):
    """ROW_MAJOR input — host-side tilize zero-pads the tails."""
    run_case(device, (1, 1, 50, 100), 1, layout=ttnn.ROW_MAJOR_LAYOUT)


def test_fp32_affine_on_tail(device):
    run_case(device, (1, 1, 47, 64), 1, affine_dtype=ttnn.float32)


# --- group straddle stays gated until Refinement 3 ----------------------------


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        pytest.param((1, 1, 64, 48), 2, id="c_tail_straddle_g2"),
        pytest.param((1, 1, 64, 320), 32, id="aligned_c_straddle_g32"),
    ],
)
def test_group_straddle_rejected(device, shape, num_groups):
    x = ttnn.from_torch(
        torch.randn(shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    with pytest.raises(NotImplementedError):
        groupnorm_sc_N_1_HW_C(x, num_groups)
