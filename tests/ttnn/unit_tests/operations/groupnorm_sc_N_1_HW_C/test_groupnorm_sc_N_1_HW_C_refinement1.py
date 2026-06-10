# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1 tests: dtype expansion (fp32, bf8b), mixed-precision affine,
compute_kernel_config exposure, and multi-core (n, g)-group distribution.

Tolerances mirror the golden suite TOLERANCES (per activation dtype):
  fp32 (0.999, 0.01) · bf16 (0.995, 0.02) · bf8b (0.99, 0.10).
bf8b tensors must be TILE layout (block format has no row-major form).
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


def run_case(
    device, shape, num_groups, dtype, affine_dtype=None, affine_layout=None, compute_kernel_config=None, rms_scale=1.0
):
    """Build tensors, run op, check vs torch reference with per-dtype tolerance.

    rms_scale widens the rel-RMS band for reduced-fidelity configs (LoFi is
    documented lower-precision hardware behavior, not an op bug)."""
    torch.manual_seed(0)
    N, _, HW, C = shape
    x = torch.randn(shape, dtype=TORCH_DTYPE[dtype])

    gamma = beta = tt_g = tt_b = None
    if affine_dtype is not None:
        layout = affine_layout or ttnn.ROW_MAJOR_LAYOUT
        gamma = torch.randn(1, 1, 1, C, dtype=TORCH_DTYPE[affine_dtype])
        beta = torch.randn(1, 1, 1, C, dtype=TORCH_DTYPE[affine_dtype])
        tt_g = ttnn.from_torch(gamma, dtype=affine_dtype, layout=layout, device=device)
        tt_b = ttnn.from_torch(beta, dtype=affine_dtype, layout=layout, device=device)

    expected = torch_groupnorm(x, num_groups, gamma, beta)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(
        groupnorm_sc_N_1_HW_C(tt_x, num_groups, gamma=tt_g, beta=tt_b, compute_kernel_config=compute_kernel_config)
    ).to(torch.float32)

    pcc_thresh, rms_thresh = TOLERANCES[dtype]
    rms_thresh *= rms_scale
    abs_err = (result - expected).abs()
    rel_rms = (abs_err.pow(2).mean().sqrt() / expected.std().clamp(min=1e-10)).item()
    print(f"\nshape={shape} G={num_groups} dtype={dtype} affine={affine_dtype}: " f"rel_rms={rel_rms:.5f}")
    assert_with_pcc(expected, result, pcc=pcc_thresh)
    assert rel_rms < rms_thresh, f"relative RMS {rel_rms:.5f} exceeds {rms_thresh}"


# --- dtype expansion: fp32 / bf8b activations, with and without affine ------


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        pytest.param((1, 1, 32, 32), 1, id="32x32_g1"),
        pytest.param((1, 1, 128, 128), 4, id="128x128_g4"),
        pytest.param((2, 1, 64, 128), 4, id="2x64x128_g4"),
        pytest.param((1, 1, 512, 256), 8, id="512x256_g8"),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["fp32", "bf8b"])
def test_new_dtype_gamma_beta(device, shape, num_groups, dtype):
    """fp32 / bf8b activations, same-dtype gamma+beta (bf8b affine in TILE)."""
    affine_layout = ttnn.TILE_LAYOUT if dtype == ttnn.bfloat8_b else ttnn.ROW_MAJOR_LAYOUT
    run_case(device, shape, num_groups, dtype, affine_dtype=dtype, affine_layout=affine_layout)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["fp32", "bf8b"])
def test_new_dtype_no_affine(device, dtype):
    run_case(device, (2, 1, 64, 128), 4, dtype)


# --- mixed precision: activation dtype != affine dtype ----------------------


@pytest.mark.parametrize(
    "dtype, affine_dtype, affine_layout",
    [
        pytest.param(ttnn.bfloat16, ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, id="bf16_act_fp32_affine"),
        pytest.param(ttnn.bfloat16, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, id="bf16_act_bf8b_affine"),
        pytest.param(ttnn.float32, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="fp32_act_bf16_affine"),
        pytest.param(ttnn.bfloat8_b, ttnn.float32, ttnn.TILE_LAYOUT, id="bf8b_act_fp32_affine"),
    ],
)
def test_mixed_precision_affine(device, dtype, affine_dtype, affine_layout):
    run_case(device, (1, 1, 128, 128), 4, dtype, affine_dtype=affine_dtype, affine_layout=affine_layout)


# --- compute_kernel_config exposure ------------------------------------------


@pytest.mark.parametrize(
    "fidelity",
    [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi],
    ids=["HiFi4", "HiFi2", "LoFi"],
)
@pytest.mark.parametrize("fp32_acc", [False, True], ids=["bf16_acc", "fp32_acc"])
def test_compute_kernel_config(device, fidelity, fp32_acc):
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
        dst_full_sync_en=False,
    )
    rms_scale = 2.0 if fidelity == ttnn.MathFidelity.LoFi else 1.0
    run_case(
        device,
        (1, 1, 128, 128),
        4,
        ttnn.bfloat16,
        affine_dtype=ttnn.bfloat16,
        compute_kernel_config=cfg,
        rms_scale=rms_scale,
    )


def test_default_config_unchanged(device):
    """Passing nothing must behave like Phase 0 (HiFi4, no fp32 acc)."""
    run_case(device, (1, 1, 128, 128), 4, ttnn.bfloat16, affine_dtype=ttnn.bfloat16)


# --- multi-core distribution -------------------------------------------------
# Work unit = one (n, g) group; cases force every split regime.


@pytest.mark.parametrize(
    "shape, num_groups",
    [
        # fewer groups than cores (3 used)
        pytest.param((1, 1, 64, 96), 3, id="under_grid_3groups"),
        # exactly 64 groups — even split on an 8x8 grid
        pytest.param((2, 1, 32, 1024), 32, id="even_64groups"),
        # 100 groups — uneven split (core_group_2 non-empty)
        pytest.param((4, 1, 32, 800), 25, id="uneven_100groups"),
        # multi-batch with multi-tile group widths
        pytest.param((8, 1, 64, 64), 2, id="batch8_g2"),
    ],
)
def test_multicore_distribution(device, shape, num_groups):
    run_case(device, shape, num_groups, ttnn.bfloat16, affine_dtype=ttnn.bfloat16)


def test_multicore_distinct_groups(device):
    """Per-(n,g) constant slabs — wrong group routing across cores changes
    output values catastrophically, so this catches mis-distribution."""
    N, HW, C, G = 4, 64, 256, 8
    shape = (N, 1, HW, C)
    Cg = C // G
    x = torch.zeros(shape, dtype=torch.bfloat16)
    for n in range(N):
        for g in range(G):
            # offset n*G+g, scale g+1 — unique per group
            base = torch.linspace(-1, 1, HW * Cg).reshape(HW, Cg)
            x[n, 0, :, g * Cg : (g + 1) * Cg] = (base * (g + 1) + (n * G + g)).to(torch.bfloat16)

    expected = torch_groupnorm(x, G)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(groupnorm_sc_N_1_HW_C(tt_x, G)).to(torch.float32)
    assert_with_pcc(expected, result, pcc=0.995)
