# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for groupnorm_sc_N_1_HW_C.

GroupNorm over an (N, 1, H*W, C) channel-last tensor: per-(image, group)
statistics over HW x (C / num_groups), centered two-pass variance, optional
per-channel affine. Groups may straddle 32-channel tile boundaries
((C / num_groups) % 32 != 0) and that case is exercised.
"""

import pytest
import torch
import ttnn

from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

# PCC gates per input dtype.
PCC_THRESHOLD = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


# --------------------------------------------------------------------------- #
# Reference + metric
# --------------------------------------------------------------------------- #
def torch_groupnorm_n_1_hw_c(x, num_groups, *, gamma=None, beta=None, eps=1e-5):
    """fp32 reference in the (N, 1, HW, C) layout; result in the input dtype."""
    orig = x.dtype
    xf = x.to(torch.float32)
    N, _, HW, C = xf.shape
    x_nchw = xf.squeeze(1).permute(0, 2, 1)  # (N, C, HW)
    w = gamma.to(torch.float32).reshape(C) if gamma is not None else None
    b = beta.to(torch.float32).reshape(C) if beta is not None else None
    y = torch.nn.functional.group_norm(x_nchw, num_groups, weight=w, bias=b, eps=eps)
    return y.permute(0, 2, 1).unsqueeze(1).to(orig)


def compute_pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 1.0 if torch.allclose(a, b) else 0.0
    return float((a * b).sum() / denom)


def _to_device(t, device, dtype, layout):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _run_case(
    device, shape, num_groups, *, layout, dtype=ttnn.bfloat16, affine="gamma_beta", eps=1e-5, mean_offset=0.0
):
    torch.manual_seed(42)
    N, one, HW, C = shape
    x = torch.randn(shape, dtype=torch.float32) + mean_offset
    x = x.to(TORCH_DTYPE[dtype])

    gamma = beta = None
    if affine in ("gamma_beta", "gamma_only"):
        gamma = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    if affine == "gamma_beta":
        beta = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)

    tt_x = _to_device(x, device, dtype, layout)
    tt_gamma = _to_device(gamma, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) if gamma is not None else None
    tt_beta = _to_device(beta, device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT) if beta is not None else None

    kwargs = {"eps": eps}
    if tt_gamma is not None:
        kwargs["gamma"] = tt_gamma
    if tt_beta is not None:
        kwargs["beta"] = tt_beta
    tt_y = groupnorm_sc_N_1_HW_C(tt_x, num_groups, **kwargs)

    # Output keeps the input's tensor spec.
    assert list(tt_y.shape) == list(shape), f"shape {tt_y.shape} != {shape}"
    assert tt_y.dtype == dtype, f"dtype {tt_y.dtype} != {dtype}"
    assert tt_y.layout == layout, f"layout {tt_y.layout} != {layout}"

    expected = torch_groupnorm_n_1_hw_c(x, num_groups, gamma=gamma, beta=beta, eps=eps)
    actual = ttnn.to_torch(tt_y)
    pcc = compute_pcc(actual.float(), expected.float())
    assert (
        pcc >= PCC_THRESHOLD[dtype]
    ), f"PCC {pcc:.6f} < {PCC_THRESHOLD[dtype]} for {shape} G={num_groups} {layout} {affine}"
    assert torch.isfinite(actual.float()).all(), "non-finite values in output"


# --------------------------------------------------------------------------- #
# Core matrix: shapes x layouts x affine variants (tile-aligned HW and C)
# --------------------------------------------------------------------------- #
SHAPES = [
    pytest.param((1, 1, 32, 32), 1, id="single_tile_G1"),
    pytest.param((1, 1, 64, 64), 2, id="2x2_tiles_G2"),
    pytest.param((1, 1, 128, 128), 4, id="multi_tile_square_G4"),
    pytest.param((1, 1, 256, 64), 2, id="tall_G2"),
    pytest.param((1, 1, 32, 512), 16, id="wide_G16"),
    pytest.param((2, 1, 64, 128), 4, id="batch2_G4"),
    pytest.param((4, 1, 128, 256), 8, id="batch4_G8"),
    pytest.param((1, 1, 64, 96), 3, id="odd_groups_G3"),
]

LAYOUTS = [
    pytest.param(ttnn.TILE_LAYOUT, id="tile"),
    pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm"),
]

AFFINE = [
    pytest.param("gamma_beta", id="gamma_beta"),
    pytest.param("gamma_only", id="gamma_only"),
    pytest.param("no_affine", id="no_affine"),
]


@pytest.mark.parametrize("shape,num_groups", SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("affine", AFFINE)
def test_groupnorm_matches_torch(device, shape, num_groups, layout, affine):
    _run_case(device, shape, num_groups, layout=layout, affine=affine)


# --------------------------------------------------------------------------- #
# Partial channels: (C / num_groups) % 32 != 0 — groups straddle tile boundaries
# --------------------------------------------------------------------------- #
STRADDLING_SHAPES = [
    pytest.param((1, 1, 32, 320), 32, id="sd_C320_CG10_hw32"),
    pytest.param((1, 1, 64, 320), 32, id="sd_C320_CG10_hw64"),
    pytest.param((1, 1, 1024, 640), 32, id="sd_C640_CG20"),
    pytest.param((1, 1, 256, 1280), 32, id="sd_C1280_CG40"),
    pytest.param((1, 1, 256, 1920), 32, id="sd_C1920_CG60"),
    pytest.param((1, 1, 64, 192), 8, id="C192_G8_CG24"),
    pytest.param((1, 1, 128, 448), 8, id="C448_G8_CG56"),
    pytest.param((2, 1, 256, 1280), 32, id="batch2_sd_C1280"),
]


@pytest.mark.parametrize("shape,num_groups", STRADDLING_SHAPES)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_group_straddling_channels(device, shape, num_groups, layout):
    _run_case(device, shape, num_groups, layout=layout, affine="gamma_beta")


# --------------------------------------------------------------------------- #
# Large-shape / wide-C cases (multi-core split along both HW and C)
# --------------------------------------------------------------------------- #
LARGE_SHAPES = [
    pytest.param((1, 1, 4096, 320), 32, id="sd15_stage0"),
    pytest.param((1, 1, 2048, 128), 4, id="tall_G4"),
    pytest.param((1, 1, 64, 4096), 32, id="wide_C4096_G32"),
    pytest.param((1, 1, 128, 2048), 32, id="wide_C2048_G32"),
]


@pytest.mark.parametrize("shape,num_groups", LARGE_SHAPES)
def test_large_shapes_tile(device, shape, num_groups):
    _run_case(device, shape, num_groups, layout=ttnn.TILE_LAYOUT, affine="gamma_beta")


# --------------------------------------------------------------------------- #
# More than 32 groups: two group-slot tiles (num_group_tiles = 2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("layout", LAYOUTS)
def test_more_than_32_groups(device, layout):
    _run_case(device, (1, 1, 64, 2048), 64, layout=layout, affine="gamma_beta")


# --------------------------------------------------------------------------- #
# Numerically stable variance: large |mean| relative to sigma
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 256, 320), 32, id="sd_C320_mean10"),
        pytest.param((2, 1, 128, 128), 4, id="batch2_mean10"),
    ],
)
def test_large_mean_stable_variance(device, shape, num_groups):
    # Activations after a residual add: |mean| ~ 10 sigma. A one-pass
    # E[x^2] - mean^2 with 16-bit accumulation loses the variance here.
    _run_case(device, shape, num_groups, layout=ttnn.TILE_LAYOUT, affine="gamma_beta", mean_offset=10.0)


# --------------------------------------------------------------------------- #
# Custom eps
# --------------------------------------------------------------------------- #
def test_custom_eps(device):
    _run_case(device, (1, 1, 128, 128), 4, layout=ttnn.TILE_LAYOUT, affine="gamma_beta", eps=1e-6)


# --------------------------------------------------------------------------- #
# Regime pin: force the streaming (non-resident) regime
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "shape,num_groups",
    [
        pytest.param((1, 1, 1024, 320), 32, id="stream_sd_C320"),
        pytest.param((2, 1, 256, 128), 4, id="stream_batch2"),
    ],
)
@pytest.mark.parametrize("layout", LAYOUTS)
def test_streaming_regime(device, monkeypatch, shape, num_groups, layout):
    from ttnn.operations.groupnorm_sc_N_1_HW_C import config

    monkeypatch.setattr(config, "FORCE_STREAMING", True)
    _run_case(device, shape, num_groups, layout=layout, affine="gamma_beta")


# --------------------------------------------------------------------------- #
# Argument validation (ValueError, not NotImplementedError)
# --------------------------------------------------------------------------- #
def test_argument_validation(device, expect_error):
    torch.manual_seed(42)
    x = _to_device(torch.randn(1, 1, 64, 64).to(torch.bfloat16), device, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    # C % num_groups != 0
    with expect_error(ValueError, "num_groups"):
        groupnorm_sc_N_1_HW_C(x, 3)

    # gamma shape mismatch
    bad_gamma = _to_device(torch.randn(1, 1, 1, 32).to(torch.bfloat16), device, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
    with expect_error(ValueError, "gamma"):
        groupnorm_sc_N_1_HW_C(x, 2, gamma=bad_gamma)

    # dim[1] != 1
    x_bad = _to_device(torch.randn(1, 2, 64, 64).to(torch.bfloat16), device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    with expect_error(ValueError, r"dim\[1\]"):
        groupnorm_sc_N_1_HW_C(x_bad, 2)

    # rank != 4
    x_3d = _to_device(torch.randn(1, 64, 64).to(torch.bfloat16), device, ttnn.bfloat16, ttnn.TILE_LAYOUT)
    with expect_error(ValueError, "4D"):
        groupnorm_sc_N_1_HW_C(x_3d, 2)

    # in_place needs an L1 shard to overwrite
    with expect_error(ValueError, "in_place"):
        groupnorm_sc_N_1_HW_C(x, 2, in_place=True)
