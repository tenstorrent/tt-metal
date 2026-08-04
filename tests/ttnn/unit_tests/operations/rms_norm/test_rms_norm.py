# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for rms_norm — IMMUTABLE.

This file is the specification. The implementer must NOT modify it.

RMSNorm(x) = x / sqrt(mean(x^2, dim=-1, keepdim=True) + epsilon) * gamma

What it pins:
  * both layouts natively (TILE and ROW_MAJOR), output layout == input layout,
    with no host-side to_layout / tilize / untilize / pad / slice anywhere;
  * non-tile-aligned H and/or W, including the pad-poison case that proves the
    RMS denominator counts only the W *logical* elements (see
    test_rms_norm_padding_is_not_folded_into_denominator);
  * both compute regimes from op_design.md section 4.2 (RESIDENT: narrow W,
    one resident block; STREAM: wide W, width-chunked) -- a regime that only
    triggers on some shapes/dtypes must not be able to hide;
  * gamma present / absent, and gamma at a dtype+layout independent of the
    input's (mixed-precision LLM convention);
  * the Phase 0 maxed-out precision corner, fp32_dest_acc_en=True.

PCC thresholds are keyed by dtype only, identical to the golden suite
(eval/golden_tests/rms_norm/helpers.py TOLERANCES) -- never tightened or
loosened per shape or per "op complexity".
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm

# Same thresholds as the golden suite.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}

# Phase 0 dtypes.
DTYPES = [ttnn.bfloat16, ttnn.float32]
LAYOUTS = [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    """Reference, computed in fp32 and returned in the input dtype."""
    original_dtype = x.dtype
    xf = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + epsilon)
    out = xf / rms
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out.to(original_dtype)


def _compute_config():
    """Phase 0 maxed-out precision corner."""
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


def _run(
    device,
    shape,
    *,
    dtype,
    layout,
    with_gamma=True,
    gamma_dtype=None,
    gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
    epsilon=1e-6,
    compute_kernel_config=None,
    poison_padding=None,
):
    """Build tensors, dispatch rms_norm, compare against torch. Returns (expected, actual)."""
    torch.manual_seed(42)
    torch_dtype = TORCH_DTYPE[dtype]
    torch_x = torch.randn(shape, dtype=torch_dtype)

    torch_gamma = None
    ttnn_gamma = None
    if with_gamma:
        gdtype = gamma_dtype if gamma_dtype is not None else dtype
        W = shape[-1]
        torch_gamma = torch.randn(W, dtype=TORCH_DTYPE[gdtype])
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=gdtype,
            layout=gamma_layout,
            device=device,
        )

    expected = torch_rms_norm(torch_x, gamma=torch_gamma, epsilon=epsilon)

    ttnn_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device)

    # Poison the implicit tile padding so an op that folds padding into the
    # reduction (or divides by the padded width) fails loudly instead of by a
    # sub-noise-floor margin. `expected` is built from the LOGICAL tensor and
    # never sees padding. TILE only -- ROW_MAJOR has no implicit tile padding.
    if poison_padding is not None and layout == ttnn.TILE_LAYOUT:
        ttnn_x = ttnn.fill_implicit_tile_padding(ttnn_x, poison_padding)
        if ttnn_gamma is not None and gamma_layout == ttnn.TILE_LAYOUT:
            ttnn_gamma = ttnn.fill_implicit_tile_padding(ttnn_gamma, poison_padding)

    ttnn_out = rms_norm(
        ttnn_x,
        gamma=ttnn_gamma,
        epsilon=epsilon,
        compute_kernel_config=compute_kernel_config or _compute_config(),
    )

    # Output layout must match the input layout -- no host-side conversion here.
    assert ttnn_out.layout == layout, f"output layout {ttnn_out.layout} != input layout {layout}"
    assert tuple(ttnn_out.shape) == tuple(shape), f"output shape {tuple(ttnn_out.shape)} != {tuple(shape)}"

    actual = ttnn.to_torch(ttnn_out)
    return expected.to(torch.float32), actual.to(torch.float32)


# ---------------------------------------------------------------------------
# Core shape / dtype / layout sweep
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        # --- tile-aligned ---
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((1, 1, 32, 256), id="non_square_wide"),
        pytest.param((1, 1, 256, 32), id="non_square_tall"),
        pytest.param((2, 4, 64, 128), id="multi_batch_4d"),
        pytest.param((4, 128, 512), id="rank3"),
        pytest.param((128, 512), id="rank2"),
        # --- W not a multiple of 32 (masked reduce path) ---
        pytest.param((1, 1, 32, 50), id="w_non_aligned_50"),
        pytest.param((1, 1, 64, 17), id="w_non_aligned_17"),
        pytest.param((2, 1, 128, 100), id="w_non_aligned_multi_batch"),
        # --- H not a multiple of 32 (row-padding path) ---
        pytest.param((1, 1, 17, 64), id="h_non_aligned_17"),
        pytest.param((1, 1, 50, 128), id="h_non_aligned_50"),
        # --- both non-aligned ---
        pytest.param((1, 1, 17, 50), id="both_non_aligned"),
        pytest.param((3, 1, 100, 47), id="both_non_aligned_multi_batch"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
def test_rms_norm(device, shape, dtype, layout):
    expected, actual = _run(device, shape, dtype=dtype, layout=layout)
    assert_with_pcc(expected, actual, pcc=PCC[dtype])


# ---------------------------------------------------------------------------
# gamma absent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 4, 32, 256), id="multi_batch"),
        pytest.param((1, 1, 32, 50), id="w_non_aligned"),
        pytest.param((1, 1, 17, 64), id="h_non_aligned"),
        pytest.param((128, 512), id="rank2"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
def test_rms_norm_no_gamma(device, shape, dtype, layout):
    expected, actual = _run(device, shape, dtype=dtype, layout=layout, with_gamma=False)
    assert_with_pcc(expected, actual, pcc=PCC[dtype])


# ---------------------------------------------------------------------------
# gamma format is independent of the input format (mixed precision)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 64, 128), id="aligned"),
        pytest.param((1, 1, 64, 100), id="w_non_aligned"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
@pytest.mark.parametrize("gamma_dtype", DTYPES, ids=["gamma_bf16", "gamma_fp32"])
@pytest.mark.parametrize("gamma_layout", LAYOUTS, ids=["gamma_tile", "gamma_row_major"])
def test_rms_norm_gamma_formats(device, shape, dtype, layout, gamma_dtype, gamma_layout):
    expected, actual = _run(
        device,
        shape,
        dtype=dtype,
        layout=layout,
        gamma_dtype=gamma_dtype,
        gamma_layout=gamma_layout,
    )
    # Precision is bounded by the coarser of the two operand dtypes.
    pcc = min(PCC[dtype], PCC[gamma_dtype])
    assert_with_pcc(expected, actual, pcc=pcc)


# ---------------------------------------------------------------------------
# Regime-pinned: op_design.md section 4.2 selects between a RESIDENT and a
# STREAM compute regime purely from (layout, dtype, gamma format, Wt). Both
# must be exercised explicitly, for every layout and dtype, with and without
# gamma -- otherwise a regime can pass on one device/config and fail on another.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, regime",
    [
        # RESIDENT: narrow W, whole row resident, coarse row blocks.
        pytest.param((1, 1, 64, 64), "resident", id="resident_W64"),
        pytest.param((1, 1, 128, 512), "resident", id="resident_W512"),
        pytest.param((1, 1, 64, 500), "resident", id="resident_W500_non_aligned"),
        # STREAM: W too wide for one resident tile-row -> width-chunked, x re-read.
        pytest.param((1, 1, 32, 4096), "stream", id="stream_W4096"),
        pytest.param((1, 1, 64, 4096), "stream", id="stream_W4096_2rows"),
        pytest.param((1, 1, 32, 4000), "stream", id="stream_W4000_non_aligned"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm_regimes(device, shape, regime, dtype, layout, with_gamma):
    expected, actual = _run(device, shape, dtype=dtype, layout=layout, with_gamma=with_gamma)
    assert_with_pcc(expected, actual, pcc=PCC[dtype])


# ---------------------------------------------------------------------------
# The RMS denominator must reflect ONLY the valid (non-padding) elements.
#
# These widths are SMALL, so one tile of padding is 11-38% of the row: an op
# that folds the padding into the reduction, or that divides by the padded
# width Wt*32 instead of W, is wrong by 6-27% -- far above bf16's own 0.39%
# quantization step. Poisoning the padding with a loud value turns a leak from
# a near-uniform scale error (which PCC is largely blind to) into a
# catastrophic one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 40), id="W40_pad24_37pct"),
        pytest.param((1, 1, 32, 72), id="W72_pad24_25pct"),
        pytest.param((1, 1, 32, 136), id="W136_pad24_15pct"),
        pytest.param((1, 1, 32, 200), id="W200_pad24_11pct"),
        pytest.param((1, 1, 224, 72), id="W72_many_rows"),
        pytest.param((1, 1, 40, 40), id="H40_W40_both_padded"),
    ],
)
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_rms_norm_padding_is_not_folded_into_denominator(device, shape, dtype, with_gamma):
    expected, actual = _run(
        device,
        shape,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        with_gamma=with_gamma,
        gamma_layout=ttnn.TILE_LAYOUT,
        poison_padding=1000.0,
    )
    assert_with_pcc(expected, actual, pcc=PCC[dtype])


# ---------------------------------------------------------------------------
# epsilon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("epsilon", [1e-6, 1e-5, 1e-2], ids=["eps1e-6", "eps1e-5", "eps1e-2"])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
def test_rms_norm_epsilon(device, epsilon, dtype, layout):
    expected, actual = _run(device, (1, 1, 64, 128), dtype=dtype, layout=layout, epsilon=epsilon)
    assert_with_pcc(expected, actual, pcc=PCC[dtype])


# ---------------------------------------------------------------------------
# Compute-config handling: None resolves through the op's exported default
# factory, and an explicitly maxed-out config is honoured, not overridden.
# math_fidelity is NOT gated -- any value must be accepted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp32"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["tile", "row_major"])
def test_rms_norm_default_compute_config(device, dtype, layout):
    """compute_kernel_config=None must resolve through default_compute_kernel_config()."""
    from ttnn.operations.rms_norm import default_compute_kernel_config

    assert default_compute_kernel_config().fp32_dest_acc_en is True

    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    torch_x = torch.randn(shape, dtype=TORCH_DTYPE[dtype])
    expected = torch_rms_norm(torch_x)

    ttnn_x = ttnn.from_torch(torch_x, dtype=dtype, layout=layout, device=device)
    ttnn_out = rms_norm(ttnn_x)  # no compute_kernel_config

    assert ttnn_out.layout == layout
    assert_with_pcc(expected.to(torch.float32), ttnn.to_torch(ttnn_out).to(torch.float32), pcc=PCC[dtype])


@pytest.mark.parametrize(
    "math_fidelity",
    [ttnn.MathFidelity.LoFi, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.HiFi4],
    ids=["LoFi", "HiFi2", "HiFi4"],
)
def test_rms_norm_math_fidelity_is_not_gated(device, math_fidelity):
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = math_fidelity
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False

    expected, actual = _run(
        device,
        (1, 1, 64, 128),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        compute_kernel_config=cfg,
    )
    assert_with_pcc(expected, actual, pcc=PCC[ttnn.bfloat16])


# ---------------------------------------------------------------------------
# Validation. The error message must name the offending concept ("rank" /
# "gamma") so CI log triage can attribute it -- see op_design.md section 9.1.
# ---------------------------------------------------------------------------


def test_rms_norm_rejects_rank_1(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn(64, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error((ValueError, RuntimeError), r"(?i)rank"):
        rms_norm(x)


def test_rms_norm_rejects_gamma_width_mismatch(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(
        torch.randn((1, 1, 32, 128), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    gamma = ttnn.from_torch(
        torch.randn((1, 1, 1, 64), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error((ValueError, RuntimeError), r"(?i)gamma"):
        rms_norm(x, gamma=gamma)
