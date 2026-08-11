# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for rms_norm — the immutable specification.

    RMSNorm(x) = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps) * gamma

This file is the spec. The implementer MUST NOT modify it.

What it pins, beyond basic numerics:

* **Both layouts natively.** TILE and ROW_MAJOR inputs go straight to the op;
  the test never calls to_layout / tilize / pad / slice to help it, and the
  output layout must match the input layout.
* **Non-tile-aligned H and W.** The RMS denominator must reflect only the
  valid elements of the reduced dimension. The `poison` tests fill the
  implicit tile padding with a large finite value, so an op that folds
  padding into the reduction fails loudly instead of being masked by PCC.
* **Both regimes of the work split.** `test_rms_norm_regime_pinned` contains
  shapes that can only land in the row-parallel regime and shapes that can
  only land in the cross-core hidden-split regime on *any* device grid — a
  regime selected from the grid size otherwise passes on one board and fails
  on another.
* **The maxed-out precision corner.** fp32_dest_acc_en=True is Phase 0, and
  a caller-supplied ComputeConfigDescriptor must be honoured, not ignored.
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.rms_norm import rms_norm


# Same thresholds as the golden suite. Do not tighten these per-shape.
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def torch_rms_norm(x, gamma=None, eps=1e-6):
    """Reference. Always computed in fp32 regardless of the device dtype."""
    x = x.float()
    denom = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x * denom
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out


def _to_device(t, device, dtype, layout):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device)


def _run(
    device,
    shape,
    *,
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    with_gamma=True,
    gamma_dtype=None,
    gamma_layout=ttnn.ROW_MAJOR_LAYOUT,
    eps=1e-6,
    compute_kernel_config=None,
    poison_padding=None,
    pcc=None,
):
    torch.manual_seed(42)
    torch_dtype = TORCH_DTYPE[dtype]
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.float32).to(torch_dtype)
    torch_gamma = None
    if with_gamma:
        gamma_shape = (1,) * (len(shape) - 1) + (W,)
        torch_gamma = torch.randn(gamma_shape, dtype=torch.float32).to(TORCH_DTYPE[gamma_dtype or dtype])

    expected = torch_rms_norm(torch_input, torch_gamma, eps)

    tt_input = _to_device(torch_input, device, dtype, layout)
    if poison_padding is not None:
        # Fill the implicit tile padding with a large finite value. A correct op
        # masks it out of the reduction; an incorrect one is off by a large,
        # shape-dependent factor.
        tt_input = ttnn.fill_implicit_tile_padding(tt_input, poison_padding)

    tt_gamma = None
    if with_gamma:
        tt_gamma = _to_device(torch_gamma, device, gamma_dtype or dtype, gamma_layout)

    kwargs = {"gamma": tt_gamma, "epsilon": eps}
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    tt_output = rms_norm(tt_input, **kwargs)

    assert tt_output.layout == tt_input.layout, "output layout must match input layout"
    assert tuple(tt_output.shape) == tuple(shape), "output shape must match input shape"

    actual = ttnn.to_torch(tt_output).float()
    assert_with_pcc(expected, actual, pcc if pcc is not None else PCC[dtype])
    return actual


# ---------------------------------------------------------------------------
# Core numerics: shapes x layouts x dtypes x gamma
# ---------------------------------------------------------------------------

SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 64, 128), id="multi_tile"),
    pytest.param((1, 1, 128, 512), id="non_square"),
    pytest.param((2, 4, 96, 256), id="multi_batch"),
    pytest.param((1, 1, 32, 4096), id="wide_hidden"),
    pytest.param((512, 64), id="rank2_tall"),
    pytest.param((4, 128, 320), id="rank3"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
def test_rms_norm(device, shape, layout, dtype):
    _run(device, shape, dtype=dtype, layout=layout)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_rms_norm_no_gamma(device, shape, layout):
    _run(device, shape, layout=layout, with_gamma=False)


# ---------------------------------------------------------------------------
# Non-tile-aligned shapes — native, no host-side padding or slicing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 47), id="w_non_aligned"),
        pytest.param((1, 1, 64, 17), id="w_non_aligned_narrow"),
        pytest.param((1, 1, 17, 64), id="h_non_aligned"),
        pytest.param((1, 1, 50, 128), id="h_non_aligned_wide"),
        pytest.param((1, 1, 17, 50), id="both_non_aligned"),
        pytest.param((2, 1, 100, 1023), id="both_non_aligned_large"),
        pytest.param((100, 47), id="rank2_both_non_aligned"),
    ],
)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "row_major"])
def test_rms_norm_non_tile_aligned(device, shape, layout):
    _run(device, shape, layout=layout)


@pytest.mark.parametrize(
    "shape",
    [
        # W is small, so one tile of padding is 11-38% of the row: a reduction
        # that folds padding in is wrong by 6-27%, far outside any PCC slack.
        pytest.param((1, 1, 32, 40), id="Wt2_pad37pct"),
        pytest.param((1, 1, 32, 72), id="Wt3_pad25pct"),
        pytest.param((1, 1, 32, 200), id="Wt7_pad11pct"),
        pytest.param((1, 1, 224, 72), id="many_rows_tiny_W"),
        pytest.param((1, 1, 40, 40), id="H_and_W_both_padded"),
    ],
)
def test_rms_norm_padding_is_masked(device, shape):
    """Poisoned tile padding must not reach the RMS denominator."""
    _run(device, shape, layout=ttnn.TILE_LAYOUT, poison_padding=1000.0)


# ---------------------------------------------------------------------------
# Regime-pinned: the work split is chosen from the device grid, so both
# branches must be exercised by construction rather than by luck.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape,regime",
    [
        # One tile-row: the independent (row) axis cannot fill any grid, so the
        # hidden axis MUST be split across cores and partials combined.
        pytest.param((1, 1, 32, 2048), "hidden_split", id="R1t_W2048_hidden_split"),
        pytest.param((1, 1, 32, 8192), "hidden_split", id="R1t_W8192_hidden_split"),
        pytest.param((1, 1, 32, 16384), "hidden_split", id="R1t_W16384_hidden_split"),
        pytest.param((1, 1, 64, 12288), "hidden_split", id="R2t_W12288_hidden_split"),
        # Many tile-rows, narrow hidden: the row axis over-fills any grid many
        # times over and the whole hidden slice fits one core, so the split is
        # row-parallel with no cross-core combine.
        pytest.param((1, 1, 8192, 64), "row_parallel", id="R256t_W64_row_parallel"),
        pytest.param((1, 1, 4096, 128), "row_parallel", id="R128t_W128_row_parallel"),
    ],
)
def test_rms_norm_regime_pinned(device, shape, regime):
    _run(device, shape, layout=ttnn.TILE_LAYOUT)


# ---------------------------------------------------------------------------
# Parameters: epsilon, gamma format, compute config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eps", [1e-6, 1e-5, 1e-2], ids=["eps_1e-6", "eps_1e-5", "eps_1e-2"])
def test_rms_norm_epsilon(device, eps):
    _run(device, (1, 1, 64, 256), eps=eps)


@pytest.mark.parametrize(
    "gamma_dtype,gamma_layout",
    [
        pytest.param(ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, id="gamma_bf16_rm"),
        pytest.param(ttnn.bfloat16, ttnn.TILE_LAYOUT, id="gamma_bf16_tile"),
        pytest.param(ttnn.float32, ttnn.ROW_MAJOR_LAYOUT, id="gamma_fp32_rm"),
        pytest.param(ttnn.float32, ttnn.TILE_LAYOUT, id="gamma_fp32_tile"),
    ],
)
def test_rms_norm_gamma_formats(device, gamma_dtype, gamma_layout):
    """gamma may be at a different dtype/layout than the activation."""
    _run(device, (1, 1, 64, 256), dtype=ttnn.bfloat16, gamma_dtype=gamma_dtype, gamma_layout=gamma_layout)


def test_rms_norm_default_compute_kernel_config_is_exported():
    from ttnn.operations.rms_norm import default_compute_kernel_config

    a = default_compute_kernel_config()
    b = default_compute_kernel_config()
    assert a is not b, "must be a factory, not a shared mutable constant"
    assert a.fp32_dest_acc_en is True, "Phase 0 is the maxed-out precision corner"


@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi])
def test_rms_norm_honours_compute_kernel_config(device, math_fidelity):
    """math_fidelity is never gated — any value is accepted and honoured."""
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=math_fidelity, fp32_dest_acc_en=True)
    _run(device, (1, 1, 64, 256), compute_kernel_config=cfg)


def test_rms_norm_maxed_out_precision_corner(device):
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    _run(device, (1, 1, 128, 1024), dtype=ttnn.float32, compute_kernel_config=cfg)


# ---------------------------------------------------------------------------
# Validation
#
# The regexes below are the contract on the error text: each rejection must
# name the thing it is rejecting, so a caller can tell the three apart.
# ---------------------------------------------------------------------------


def test_rms_norm_rejects_rank1(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(torch.randn((64,)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), "(?i)rank"):
        rms_norm(x)


def test_rms_norm_rejects_gamma_width_mismatch(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    g = ttnn.from_torch(torch.randn((1, 1, 1, 128)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), "(?i)gamma"):
        rms_norm(x, gamma=g)


def test_rms_norm_rejects_fp32_without_fp32_dest_acc(device, expect_error):
    """float32 input with fp32_dest_acc_en=False is refused natively."""
    torch.manual_seed(42)
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    cfg = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False)
    with expect_error((ValueError, RuntimeError), "(?i)fp32_dest_acc_en"):
        rms_norm(x, compute_kernel_config=cfg)
