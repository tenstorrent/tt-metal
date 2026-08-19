# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Acceptance test for ttnn.operations.rms_norm.

IMMUTABLE SPEC — do not modify this file to make an implementation pass.

Covers, per ttnn/ttnn/operations/rms_norm/op_design.md:
  * single-tile, multi-tile, non-square, multi-batch, wide-hidden shapes
  * ROW_MAJOR_LAYOUT and TILE_LAYOUT (output layout must match input layout)
  * non-tile-aligned H and/or W, natively (no host-side pad/slice/to_layout)
  * gamma present / absent, gamma dtype and layout independent of the input's
  * custom epsilon
  * the maxed-out precision corner (fp32_dest_acc_en=True) and an explicit
    ttnn.ComputeConfigDescriptor pass-through
  * the four regime-pinned cases named in op_design.md "Regime selection"
"""

import pytest
import torch

import ttnn
from ttnn.operations.rms_norm import rms_norm
from tests.ttnn.utils_for_testing import assert_with_pcc

# Same thresholds as the golden suite (eval/golden_tests/rms_norm/helpers.py::TOLERANCES).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}


def torch_rms_norm(x, gamma=None, epsilon=1e-6):
    """Reference implementation, evaluated in fp32."""
    x32 = x.to(torch.float32)
    out = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    if gamma is not None:
        out = out * gamma.to(torch.float32)
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
    gamma_layout=None,
    epsilon=1e-6,
    compute_kernel_config=None,
):
    torch.manual_seed(42)

    W = shape[-1]
    torch_x = torch.randn(shape, dtype=torch.float32)
    torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32) if with_gamma else None

    tt_x = _to_device(torch_x, device, dtype, layout)
    tt_gamma = None
    if with_gamma:
        tt_gamma = _to_device(
            torch_gamma,
            device,
            gamma_dtype if gamma_dtype is not None else dtype,
            gamma_layout if gamma_layout is not None else ttnn.ROW_MAJOR_LAYOUT,
        )

    kwargs = {"gamma": tt_gamma, "epsilon": epsilon}
    if compute_kernel_config is not None:
        kwargs["compute_kernel_config"] = compute_kernel_config

    tt_out = rms_norm(tt_x, **kwargs)

    # Output layout must match the input layout, with no host-side conversion.
    assert tt_out.layout == layout, f"expected output layout {layout}, got {tt_out.layout}"
    assert tuple(tt_out.shape) == tuple(shape), f"expected shape {tuple(shape)}, got {tuple(tt_out.shape)}"

    expected = torch_rms_norm(torch_x, torch_gamma, epsilon)
    actual = ttnn.to_torch(tt_out).to(torch.float32).reshape(expected.shape)

    assert_with_pcc(expected, actual, PCC[dtype])


# --------------------------------------------------------------------------- #
# Core shape / layout / dtype sweep
# --------------------------------------------------------------------------- #

SHAPES = [
    (1, 1, 32, 32),  # single tile
    (1, 1, 64, 128),  # multi-tile, non-square
    (4, 8, 32, 256),  # multi-batch
    (2, 4, 128, 512),  # multi-batch, multi-tile
    (1, 1, 32, 4096),  # wide hidden (LLM-realistic)
    (128, 512),  # rank 2
    (4, 128, 512),  # rank 3
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_rms_norm(device, shape, dtype, layout):
    _run(device, shape, dtype=dtype, layout=layout)


# --------------------------------------------------------------------------- #
# Non-tile-aligned shapes — must be handled natively, in BOTH layouts.
# The RMS denominator must reflect only valid (non-padding) elements.
# --------------------------------------------------------------------------- #

NON_ALIGNED_SHAPES = [
    (1, 1, 32, 50),  # W non-aligned
    (1, 1, 64, 17),  # W non-aligned, narrow
    (1, 1, 17, 64),  # H non-aligned
    (1, 1, 50, 128),  # H non-aligned
    (1, 1, 17, 50),  # both non-aligned
    (2, 1, 100, 47),  # both non-aligned, multi-batch
    (1, 1, 32, 72),  # narrow W: one tile of padding is 25% of the row
    (1, 1, 32, 40),  # narrow W: one tile of padding is 37.5% of the row
    (32, 17),  # rank 2, smallest regime
]


@pytest.mark.parametrize("shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_rms_norm_non_tile_aligned(device, shape, layout):
    _run(device, shape, dtype=ttnn.bfloat16, layout=layout)


# --------------------------------------------------------------------------- #
# gamma: presence, dtype, layout
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", [(1, 1, 32, 64), (2, 4, 128, 512), (1, 1, 32, 50)])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_rms_norm_no_gamma(device, shape, layout):
    _run(device, shape, dtype=ttnn.bfloat16, layout=layout, with_gamma=False)


@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 1, 32, 47)])
@pytest.mark.parametrize("gamma_layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
@pytest.mark.parametrize("gamma_dtype", [ttnn.bfloat16, ttnn.float32])
def test_rms_norm_gamma_format(device, shape, gamma_layout, gamma_dtype):
    """gamma's dtype/layout are independent of the input's (mixed-precision LLM case)."""
    _run(
        device,
        shape,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        gamma_dtype=gamma_dtype,
        gamma_layout=gamma_layout,
    )


# --------------------------------------------------------------------------- #
# epsilon and compute_kernel_config
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("epsilon", [1e-6, 1e-5, 1e-2])
def test_rms_norm_epsilon(device, epsilon):
    _run(device, (1, 1, 64, 128), epsilon=epsilon)


def test_rms_norm_default_epsilon_and_no_kwargs(device):
    """rms_norm(x) — no gamma, default epsilon, default compute config."""
    _run(device, (1, 1, 64, 128), with_gamma=False)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_rms_norm_maxed_out_compute_config(device, dtype):
    """The Phase 0 precision corner, passed explicitly by the caller."""
    _run(
        device,
        (1, 1, 64, 128),
        dtype=dtype,
        compute_kernel_config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
        ),
    )


def test_rms_norm_default_compute_kernel_config_is_a_factory(device):
    """references/precision_convention.md: one exported factory, fresh descriptor per call."""
    from ttnn.operations.rms_norm import default_compute_kernel_config

    a = default_compute_kernel_config()
    b = default_compute_kernel_config()
    assert a is not b, "default_compute_kernel_config() must be a factory, not a shared constant"
    assert a.fp32_dest_acc_en is True


# --------------------------------------------------------------------------- #
# Regime-pinned cases (op_design.md -> "Mandatory regime-pinned tests").
# A regime that only triggers on some grids/L1 budgets can pass on one device
# and fail on another, so each regime gets an explicitly named case.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "shape,layout,regime",
    [
        ((1, 1, 64, 128), ttnn.TILE_LAYOUT, "A: aligned TILE, working set fits"),
        ((1, 1, 32, 50), ttnn.ROW_MAJOR_LAYOUT, "A: RM non-aligned W, pad zero-filled by reader"),
        ((1, 1, 32, 72), ttnn.TILE_LAYOUT, "B: TILE non-aligned W, masked reduce mandatory"),
        ((1, 1, 32, 16384), ttnn.TILE_LAYOUT, "B: aligned TILE, working set exceeds L1"),
    ],
)
def test_rms_norm_regimes(device, shape, layout, regime):
    _run(device, shape, dtype=ttnn.bfloat16, layout=layout)


# --------------------------------------------------------------------------- #
# Validation. The error text is part of the contract (op_design.md -> Validation):
#   rank < 2                    -> message contains "rank"
#   gamma last-dim mismatch     -> message contains "gamma"
# --------------------------------------------------------------------------- #


def test_rms_norm_rejects_rank_1(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(torch.randn((64,)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error((ValueError, RuntimeError), "(?i)rank"):
        rms_norm(x)


def test_rms_norm_rejects_gamma_last_dim_mismatch(device, expect_error):
    torch.manual_seed(42)
    x = ttnn.from_torch(torch.randn((1, 1, 32, 64)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    gamma = ttnn.from_torch(
        torch.randn((1, 1, 1, 32)), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    with expect_error((ValueError, RuntimeError), "(?i)gamma"):
        rms_norm(x, gamma=gamma)
