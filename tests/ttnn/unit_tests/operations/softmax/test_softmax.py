# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for ttnn.operations.softmax.softmax.

This file is the Phase-0 spec — DO NOT MODIFY when implementing.
The implementer's job is to make this pass against `torch.softmax`
across the parametrized matrix below.

Phase-0 envelope:
- dtype: float32
- layout: TILE_LAYOUT
- rank: 4 (N, C, H, W) with H % 32 == 0 and W % 32 == 0
- dim ∈ {-1, -2}
- numeric_stable ∈ {True, False}
- compute_kernel_config: None (entry point installs default) or
    ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True)

PCC threshold for fp32 acceptance is 0.999 (per the planner spec).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.softmax import softmax


# --------------------------------------------------------------------------
# Shapes — covers single-tile, multi-tile, non-square, multi-batch combos.
# All shapes are 4D and tile-aligned in H and W.
# --------------------------------------------------------------------------
SHAPES = [
    # single-tile per (N,C) slice
    (1, 1, 32, 32),
    # multi-tile, single (N,C)
    (1, 1, 64, 128),
    # non-square, multi-tile
    (1, 1, 128, 64),
    # multi-batch
    (2, 4, 32, 256),
    # larger H, square-ish
    (1, 2, 128, 128),
    # wider W (still tile-aligned)
    (1, 1, 32, 512),
]


# Phase-0 supported compute_kernel_config — both forms.
PHASE0_CONFIGS = [
    pytest.param(None, id="config=default(None)"),
    pytest.param(
        ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
        ),
        id="config=explicit_hifi4_fp32acc",
    ),
]


# --------------------------------------------------------------------------
# Main acceptance test — torch.softmax reference, PCC ≥ 0.999.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dim", [-1, -2])
@pytest.mark.parametrize("numeric_stable", [True, False])
def test_softmax_acceptance(device, shape, dim, numeric_stable):
    """Phase-0 acceptance: softmax(float32, TILE, 4D) == torch.softmax."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(ttnn_input, dim=dim, numeric_stable=numeric_stable)

    assert (
        ttnn_output.shape == ttnn_input.shape
    ), f"shape mismatch: got {ttnn_output.shape}, expected {ttnn_input.shape}"
    assert ttnn_output.dtype == ttnn.float32, f"dtype mismatch: got {ttnn_output.dtype}, expected {ttnn.float32}"
    assert (
        ttnn_output.layout == ttnn.TILE_LAYOUT
    ), f"layout mismatch: got {ttnn_output.layout}, expected {ttnn.TILE_LAYOUT}"

    torch_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_expected, torch_output, 0.999)

    # softmax should sum to ~1 along the reduce dim
    sum_along_dim = torch_output.sum(dim=dim)
    assert torch.allclose(
        sum_along_dim,
        torch.ones_like(sum_along_dim),
        atol=1e-3,
        rtol=1e-3,
    ), (
        f"softmax rows do not sum to 1 (max abs deviation = " f"{(sum_along_dim - 1).abs().max().item():.3e})"
    )


# --------------------------------------------------------------------------
# Compute-kernel-config acceptance — both None and the explicit Phase-0
# descriptor must succeed. Smaller matrix; one shape per (dim, stable) combo
# is enough.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("compute_kernel_config", PHASE0_CONFIGS)
@pytest.mark.parametrize("dim", [-1, -2])
def test_softmax_compute_kernel_config_accepted(device, compute_kernel_config, dim):
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=dim)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = softmax(
        ttnn_input,
        dim=dim,
        numeric_stable=True,
        compute_kernel_config=compute_kernel_config,
    )

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# Validation — every input outside the Phase-0 envelope must be rejected.
# `validate()` may raise NotImplementedError, ValueError, or RuntimeError;
# we accept any of those (per the planner spec).
# --------------------------------------------------------------------------
VALIDATION_ERRORS = (NotImplementedError, ValueError, RuntimeError)


def _make_input(device, shape=(1, 1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT):
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32).to(
        {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}.get(dtype, torch.float32)
    )
    return ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.parametrize("bad_dim", [-3, -4, 0, 1, 2, 3])
def test_softmax_rejects_unsupported_dim(device, bad_dim):
    ttnn_input = _make_input(device)
    with pytest.raises(VALIDATION_ERRORS):
        softmax(ttnn_input, dim=bad_dim)


def test_softmax_rejects_non_tile_aligned_h(device):
    # H = 17 (not divisible by 32) — must be rejected by validate().
    # NOTE: ttnn.from_torch will pad to a tile boundary; we encode the
    # logical shape via a 4D tensor whose H dimension is 17.
    torch_input = torch.randn((1, 1, 17, 64), dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(VALIDATION_ERRORS):
        softmax(ttnn_input, dim=-1)


def test_softmax_rejects_non_tile_aligned_w(device):
    torch_input = torch.randn((1, 1, 32, 50), dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(VALIDATION_ERRORS):
        softmax(ttnn_input, dim=-1)


def test_softmax_rejects_bfloat16_with_unsupported_config(device):
    """bf16 input was the Phase-0 rejection case; Refinement 2 adds the four
    bf16 precision names. The op still rejects bf16 paired with a (fidelity,
    accumulator) combo not in PRECISION_CONFIG (e.g. HiFi3 — bf16 modes ship
    only at HiFi2 and HiFi4)."""
    ttnn_input = _make_input(device, dtype=ttnn.bfloat16)
    bad_config = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi3,
        fp32_dest_acc_en=True,
    )
    with pytest.raises(VALIDATION_ERRORS):
        softmax(ttnn_input, dim=-1, compute_kernel_config=bad_config)


def test_softmax_accepts_row_major_layout(device):
    """Refinement 3: ROW_MAJOR_LAYOUT is now in SUPPORTED["layout"]. The
    entry-point converts to TILE internally and converts back on the way
    out, preserving the user's layout end-to-end."""
    torch_input = torch.randn((1, 1, 32, 32), dtype=torch.float32)
    torch_expected = torch.softmax(torch_input, dim=-1)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn_output = softmax(ttnn_input, dim=-1)
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    assert tuple(ttnn_output.shape) == (1, 1, 32, 32)
    assert_with_pcc(torch_expected, ttnn.to_torch(ttnn_output), 0.999)


@pytest.mark.parametrize(
    "bad_shape",
    [
        (1, 1, 1, 32, 32),  # rank 5 — still outside SUPPORTED["rank"]
        (1, 1, 1, 1, 32, 32),  # rank 6 — same
    ],
)
def test_softmax_rejects_unsupported_rank(device, bad_shape):
    """Rank 2/3/4 are supported by Refinement 3 (rank canonicalisation);
    higher ranks stay out of SUPPORTED["rank"]. Rank 0/1 are not modelled
    here — softmax needs at least an H and W axis for the (-2, -1) dim
    choices we accept, and the test universe (feature_spec.py:INPUTS)
    never emits rank < 2 shapes."""
    torch_input = torch.randn(bad_shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    with pytest.raises(VALIDATION_ERRORS):
        softmax(ttnn_input, dim=-1)


@pytest.mark.parametrize(
    "bad_config",
    [
        pytest.param(
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                fp32_dest_acc_en=True,
            ),
            id="HiFi2_rejected",
        ),
        pytest.param(
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=False,
            ),
            id="fp32_dest_acc_en=False_rejected",
        ),
        pytest.param(
            ttnn.ComputeConfigDescriptor(
                math_fidelity=ttnn.MathFidelity.LoFi,
                fp32_dest_acc_en=True,
            ),
            id="LoFi_rejected",
        ),
    ],
)
def test_softmax_rejects_unsupported_compute_kernel_config(device, bad_config):
    ttnn_input = _make_input(device)
    with pytest.raises(VALIDATION_ERRORS):
        softmax(ttnn_input, dim=-1, compute_kernel_config=bad_config)
