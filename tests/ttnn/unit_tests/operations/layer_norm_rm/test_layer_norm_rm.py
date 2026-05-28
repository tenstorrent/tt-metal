# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for layer_norm_rm — the immutable Phase-0 spec.

DO NOT modify this file from the implementer side. It is the contract
against which the operation is judged. New axes (other dtypes, sharded
inputs, non-tile-aligned shapes, …) belong in the golden-tests suite
under `eval/golden_tests/layer_norm_rm/`, not here.

What it tests:
- layer_norm over the last dim of a ROW_MAJOR fp32 tensor, with
  optional gamma scale and optional beta shift.
- Output shape, dtype (fp32), and layout (ROW_MAJOR) preserved.
- Numerical agreement vs torch.nn.functional.layer_norm at PCC >= 0.999
  (fp32 tolerance keyed on the dtype, matching the golden-suite policy).
- All four call patterns specified in the operation contract:
    layer_norm(input)
    layer_norm(input, gamma)
    layer_norm(input, gamma, beta)
    layer_norm(input, gamma, beta, epsilon=X)

Phase 0 envelope (tested here):
- dtype = float32, layout = ROW_MAJOR, last two dims tile-aligned (mult of 32).
- Rank 4 only (the simplest shape arrangement; rank 2/3 are golden-suite axes).
- W <= 1024 (the L1-fits-in-one-tile-row budget — see op_design.md).
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from ttnn.operations.layer_norm import layer_norm


# ---------------------------------------------------------------------------
# Tolerances — same per-dtype thresholds as the golden suite.
# fp32 PCC threshold = 0.999; do NOT tighten based on op "complexity".
# ---------------------------------------------------------------------------

_PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two flattened tensors."""
    af = a.float().flatten()
    bf = b.float().flatten()
    # corrcoef yields NaN if either input is constant; guard against
    # exact-zero variance (e.g. all-equal input rows produce a constant
    # normalized output of zero on the normalize step — PCC is undefined,
    # but a pointwise allclose at high tolerance is fine).
    if af.std() == 0 and bf.std() == 0:
        return 1.0
    return torch.corrcoef(torch.stack([af, bf]))[0, 1].item()


def _layer_norm_torch(
    x: torch.Tensor,
    gamma: torch.Tensor | None,
    beta: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    """fp32 reference. gamma/beta are reshaped to 1D for the torch API."""
    g = gamma.reshape(-1) if gamma is not None else None
    b = beta.reshape(-1) if beta is not None else None
    return torch.nn.functional.layer_norm(
        x.float(),
        normalized_shape=(x.shape[-1],),
        weight=g,
        bias=b,
        eps=epsilon,
    )


# ---------------------------------------------------------------------------
# Shapes — minimum 4: single-tile, multi-tile, non-square, multi-batch.
# All rank-4, last two dims tile-aligned (mult of 32), W <= 1024.
# ---------------------------------------------------------------------------

SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile_32x32"),
    pytest.param((1, 1, 64, 128), id="multi_tile_64x128"),
    pytest.param((1, 1, 32, 256), id="non_square_32x256"),
    pytest.param((2, 4, 64, 64), id="multi_batch_2x4x64x64"),
    pytest.param((1, 1, 128, 512), id="taller_128x512"),
    # Phase 0 caps W at 512 — the W=1024 + gamma+beta configuration
    # overshoots per-core L1 (1.7 MB vs. the 1.5 MB budget). The wider-W
    # path is the W-axis chunking refinement; see op_requirements.md.
    pytest.param((4, 1, 32, 512), id="widest_in_budget_4x1x32x512"),
]


# ---------------------------------------------------------------------------
# Call patterns — exercise the four documented signatures.
# ---------------------------------------------------------------------------

AFFINE_MODES = [
    pytest.param("no_affine", id="no_affine"),
    pytest.param("gamma_only", id="gamma_only"),
    pytest.param("gamma_beta", id="gamma_beta"),
    pytest.param("custom_eps", id="custom_eps"),  # full affine + non-default epsilon
]


def _build_affine(
    mode: str,
    W: int,
    device,
) -> tuple[torch.Tensor | None, torch.Tensor | None, ttnn.Tensor | None, ttnn.Tensor | None, float]:
    """Return (torch_gamma, torch_beta, ttnn_gamma, ttnn_beta, epsilon)."""
    epsilon = 1e-5
    torch_gamma, torch_beta = None, None
    ttnn_gamma, ttnn_beta = None, None

    if mode in ("gamma_only", "gamma_beta", "custom_eps"):
        torch_gamma = torch.randn(W, dtype=torch.float32)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if mode in ("gamma_beta", "custom_eps"):
        torch_beta = torch.randn(W, dtype=torch.float32)
        ttnn_beta = ttnn.from_torch(
            torch_beta.reshape(1, 1, 1, W),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if mode == "custom_eps":
        epsilon = 7.5e-4
    return torch_gamma, torch_beta, ttnn_gamma, ttnn_beta, epsilon


# ---------------------------------------------------------------------------
# The single parametrized acceptance test.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
@pytest.mark.parametrize("shape", SHAPES)
def test_layer_norm_rm(device, shape, affine_mode):
    torch.manual_seed(42)

    # --- Build input ---
    torch_input = torch.randn(shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # --- Build optional affine tensors ---
    W = shape[-1]
    torch_gamma, torch_beta, ttnn_gamma, ttnn_beta, epsilon = _build_affine(affine_mode, W, device)

    # --- Reference (fp32) ---
    expected = _layer_norm_torch(torch_input, torch_gamma, torch_beta, epsilon)

    # --- Dispatch — exercise the documented signature variants ---
    if affine_mode == "no_affine":
        ttnn_output = layer_norm(ttnn_input)
    elif affine_mode == "gamma_only":
        ttnn_output = layer_norm(ttnn_input, ttnn_gamma)
    elif affine_mode == "gamma_beta":
        ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)
    elif affine_mode == "custom_eps":
        ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta, epsilon=epsilon)
    else:
        raise AssertionError(f"unknown affine_mode {affine_mode!r}")

    # --- Shape / dtype / layout invariants ---
    assert list(ttnn_output.shape) == list(
        shape
    ), f"output shape {list(ttnn_output.shape)} != input shape {list(shape)}"
    assert ttnn_output.dtype == ttnn.float32, f"output dtype {ttnn_output.dtype} != ttnn.float32"
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT, f"output layout {ttnn_output.layout} != ROW_MAJOR_LAYOUT"

    # --- Numerical agreement ---
    torch_output = ttnn.to_torch(ttnn_output)
    pcc_threshold = _PCC_BY_DTYPE[ttnn.float32]
    pcc = _pcc(torch_output, expected)
    assert pcc >= pcc_threshold, (
        f"PCC {pcc:.6f} below threshold {pcc_threshold} for "
        f"shape={shape}, affine_mode={affine_mode}, epsilon={epsilon}\n"
        f"  max abs diff = {(torch_output - expected).abs().max().item():.6f}\n"
        f"  output[0,0,0,:4]   = {torch_output.flatten()[:4].tolist()}\n"
        f"  expected[0,0,0,:4] = {expected.flatten()[:4].tolist()}"
    )


# ---------------------------------------------------------------------------
# Per-row mean ≈ 0 / per-row var ≈ 1 sanity check on the no-affine path.
# Once gamma/beta are applied this is no longer true, so the assertion is
# only meaningful on the affine-free output.
# ---------------------------------------------------------------------------


def test_layer_norm_rm_normalization_invariants(device):
    torch.manual_seed(42)
    shape = (1, 1, 64, 128)

    torch_input = torch.randn(shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input)
    torch_output = ttnn.to_torch(ttnn_output)

    # Per-last-dim mean should be ~0 and unbiased var ~ 1 (within fp32
    # numerical tolerance; layer_norm uses population variance so the
    # sample-var of the normalized output is slightly > 1 — use atol).
    row_mean = torch_output.mean(dim=-1)
    row_var = torch_output.var(dim=-1, unbiased=False)

    assert torch.allclose(
        row_mean, torch.zeros_like(row_mean), atol=1e-3
    ), f"per-row mean not ~0: max abs = {row_mean.abs().max().item():.6f}"
    assert torch.allclose(row_var, torch.ones_like(row_var), atol=5e-3), (
        f"per-row var not ~1: max abs diff = " f"{(row_var - 1).abs().max().item():.6f}"
    )
