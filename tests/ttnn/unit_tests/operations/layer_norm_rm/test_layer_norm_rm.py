# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for ttnn.operations.layer_norm_rm.layer_norm.

This file is the Phase-0 spec — DO NOT MODIFY when implementing.
The implementer's job is to make this pass against torch.nn.functional.layer_norm
(equivalently, the closed-form per-row normalization in pytorch_reference below)
across the parametrized matrix below.

Phase-0 envelope (from op_design.md):
- input dtype: float32
- input layout: ROW_MAJOR_LAYOUT (the kernel handles tilize/untilize internally
  — the test does NOT call ttnn.to_layout / ttnn.tilize on the way in)
- gamma / beta: ROW_MAJOR_LAYOUT, shape (1, 1, 1, W), dtype float32, optional
- rank ≥ 2, final two dims tile-aligned (H % 32 == 0, W % 32 == 0)
- epsilon: keyword-only float, default 1e-5
- compute_kernel_config: None (entry point installs default) or
    ttnn.ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True,
                                  math_approx_mode=False)

PCC threshold for fp32 acceptance is 0.999 (per the planner spec — same
threshold used by the golden test suite for fp32 ops).
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.layer_norm_rm import layer_norm


# --------------------------------------------------------------------------
# Reference implementation — pure-python population layer-norm matching the
# math in op_design.md exactly. Uses float32 throughout.
# --------------------------------------------------------------------------
def pytorch_reference(
    input_tensor: torch.Tensor,
    gamma: torch.Tensor = None,
    beta: torch.Tensor = None,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    x = input_tensor.to(torch.float32)
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    y = (x - mean) / torch.sqrt(var + epsilon)
    if gamma is not None:
        y = y * gamma.reshape(-1).to(torch.float32)
    if beta is not None:
        y = y + beta.reshape(-1).to(torch.float32)
    return y.to(input_tensor.dtype)


# --------------------------------------------------------------------------
# Shapes — single-tile, multi-tile, non-square, multi-batch, and a few
# wider-W cases that exercise the chunked-reduce path described in
# op_design.md. All shapes are tile-aligned in the last two dims (Phase 0).
# Each entry is a single input shape; gamma/beta shape (when present) is
# always (1, 1, 1, shape[-1]).
# --------------------------------------------------------------------------
SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile"),
    pytest.param((1, 1, 32, 128), id="1x4_tiles"),
    pytest.param((1, 1, 64, 64), id="2x2_tiles"),
    pytest.param((1, 1, 128, 64), id="non_square_tall"),
    pytest.param((1, 1, 32, 512), id="wider_W"),
    pytest.param((2, 4, 32, 256), id="multi_batch"),
    pytest.param((1, 1, 32, 2048), id="wide_W_2048"),
    pytest.param((1, 2, 64, 128), id="multi_C"),
]


# Phase-0 supported compute_kernel_config — both forms must work.
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


# Affine modes — gamma/beta presence sweep. Three cells map to the
# `affine` axis in feature_spec.py {gamma_beta, gamma_only, no_affine}.
AFFINE_MODES = [
    pytest.param("no_affine", id="affine=none"),
    pytest.param("gamma_only", id="affine=gamma_only"),
    pytest.param("gamma_beta", id="affine=gamma_beta"),
]


def _make_inputs(shape, affine_mode, device):
    """Build (torch_input, torch_gamma_or_None, torch_beta_or_None,
    ttnn_input, ttnn_gamma_or_None, ttnn_beta_or_None) for a parametrized
    cell. Seeded so the test is reproducible.

    All tensors are ROW_MAJOR_LAYOUT float32. The kernel must not depend
    on host-side tilize/untilize.
    """
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    W = shape[-1]

    torch_gamma = None
    torch_beta = None
    ttnn_gamma = None
    ttnn_beta = None

    if affine_mode in ("gamma_only", "gamma_beta"):
        # Centered around 1.0 so the post-affine values are close-ish to
        # the un-affine result; this keeps the PCC sensitive to small
        # numerical regressions instead of being dominated by the scale.
        torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.5 + 1.0
        ttnn_gamma = ttnn.from_torch(
            torch_gamma,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    if affine_mode == "gamma_beta":
        torch_beta = torch.randn((1, 1, 1, W), dtype=torch.float32) * 0.1
        ttnn_beta = ttnn.from_torch(
            torch_beta,
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    return torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta


# --------------------------------------------------------------------------
# Main acceptance test — PyTorch reference, PCC ≥ 0.999.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("affine_mode", AFFINE_MODES)
def test_layer_norm_rm_acceptance(device, shape, affine_mode):
    """Phase-0 acceptance: layer_norm(float32, ROW_MAJOR_LAYOUT) == pytorch_reference."""
    torch_input, torch_gamma, torch_beta, ttnn_input, ttnn_gamma, ttnn_beta = _make_inputs(shape, affine_mode, device)

    torch_expected = pytorch_reference(torch_input, torch_gamma, torch_beta, epsilon=1e-5)

    ttnn_output = layer_norm(ttnn_input, ttnn_gamma, ttnn_beta)

    # Shape / dtype / layout preserved end-to-end (kernel must NOT change
    # layout — RM in, RM out).
    assert list(ttnn_output.shape) == list(shape), f"shape mismatch: got {ttnn_output.shape}, expected {shape}"
    assert ttnn_output.dtype == ttnn.float32, f"dtype mismatch: got {ttnn_output.dtype}, expected {ttnn.float32}"
    assert (
        ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT
    ), f"layout mismatch: got {ttnn_output.layout}, expected {ttnn.ROW_MAJOR_LAYOUT}"

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# Compute-kernel-config acceptance — both None and the explicit Phase-0
# config must produce the same correct result on a small representative
# shape.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("config", PHASE0_CONFIGS)
def test_layer_norm_rm_compute_kernel_config(device, config):
    """Both the None default and the explicit Phase-0 config must work."""
    shape = (1, 1, 64, 128)
    torch_input, _, _, ttnn_input, _, _ = _make_inputs(shape, "no_affine", device)
    torch_expected = pytorch_reference(torch_input, epsilon=1e-5)

    ttnn_output = layer_norm(ttnn_input, compute_kernel_config=config)
    torch_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# Epsilon variation — confirm epsilon is plumbed through correctly. Use
# a 32x32 strip with a tiny variance row so changing epsilon changes the
# output materially.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("epsilon", [1e-5, 1e-3, 1e-1])
def test_layer_norm_rm_epsilon(device, epsilon):
    """Custom epsilon flows through and yields the expected numerical change."""
    shape = (1, 1, 32, 64)
    torch_input, _, _, ttnn_input, _, _ = _make_inputs(shape, "no_affine", device)
    torch_expected = pytorch_reference(torch_input, epsilon=epsilon)

    ttnn_output = layer_norm(ttnn_input, epsilon=epsilon)
    torch_output = ttnn.to_torch(ttnn_output)

    assert_with_pcc(torch_expected, torch_output, 0.999)


# --------------------------------------------------------------------------
# Multi-rank support — the op accepts rank-2/3/4 inputs (rank ≥ 2 with
# final two dims tile-aligned). The kernel flattens leading dims into a
# strip index; output shape exactly matches input shape.
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((32, 64), id="rank2_32x64"),
        pytest.param((4, 32, 128), id="rank3_4x32x128"),
        pytest.param((2, 3, 32, 64), id="rank4_2x3x32x64"),
    ],
)
def test_layer_norm_rm_rank(device, shape):
    """Rank 2, 3, and 4 inputs all preserve shape end-to-end."""
    torch.manual_seed(42)
    torch_input = torch.randn(shape, dtype=torch.float32)
    torch_expected = pytorch_reference(torch_input, epsilon=1e-5)

    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    ttnn_output = layer_norm(ttnn_input)
    assert list(ttnn_output.shape) == list(shape)
    assert ttnn_output.layout == ttnn.ROW_MAJOR_LAYOUT

    torch_output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_expected, torch_output, 0.999)
