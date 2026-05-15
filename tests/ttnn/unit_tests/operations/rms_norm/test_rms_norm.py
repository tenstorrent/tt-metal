# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Acceptance test for rms_norm.

This file is the IMMUTABLE specification for the operation. Do NOT modify it
when iterating on the implementation. The implementer makes the operation pass
this test; the test does not adapt to the implementer.

Reference (PyTorch):
    rms = sqrt(mean(x**2, dim=-1, keepdim=True) + eps)
    out = (x / rms) * gamma                  # gamma broadcast over last dim

Coverage:
  - shapes: single-tile, multi-tile, wide W, multi-batch, 3D and 4D rank
  - layouts: ROW_MAJOR_LAYOUT and TILE_LAYOUT (gated by shape divisibility)
  - dtypes: bfloat16 and float32
  - gamma:  None and provided
  - epsilon: default (1e-6) and custom (1e-5)

PCC thresholds (matching the golden suite, keyed by dtype):
  float32  -> 0.999
  bfloat16 -> 0.995
"""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from ttnn.operations.rms_norm import rms_norm


# --------------------------------------------------------------------------- #
#  PyTorch reference                                                           #
# --------------------------------------------------------------------------- #


def _torch_rms_norm(x: torch.Tensor, gamma: torch.Tensor | None, eps: float) -> torch.Tensor:
    """RMSNorm along last dim. All math in float32 for stability."""
    x_fp = x.float()
    rms = torch.sqrt(x_fp.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = x_fp / rms
    if gamma is not None:
        out = out * gamma.float().view(*([1] * (x.dim() - 1)), -1)
    return out


# --------------------------------------------------------------------------- #
#  PCC thresholds                                                              #
# --------------------------------------------------------------------------- #

_PCC_BY_DTYPE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
}

_TTNN_TO_TORCH_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
}


# --------------------------------------------------------------------------- #
#  Parametrize matrix                                                          #
# --------------------------------------------------------------------------- #
#
# Each entry is (shape, layout). Shapes with H % 32 != 0 or W % 32 != 0 are
# only included for ROW_MAJOR_LAYOUT (the op rejects TILE_LAYOUT for such
# shapes).

_SHAPE_LAYOUT_CASES = [
    # --- 4D tile-aligned: BOTH layouts ---
    pytest.param((1, 1, 32, 32), ttnn.TILE_LAYOUT, id="4D_single_tile-tile"),
    pytest.param((1, 1, 32, 32), ttnn.ROW_MAJOR_LAYOUT, id="4D_single_tile-rm"),
    pytest.param((1, 1, 32, 64), ttnn.TILE_LAYOUT, id="4D_1x2_tiles-tile"),
    pytest.param((1, 1, 32, 64), ttnn.ROW_MAJOR_LAYOUT, id="4D_1x2_tiles-rm"),
    pytest.param((1, 1, 64, 128), ttnn.TILE_LAYOUT, id="4D_2x4_tiles-tile"),
    pytest.param((1, 1, 64, 128), ttnn.ROW_MAJOR_LAYOUT, id="4D_2x4_tiles-rm"),
    pytest.param((2, 1, 64, 64), ttnn.TILE_LAYOUT, id="4D_multibatch-tile"),
    pytest.param((2, 1, 64, 64), ttnn.ROW_MAJOR_LAYOUT, id="4D_multibatch-rm"),
    pytest.param((1, 1, 32, 1024), ttnn.TILE_LAYOUT, id="4D_wide_W=1024-tile"),
    pytest.param((1, 1, 32, 1024), ttnn.ROW_MAJOR_LAYOUT, id="4D_wide_W=1024-rm"),
    # --- 3D tile-aligned ---
    pytest.param((1, 32, 128), ttnn.TILE_LAYOUT, id="3D_small-tile"),
    pytest.param((1, 32, 128), ttnn.ROW_MAJOR_LAYOUT, id="3D_small-rm"),
    pytest.param((2, 64, 256), ttnn.TILE_LAYOUT, id="3D_multibatch-tile"),
    # --- 2D tile-aligned ---
    pytest.param((32, 64), ttnn.TILE_LAYOUT, id="2D_small-tile"),
    pytest.param((32, 64), ttnn.ROW_MAJOR_LAYOUT, id="2D_small-rm"),
    pytest.param((128, 256), ttnn.TILE_LAYOUT, id="2D_multitile-tile"),
    # --- non-tile-aligned (RM only) ---
    pytest.param((1, 1, 32, 50), ttnn.ROW_MAJOR_LAYOUT, id="4D_W_nonaligned-rm"),
    pytest.param((1, 1, 17, 64), ttnn.ROW_MAJOR_LAYOUT, id="4D_H_nonaligned-rm"),
    pytest.param((32, 47), ttnn.ROW_MAJOR_LAYOUT, id="2D_W_nonaligned-rm"),
]


def _to_device(t: torch.Tensor, *, dtype, layout, device):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)


# --------------------------------------------------------------------------- #
#  Core matrix test: shape × layout × dtype × gamma_mode × epsilon            #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape,layout", _SHAPE_LAYOUT_CASES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("gamma_mode", ["no_gamma", "gamma"])
@pytest.mark.parametrize("epsilon", [1e-6, 1e-5], ids=["eps_default", "eps_1e-5"])
def test_rms_norm(device, shape, layout, dtype, gamma_mode, epsilon):
    """Cartesian product of (shape × layout) × dtype × gamma × epsilon."""
    torch.manual_seed(42)

    torch_dtype = _TTNN_TO_TORCH_DTYPE[dtype]
    torch_input = torch.randn(shape, dtype=torch_dtype)

    if gamma_mode == "gamma":
        # Gamma is always RM, shape (1, 1, 1, W).
        torch_gamma_4d = torch.randn(1, 1, 1, shape[-1], dtype=torch_dtype)
        torch_gamma = torch_gamma_4d
        ttnn_gamma = _to_device(torch_gamma_4d, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    else:
        torch_gamma = None
        ttnn_gamma = None

    ttnn_input = _to_device(torch_input, dtype=dtype, layout=layout, device=device)

    ttnn_output = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=epsilon)

    # Shape preserved.
    assert list(ttnn_output.shape) == list(shape), f"shape mismatch: expected {shape}, got {list(ttnn_output.shape)}"

    # Layout preserved.
    assert ttnn_output.layout == layout, f"output layout {ttnn_output.layout} != input layout {layout}"

    # Dtype preserved.
    assert ttnn_output.dtype == dtype, f"output dtype {ttnn_output.dtype} != input dtype {dtype}"

    # Compare against PyTorch reference (in fp32 to avoid double rounding).
    torch_output = ttnn.to_torch(ttnn_output).float()
    torch_expected = _torch_rms_norm(torch_input, torch_gamma, epsilon).to(torch_output.dtype)

    pcc = _PCC_BY_DTYPE[dtype]
    assert_with_pcc(torch_expected, torch_output, pcc=pcc)


# --------------------------------------------------------------------------- #
#  Validation tests (Python-side checks)                                       #
# --------------------------------------------------------------------------- #


def test_rms_norm_rejects_rank_1(device):
    """rank < 2 must raise ValueError or RuntimeError."""
    torch.manual_seed(42)
    torch_input = torch.randn(32, dtype=torch.bfloat16)
    ttnn_input = _to_device(torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(ttnn_input)


def test_rms_norm_rejects_gamma_shape_mismatch(device):
    """Gamma whose last dim does not match input last dim must raise."""
    torch.manual_seed(42)
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.bfloat16)
    ttnn_input = _to_device(torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Wrong last-dim: 32 != 64.
    torch_gamma = torch.randn(1, 1, 1, 32, dtype=torch.bfloat16)
    ttnn_gamma = _to_device(torch_gamma, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    with pytest.raises((ValueError, RuntimeError)):
        rms_norm(ttnn_input, gamma=ttnn_gamma)
