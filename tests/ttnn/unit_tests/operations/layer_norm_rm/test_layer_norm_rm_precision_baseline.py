# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline test for layer_norm_rm.

Measures PCC, max abs error, mean abs error, and relative RMS error across
a standard set of shapes. These measurements establish the Phase 0 accuracy
baseline for refinement agents.

Uses the standard testing utilities from tests.ttnn.utils_for_testing
and models.common.utility_functions.
"""

import pytest
import torch
import torch.nn.functional as F
import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.operations.layer_norm_rm import layer_norm_rm


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


def torch_layer_norm_rm(x, gamma=None, beta=None, epsilon=1e-5):
    """PyTorch reference computed in float32 for max precision."""
    W = x.shape[-1]
    x_f32 = x.float()
    g = gamma.float() if gamma is not None else None
    b = beta.float() if beta is not None else None
    return F.layer_norm(x_f32, [W], weight=g, bias=b, eps=epsilon).to(x.dtype)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def to_device(tensor_torch, device):
    return ttnn.from_torch(
        tensor_torch,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )


def compute_metrics(actual, expected):
    """Compute PCC, max abs error, mean abs error, relative RMS error."""
    a = actual.float()
    e = expected.float()

    # PCC via check_with_pcc (returns (passed, message))
    pcc_passed, pcc_msg = check_with_pcc(e, a, pcc=0.0)
    # Extract PCC value from message
    pcc_val = (
        float(
            pcc_msg.split("PCC got ")[1].split(".")[0] + "." + pcc_msg.split("PCC got ")[1].split(".")[1].split(" ")[0]
        )
        if "PCC got " in pcc_msg
        else 0.0
    )

    abs_diff = (a - e).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    # Relative RMS error
    rms_num = (a - e).pow(2).mean().sqrt()
    rms_den = e.pow(2).mean().sqrt()
    rel_rms_err = (rms_num / rms_den).item() if rms_den > 1e-30 else 0.0

    return pcc_val, max_abs_err, mean_abs_err, rel_rms_err


def compute_pcc_direct(actual, expected):
    """Compute PCC directly without dependency on message parsing."""
    a = actual.float().flatten()
    e = expected.float().flatten()
    a_c = a - a.mean()
    e_c = e - e.mean()
    num = (a_c * e_c).sum()
    den = a_c.norm() * e_c.norm()
    return (num / den).item() if den > 1e-30 else 1.0


# ---------------------------------------------------------------------------
# Precision baseline shapes
# ---------------------------------------------------------------------------

PRECISION_SHAPES = [
    pytest.param((1, 1, 32, 32), id="small_32x32"),
    pytest.param((1, 1, 64, 128), id="medium_64x128"),
    pytest.param((1, 1, 128, 256), id="large_128x256"),
    pytest.param((1, 1, 32, 512), id="wide_32x512"),
    pytest.param((1, 1, 256, 32), id="tall_256x32"),
    pytest.param((2, 2, 64, 64), id="batched_2x2x64x64"),
    pytest.param((4, 1, 32, 256), id="multibatch_4x1x32x256"),
    pytest.param((1, 1, 32, 768), id="wide_32x768"),
]


# ---------------------------------------------------------------------------
# Tests: precision baseline with gamma+beta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", PRECISION_SHAPES)
def test_precision_baseline_gamma_beta(shape, device):
    """Measure precision with gamma and beta (full layer norm)."""
    torch.manual_seed(42)
    W = shape[-1]
    x_torch = torch.randn(shape, dtype=torch.bfloat16)
    gamma_torch = torch.randn(W, dtype=torch.bfloat16)
    beta_torch = torch.randn(W, dtype=torch.bfloat16)

    expected = torch_layer_norm_rm(x_torch, gamma=gamma_torch, beta=beta_torch)

    x_tt = to_device(x_torch, device)
    gamma_tt = to_device(gamma_torch.reshape(1, 1, 1, W), device)
    beta_tt = to_device(beta_torch.reshape(1, 1, 1, W), device)
    result_tt = layer_norm_rm(x_tt, gamma_tt, beta_tt)
    result_torch = ttnn.to_torch(result_tt)

    pcc = compute_pcc_direct(result_torch, expected)

    abs_diff = (result_torch.float() - expected.float()).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    rms_num = (result_torch.float() - expected.float()).pow(2).mean().sqrt()
    rms_den = expected.float().pow(2).mean().sqrt()
    rel_rms_err = (rms_num / rms_den).item() if rms_den > 1e-30 else 0.0

    # Log the metrics
    print(f"\n[PRECISION] shape={shape} mode=gamma_beta")
    print(f"  PCC={pcc:.6f}")
    print(f"  max_abs_err={max_abs_err:.6f}")
    print(f"  mean_abs_err={mean_abs_err:.6f}")
    print(f"  rel_rms_err={rel_rms_err:.6f}")

    # Assert PCC >= 0.999
    assert pcc >= 0.999, f"PCC too low: {pcc:.6f} < 0.999 for shape {shape}"


# ---------------------------------------------------------------------------
# Tests: precision baseline without affine (pure normalization)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", PRECISION_SHAPES)
def test_precision_baseline_pure(shape, device):
    """Measure precision without affine parameters (pure normalization)."""
    torch.manual_seed(42)
    x_torch = torch.randn(shape, dtype=torch.bfloat16)

    expected = torch_layer_norm_rm(x_torch)

    x_tt = to_device(x_torch, device)
    result_tt = layer_norm_rm(x_tt)
    result_torch = ttnn.to_torch(result_tt)

    pcc = compute_pcc_direct(result_torch, expected)

    abs_diff = (result_torch.float() - expected.float()).abs()
    max_abs_err = abs_diff.max().item()
    mean_abs_err = abs_diff.mean().item()

    rms_num = (result_torch.float() - expected.float()).pow(2).mean().sqrt()
    rms_den = expected.float().pow(2).mean().sqrt()
    rel_rms_err = (rms_num / rms_den).item() if rms_den > 1e-30 else 0.0

    print(f"\n[PRECISION] shape={shape} mode=pure")
    print(f"  PCC={pcc:.6f}")
    print(f"  max_abs_err={max_abs_err:.6f}")
    print(f"  mean_abs_err={mean_abs_err:.6f}")
    print(f"  rel_rms_err={rel_rms_err:.6f}")

    assert pcc >= 0.999, f"PCC too low: {pcc:.6f} < 0.999 for shape {shape}"
