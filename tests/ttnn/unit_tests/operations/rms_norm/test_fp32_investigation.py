# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Investigation: float32 multi-tile-row failure in rms_norm.

Goal: isolate exactly which combination of (dtype, shape, layout, gamma) fails.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm


def pytorch_rms_norm(x, gamma=None, epsilon=1e-6):
    x = x.to(torch.float32)
    rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + epsilon)
    result = x / rms
    if gamma is not None:
        result = result * gamma.to(torch.float32)
    return result


def compare(ttnn_out, expected, label):
    actual = ttnn.to_torch(ttnn_out).to(torch.float64)
    expected_f = expected.to(torch.float64)
    a = actual.flatten()
    e = expected_f.flatten()
    ac = a - a.mean()
    ec = e - e.mean()
    num = (ac * ec).sum()
    den = ac.norm() * ec.norm()
    pcc = (num / den).item() if den > 1e-30 else 1.0
    rms_err = ((actual - expected_f) ** 2).mean().sqrt().item()
    max_abs_err = (actual - expected_f).abs().max().item()

    # Check for zeros
    zero_fraction = (actual.abs() < 1e-30).float().mean().item()

    status = "PASS" if pcc > 0.999 else "FAIL"
    print(
        f"  [{status}] {label}: PCC={pcc:.6f}, RMS={rms_err:.6f}, MaxErr={max_abs_err:.6f}, ZeroFrac={zero_fraction:.4f}"
    )
    return pcc, rms_err, zero_fraction


# ---- Test 1: Narrow down dtype x shape x layout (no gamma) ----
@pytest.mark.parametrize(
    "dtype_pair",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),  # 1 tile-row, 1 Wt
        (1, 1, 64, 32),  # 2 tile-rows, 1 Wt
        (1, 1, 128, 32),  # 4 tile-rows, 1 Wt
        (1, 1, 32, 256),  # 1 tile-row, 8 Wt
        (1, 1, 128, 256),  # 4 tile-rows, 8 Wt -- the failing case
    ],
    ids=["32x32", "64x32", "128x32", "32x256", "128x256"],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
def test_no_gamma_matrix(device, dtype_pair, shape, layout):
    """Matrix of dtype x shape x layout WITHOUT gamma."""
    torch_dtype, tt_dtype = dtype_pair
    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=torch_dtype)
    expected = pytorch_rms_norm(x).to(torch_dtype)

    x_tt = ttnn.from_torch(x, dtype=tt_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)

    pcc, rms_err, zero_frac = compare(out, expected, f"{shape} {layout} {tt_dtype} no_gamma")
    assert pcc > 0.99, f"PCC {pcc:.6f} too low"


# ---- Test 2: With gamma, same matrix ----
@pytest.mark.parametrize(
    "dtype_pair",
    [
        (torch.bfloat16, ttnn.bfloat16),
        (torch.float32, ttnn.float32),
    ],
    ids=["bf16", "fp32"],
)
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 64, 32),
        (1, 1, 128, 256),
    ],
    ids=["32x32", "64x32", "128x256"],
)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["RM", "TILE"])
def test_with_gamma_matrix(device, dtype_pair, shape, layout):
    """Matrix of dtype x shape x layout WITH gamma."""
    torch_dtype, tt_dtype = dtype_pair
    torch.manual_seed(42)
    N, C, H, W = shape
    x = torch.randn(N, C, H, W, dtype=torch_dtype)
    gamma = torch.randn(1, 1, 1, W, dtype=torch_dtype)
    expected = pytorch_rms_norm(x, gamma=gamma).to(torch_dtype)

    x_tt = ttnn.from_torch(x, dtype=tt_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    gamma_tt = ttnn.from_torch(
        gamma, dtype=tt_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=gamma_tt, epsilon=1e-6)

    pcc, rms_err, zero_frac = compare(out, expected, f"{shape} {layout} {tt_dtype} with_gamma")
    assert pcc > 0.99, f"PCC {pcc:.6f} too low"
