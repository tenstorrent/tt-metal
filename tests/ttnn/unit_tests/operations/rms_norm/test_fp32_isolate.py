# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Isolation test: fp32 + Wt>1, RM vs TILE path.
If TILE path passes, the bug is in tilize or tilize->reduce interaction.
If TILE path also fails, the bug is in reduce or later phases.
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
    zero_fraction = (actual.abs() < 1e-30).float().mean().item()

    status = "PASS" if pcc > 0.999 else "FAIL"
    print(
        f"  [{status}] {label}: PCC={pcc:.6f}, RMS={rms_err:.6f}, MaxErr={max_abs_err:.6f}, ZeroFrac={zero_fraction:.4f}"
    )
    return pcc


# Shape with Wt=8 — the minimal failing case
SHAPE = (1, 1, 32, 256)


def test_fp32_rm_wt8(device):
    """fp32, RM path, Wt=8 — known failing."""
    torch.manual_seed(42)
    x = torch.randn(*SHAPE, dtype=torch.float32)
    expected = pytorch_rms_norm(x)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)
    pcc = compare(out, expected, "fp32 RM Wt=8")
    assert pcc > 0.99, f"PCC {pcc:.6f}"


def test_fp32_tile_wt8(device):
    """fp32, TILE path, Wt=8 — does this also fail?"""
    torch.manual_seed(42)
    x = torch.randn(*SHAPE, dtype=torch.float32)
    expected = pytorch_rms_norm(x)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)
    pcc = compare(out, expected, "fp32 TILE Wt=8")
    assert pcc > 0.99, f"PCC {pcc:.6f}"


def test_fp32_rm_wt2(device):
    """fp32, RM, Wt=2 — smallest multi-tile width."""
    shape = (1, 1, 32, 64)
    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=torch.float32)
    expected = pytorch_rms_norm(x)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)
    pcc = compare(out, expected, "fp32 RM Wt=2")
    assert pcc > 0.99, f"PCC {pcc:.6f}"


def test_fp32_tile_wt2(device):
    """fp32, TILE, Wt=2."""
    shape = (1, 1, 32, 64)
    torch.manual_seed(42)
    x = torch.randn(*shape, dtype=torch.float32)
    expected = pytorch_rms_norm(x)
    x_tt = ttnn.from_torch(
        x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = rms_norm(x_tt, gamma=None, epsilon=1e-6)
    pcc = compare(out, expected, "fp32 TILE Wt=2")
    assert pcc > 0.99, f"PCC {pcc:.6f}"
