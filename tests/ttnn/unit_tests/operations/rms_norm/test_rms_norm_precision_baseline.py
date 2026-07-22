# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision baseline for rms_norm (Phase-0 supported corner).

Measures PCC, max/mean abs error, relative RMS error, and the got/true ratio
spread (the scale-bug detector) across a small shape sweep at the Phase-0
supported precision corner (bf16 / f32, fp32_dest_acc_en=True, TILE, RM gamma).

Not part of the acceptance contract — this file is the verifier's measured
baseline. It uses assert_with_pcc (tests.ttnn.utils_for_testing) and
comp_allclose (models.common.utility_functions) rather than hand-rolled checks.
"""

import pytest
import torch
import ttnn

from ttnn.operations.rms_norm import rms_norm
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose


TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}

# Phase-0 per-dtype PCC gate (matches the golden suite TOLERANCES).
PCC = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
}


def _reference(x, gamma, eps):
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    out = x * torch.rsqrt(var + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out


def _ratio_spread(got, exp):
    """got/true ratio over finite, non-tiny-reference elements.

    A tight cluster of r around a NON-1.0 constant => uniform scale/structural
    bug. A broad spread centered on 1.0 => ordinary precision noise. Returned as
    (median, p5, p95, std)."""
    got = got.flatten().to(torch.float32)
    exp = exp.flatten().to(torch.float32)
    mask = torch.isfinite(got) & torch.isfinite(exp) & (exp.abs() > 1e-3)
    r = got[mask] / exp[mask]
    if r.numel() == 0:
        return (float("nan"),) * 4
    return (
        torch.median(r).item(),
        torch.quantile(r, 0.05).item(),
        torch.quantile(r, 0.95).item(),
        torch.std(r).item(),
    )


SHAPES = [
    pytest.param((32, 64), id="small_2d"),
    pytest.param((2, 64, 128), id="medium_3d"),
    pytest.param((2, 4, 128, 512), id="medium_4d"),
    pytest.param((1, 1, 128, 4096), id="large_wide"),
]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "f32"])
@pytest.mark.parametrize("with_gamma", [True, False], ids=["gamma", "no_gamma"])
def test_precision_baseline(device, shape, dtype, with_gamma):
    torch.manual_seed(42)
    eps = 1e-6
    W = shape[-1]

    torch_input = torch.randn(shape, dtype=torch.float32)
    ttnn_input = ttnn.from_torch(
        torch_input,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    torch_gamma = None
    ttnn_gamma = None
    if with_gamma:
        torch_gamma = torch.randn(W, dtype=torch.float32)
        ttnn_gamma = ttnn.from_torch(
            torch_gamma.reshape(1, 1, 1, W),
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    config = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    ttnn_out = rms_norm(ttnn_input, gamma=ttnn_gamma, epsilon=eps, compute_kernel_config=config)
    got = ttnn.to_torch(ttnn_out).to(torch.float32)
    exp = _reference(torch_input, torch_gamma, eps)

    max_abs = (got - exp).abs().max().item()
    mean_abs = (got - exp).abs().mean().item()
    rel_rms = ((got - exp).pow(2).mean().sqrt() / exp.pow(2).mean().sqrt()).item()
    _, allclose_str = comp_allclose(exp, got)
    med, p5, p95, std = _ratio_spread(got, exp)

    print(
        f"\n[precision] shape={tuple(shape)} dtype={dtype} gamma={with_gamma}\n"
        f"    max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} rel_rms={rel_rms:.6f}\n"
        f"    ratio(got/true): median={med:.5f} p5={p5:.5f} p95={p95:.5f} std={std:.5f}\n"
        f"    {allclose_str}"
    )

    assert_with_pcc(exp, got, pcc=PCC[dtype])
