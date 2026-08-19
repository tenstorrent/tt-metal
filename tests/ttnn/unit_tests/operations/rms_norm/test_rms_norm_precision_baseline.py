# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision baseline for ttnn.operations.rms_norm (verifier artifact).

Measures, per (shape, dtype, regime):
  * PCC                      (tests.ttnn.utils_for_testing.assert_with_pcc)
  * max abs error / mean abs error
  * relative RMS error
  * the got/true RATIO SPREAD — the scale-bug detector.  A tight cluster of
    r = actual / expected around a NON-1.0 constant is a uniform scale /
    structural bug (fix the kernel), whereas a broad spread centred on 1.0 is
    ordinary rounding noise.  RMS-norm is exactly the op where this matters:
    folding tile padding into the denominator produces a near-uniform scale
    error that PCC is largely blind to (op_design.md risk R1).

Numbers are recorded in ttnn/ttnn/operations/rms_norm/verification_report.md.
"""

import pytest
import torch

import ttnn
from ttnn.operations.rms_norm import rms_norm
from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import comp_allclose

# Same thresholds as the golden suite (eval/golden_tests/rms_norm/helpers.py).
PCC_TARGET = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
}

# Shapes: small single-tile, medium multi-tile, non-tile-aligned (Regime B,
# masked reduce), and one large wide-hidden shape.
SHAPES = [
    (1, 1, 32, 64),
    (1, 1, 128, 512),
    (1, 1, 32, 72),  # W non-aligned -> Regime B, masked reduce
    (1, 1, 2048, 4096),  # large / wide hidden
]

EPS = 1e-6


def _torch_rms_norm(x, gamma, epsilon=EPS):
    x32 = x.to(torch.float32)
    out = x32 * torch.rsqrt(x32.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    return out * gamma.to(torch.float32)


def _ratio_stats(expected, actual):
    """Median and spread of r = actual / expected over finite, non-tiny refs."""
    e = expected.flatten().to(torch.float64)
    a = actual.flatten().to(torch.float64)
    scale = e.abs().max().clamp_min(1e-30)
    keep = torch.isfinite(e) & torch.isfinite(a) & (e.abs() > 1e-3 * scale)
    if keep.sum() < 8:
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = a[keep] / e[keep]
    p5, p95 = torch.quantile(r, torch.tensor([0.05, 0.95], dtype=torch.float64))
    return r.median().item(), r.std().item(), p5.item(), p95.item()


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(str(d) for d in s))
def test_rms_norm_precision_baseline(device, shape, dtype):
    torch.manual_seed(0)
    W = shape[-1]
    torch_x = torch.randn(shape, dtype=torch.float32)
    torch_gamma = torch.randn((1, 1, 1, W), dtype=torch.float32)

    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    tt_out = rms_norm(tt_x, gamma=tt_gamma, epsilon=EPS)
    actual = ttnn.to_torch(tt_out).to(torch.float32)
    expected = _torch_rms_norm(torch_x, torch_gamma, EPS)

    err = (actual - expected).abs()
    max_abs = err.max().item()
    mean_abs = err.mean().item()
    denom = expected.pow(2).mean().sqrt().item()
    rel_rms = (err.pow(2).mean().sqrt().item() / denom) if denom > 0 else float("nan")
    r_med, r_std, r_p5, r_p95 = _ratio_stats(expected, actual)

    _, allclose_msg = comp_allclose(expected, actual, rtol=1e-2, atol=1e-2)

    print(
        f"\nPRECISION shape={tuple(shape)} dtype={dtype} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rel_rms={rel_rms:.6g} "
        f"ratio_median={r_med:.6f} ratio_std={r_std:.6g} ratio_p5={r_p5:.6f} ratio_p95={r_p95:.6f}\n"
        f"  {allclose_msg}"
    )

    # Scale-bug detector: a tight cluster of r around a non-1.0 constant is a
    # uniform scale / structural bug, NOT precision noise.
    assert abs(r_med - 1.0) < 0.02, f"got/true ratio median {r_med} is off 1.0 — uniform scale error"

    assert_with_pcc(expected, actual, PCC_TARGET[dtype])
