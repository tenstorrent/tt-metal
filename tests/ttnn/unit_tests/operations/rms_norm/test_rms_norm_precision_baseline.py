# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""rms_norm precision baseline (Phase 0 corner: fp32_dest_acc_en=True, HiFi4, interleaved DRAM).

Measures PCC (comp_pcc), max / mean abs error (comp_allclose + torch), relative RMS
(RMS error / std(reference), as eval.metrics.check_output does), ULP p99 at the output dtype,
and the got/true ratio spread (scale-bug detector: a tight cluster around a non-1.0 constant
means a uniform scale/structural bug, a broad spread centred on 1.0 is rounding noise).

Gate: PCC and relative RMS together against helpers.TOLERANCES[dtype].
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose, comp_pcc

from eval.golden_tests.rms_norm.helpers import TOLERANCES
from eval.metrics import compute_metrics_torch
from ttnn.operations.rms_norm import rms_norm

_TORCH_DTYPE = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}

SHAPES = [
    (1, 1, 32, 64),  # small, single tile-row, single core
    (1, 1, 100, 1000),  # both non-aligned, masked W tail, h padding
    (2, 1, 256, 4096),  # medium, LLM-wide hidden, multi-core
    (1, 1, 2048, 8192),  # large, wide rows (W chunked, K > 1)
]


def _reference(x, gamma, eps):
    xf = x.to(torch.float32)
    out = xf / torch.sqrt(torch.mean(xf * xf, dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.to(torch.float32).reshape(-1)
    return out  # kept in fp32: errors are measured against the exact result


def _config():
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = ttnn.MathFidelity.HiFi4
    cfg.fp32_dest_acc_en = True
    cfg.math_approx_mode = False
    return cfg


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["tile", "rm"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32], ids=["bf16", "fp32"])
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_precision_baseline(device, shape, dtype, layout):
    torch.manual_seed(1234)
    eps = 1e-6
    tdt = _TORCH_DTYPE[dtype]
    W = shape[-1]
    x = torch.randn(shape, dtype=torch.float32).to(tdt)
    gamma = torch.randn(W, dtype=torch.float32).to(tdt)

    tx = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tg = ttnn.from_torch(
        gamma.reshape(1, 1, 1, W),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    got = ttnn.to_torch(rms_norm(tx, gamma=tg, epsilon=eps, compute_kernel_config=_config()))
    expected = _reference(x, gamma, eps)
    # Reference rounded to the output dtype: the best any kernel can do at this output precision.
    expected_q = expected.to(tdt)

    _, pcc = comp_pcc(expected_q, got, 0.99)
    _, allclose_msg = comp_allclose(expected_q, got)
    m = compute_metrics_torch(got, expected_q, readback_dtype=got.dtype)

    gf = got.to(torch.float32)
    abs_err = (gf - expected).abs()
    mask = torch.isfinite(gf) & (expected.abs() > 1e-3)
    ratio = gf[mask] / expected[mask]
    r_med = ratio.median().item()
    r_p5, r_p95 = torch.quantile(ratio[:1_000_000].double(), torch.tensor([0.05, 0.95], dtype=torch.float64)).tolist()

    print(
        f"\nPRECISION shape={shape} dtype={dtype} layout={layout} "
        f"pcc={pcc:.7f} rel_rms={m.rms:.5f} max_abs={abs_err.max().item():.5f} "
        f"mean_abs={abs_err.mean().item():.6f} ulp_p99={m.ulp_p99:.1f} "
        f"ratio_median={r_med:.6f} ratio_p5={r_p5:.6f} ratio_p95={r_p95:.6f} | {allclose_msg}"
    )

    pcc_tol, rms_tol = TOLERANCES[dtype]
    assert not (m.has_inf or m.has_nan)
    assert pcc >= pcc_tol, f"pcc {pcc} < {pcc_tol}"
    assert m.rms <= rms_tol, f"relative rms {m.rms} > {rms_tol}"
    # Scale-bug guard: the central ratio must sit on 1.0 (a uniform scale error moves the median).
    assert abs(r_med - 1.0) < 1e-2, f"got/true ratio median {r_med} off 1.0 — suspect a scale bug"
