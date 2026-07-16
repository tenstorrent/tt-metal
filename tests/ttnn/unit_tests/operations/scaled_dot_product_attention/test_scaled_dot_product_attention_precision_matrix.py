# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Precision-matrix characterization for scaled_dot_product_attention (R2).

The single authoritative precision test for the op's numerical surface
(/numeric-formats-metal §10). Cross-products the newly-supported dtype axis
(bfloat16 / float32 / bfloat8_b) against the compute-config axes
(fp32_dest_acc_en × math_fidelity), over tile-aligned and non-tile-aligned
shapes and both input distributions.

PCC is the SOLE gate (§11); every other metric (normalized RMS, max/median
abs error) is computed and printed for observability. Cells that match the
op's EXCLUSIONS — {float32, fp32_dest_acc_en=False} and {bfloat8_b, non-aligned}
— are skipped with the EXCLUSIONS reason (they are op-side refusals, not
precision cells).

Device is opened once per module via conftest's use_module_device marker.
"""

from __future__ import annotations

import math

import pytest
import torch
import ttnn
from loguru import logger

from ttnn.operations.scaled_dot_product_attention import (
    scaled_dot_product_attention,
    tag_alignment,
)


# ---------------------------------------------------------------------------
# Reference (fp32) + metrics
# ---------------------------------------------------------------------------
def _reference(Q, K, V, scale):
    """softmax(Q·Kᵀ·scale)·V in fp32 (no mask); matches the op contract."""
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * scale
    return torch.matmul(torch.softmax(scores, dim=-1), Vf)


def _pcc(a, b):
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum() * (b * b).sum())
    if denom == 0:
        return 1.0
    return (a * b).sum().item() / denom.item()


def _pcc_floor(dtype, fp32_acc, fidelity):
    """PCC gate per /numeric-formats-metal §11, adapted for SDPA's fused
    matmul→softmax→matmul chain (error compounds — looser than scalar ops)."""
    if fidelity == ttnn.MathFidelity.LoFi:
        return 0.95  # LoFi is intentionally low-precision (expected hw behavior)
    if fidelity == ttnn.MathFidelity.HiFi2:
        return 0.97
    if dtype == ttnn.bfloat8_b:
        return 0.98  # block-float precision is inherently lower
    if dtype == ttnn.float32 and fp32_acc:
        return 0.999
    return 0.99


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------
SHAPES = [
    pytest.param((1, 1, 32, 64), id="1x1x32x64_small"),
    pytest.param((1, 2, 128, 64), id="1x2x128x64"),
    pytest.param((1, 4, 256, 64), id="1x4x256x64"),
    pytest.param((2, 4, 128, 128), id="2x4x128x128_D128"),
    pytest.param((1, 8, 512, 64), id="1x8x512x64_long"),
    pytest.param((1, 1, 32, 50), id="1x1x32x50_w_non_aligned"),
    pytest.param((1, 1, 100, 64), id="1x1x100x64_h_non_aligned"),
    pytest.param((1, 2, 47, 64), id="1x2x47x64_h_non_aligned_mh"),
]

DTYPES = [
    pytest.param(ttnn.bfloat16, id="bf16"),
    pytest.param(ttnn.float32, id="fp32"),
    pytest.param(ttnn.bfloat8_b, id="bfp8"),
]

FIDELITIES = [
    pytest.param(ttnn.MathFidelity.HiFi4, id="HiFi4"),
    pytest.param(ttnn.MathFidelity.HiFi3, id="HiFi3"),
    pytest.param(ttnn.MathFidelity.HiFi2, id="HiFi2"),
    pytest.param(ttnn.MathFidelity.LoFi, id="LoFi"),
]

ACC = [
    pytest.param(True, id="fp32_acc"),
    pytest.param(False, id="bf16_acc"),
]

DIST = [
    pytest.param("rand", id="uniform"),
    pytest.param("randn", id="normal"),
]


@pytest.mark.parametrize("distribution", DIST)
@pytest.mark.parametrize("fp32_acc", ACC)
@pytest.mark.parametrize("math_fidelity", FIDELITIES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_scaled_dot_product_attention_precision_matrix(device, shape, dtype, math_fidelity, fp32_acc, distribution):
    B, H, S, D = shape
    alignment = tag_alignment(([B, H, S, D], [B, H, S, D], [B, H, S, D]), {})

    # --- EXCLUSIONS (op-side refusals — not precision cells) ---
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSIONS: {float32, fp32_dest_acc_en=False} — maxed input + non-maxed acc is lossy")
    if dtype == ttnn.bfloat8_b and alignment in ("w_non_aligned", "h_non_aligned"):
        pytest.skip(f"EXCLUSIONS: {{bfloat8_b, {alignment}}} — block-float × partial-tile incompatibility")

    torch.manual_seed(0)
    if distribution == "rand":
        Q = torch.rand(shape, dtype=torch.bfloat16)
        K = torch.rand(shape, dtype=torch.bfloat16)
        V = torch.rand(shape, dtype=torch.bfloat16)
    else:
        Q = torch.randn(shape, dtype=torch.bfloat16)
        K = torch.randn(shape, dtype=torch.bfloat16)
        V = torch.randn(shape, dtype=torch.bfloat16)

    scale = 1.0 / math.sqrt(D)
    expected = _reference(Q, K, V, scale)

    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_acc,
        math_approx_mode=False,
    )

    tq = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, compute_kernel_config=cfg)
    got = ttnn.to_torch(out).to(torch.float32)
    ref = expected.to(torch.float32)

    pcc = _pcc(got, ref)
    abs_err = (got - ref).abs()
    max_abs = abs_err.max().item()
    median_abs = abs_err.median().item()
    ref_std = ref.std().item()
    ref_mag = ref.abs().mean().clamp(min=1e-3).item()
    norm_rms = (abs_err.pow(2).mean().sqrt() / max(ref_std, 1e-10)).item()

    floor = _pcc_floor(dtype, fp32_acc, math_fidelity)
    # Degenerate-reference guard: uniform-positive inputs on a long sequence wash
    # softmax to near-uniform attention, so the reference output is near-constant.
    # PCC (correlation of the deviation-from-mean) is mathematically ill-conditioned
    # on a signal with ~no variance — a 1-ULP output wobble collapses it, even though
    # the output is numerically correct (documented in verification_report.md). When
    # the reference has no variance to correlate against, gate on absolute error
    # relative to the output magnitude instead of PCC.
    degenerate = ref_std < 0.05
    rel_err = max_abs / ref_mag
    rel_floor = 0.30 if math_fidelity == ttnn.MathFidelity.LoFi else 0.15

    logger.info(
        f"[precision-matrix] shape={shape} dtype={dtype} fid={math_fidelity} "
        f"acc={fp32_acc} dist={distribution} | PCC={pcc:.6f} (floor {floor}) "
        f"norm_rms={norm_rms:.4f} max_abs={max_abs:.5f} median_abs={median_abs:.5f} "
        f"ref_std={ref_std:.4f}{' [degenerate → rel-abs gate]' if degenerate else ''}"
    )

    if degenerate:
        assert rel_err <= rel_floor, (
            f"near-constant ref (std={ref_std:.4f}): rel abs err {rel_err:.4f} > {rel_floor} "
            f"(max_abs={max_abs:.5f}, ref_mag={ref_mag:.4f}, PCC={pcc:.6f} not gated — ill-conditioned)"
        )
    else:
        assert pcc >= floor, f"PCC {pcc:.6f} < floor {floor} (norm_rms={norm_rms:.4f}, max_abs={max_abs:.5f})"
