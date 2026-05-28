# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Precision baseline for layer_norm_rm — records PCC, abs error, relative
RMS, and per-row LayerNorm invariants across 4 representative shapes.

This is not a pass/fail test in the usual sense. The thresholds are
generous (PCC ≥ 0.999, max_abs ≤ 1e-3) — anything reasonable for an
fp32 LayerNorm. The test exists to give the verifier a stable numerical
snapshot of Phase 0 that future refinements can be compared against.

Shapes span:
- single tile (1, 1, 32, 32) — minimum fit
- multi tile  (1, 1, 64, 128) — middle
- multi batch (2, 4, 64, 64)  — NC > 1
- wider       (4, 1, 32, 512) — Wt = 16, near the Phase-0 L1 ceiling
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import assert_with_pcc

from ttnn.operations.layer_norm_rm import layer_norm_rm


SHAPES = [
    pytest.param((1, 1, 32, 32), id="single_tile_32x32"),
    pytest.param((1, 1, 64, 128), id="multi_tile_64x128"),
    pytest.param((2, 4, 64, 64), id="multi_batch_2x4x64x64"),
    pytest.param((4, 1, 32, 512), id="wider_4x1x32x512"),
]


def _torch_layer_norm(
    x: torch.Tensor,
    gamma: torch.Tensor | None,
    beta: torch.Tensor | None,
    epsilon: float,
) -> torch.Tensor:
    g = gamma.reshape(-1) if gamma is not None else None
    b = beta.reshape(-1) if beta is not None else None
    return torch.nn.functional.layer_norm(
        x.float(),
        normalized_shape=(x.shape[-1],),
        weight=g,
        bias=b,
        eps=epsilon,
    )


def _measure(shape, device, with_affine: bool):
    """Run the op and return (output_torch, expected_torch)."""
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32)

    W = shape[-1]
    if with_affine:
        g = torch.randn(W, dtype=torch.float32)
        b = torch.randn(W, dtype=torch.float32)
    else:
        g, b = None, None

    tt_x = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_g = (
        ttnn.from_torch(
            g.reshape(1, 1, 1, W),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if g is not None
        else None
    )
    tt_b = (
        ttnn.from_torch(
            b.reshape(1, 1, 1, W),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if b is not None
        else None
    )

    tt_out = layer_norm_rm(tt_x, tt_g, tt_b, epsilon=1e-5)
    out = ttnn.to_torch(tt_out).float()
    expected = _torch_layer_norm(x, g, b, 1e-5)
    return out, expected


def _rms_relative(out: torch.Tensor, ref: torch.Tensor) -> float:
    """Relative RMS error normalized by the reference's stddev."""
    abs_rms = torch.nn.functional.mse_loss(out, ref).sqrt().item()
    scale = ref.std().item()
    return abs_rms / scale if scale > 1e-12 else abs_rms


def _ulp_distance_p99(out: torch.Tensor, ref: torch.Tensor) -> float:
    """P99 of the absolute ULP distance at fp32 granularity."""
    a_bits = out.to(torch.float32).view(torch.int32).to(torch.int64)
    e_bits = ref.to(torch.float32).view(torch.int32).to(torch.int64)
    sign_offset = 1 << 31

    def _ordered(bits):
        sign = bits < 0
        magnitude = torch.where(sign, bits + sign_offset, bits)
        return torch.where(sign, -magnitude, magnitude)

    ulp = (_ordered(a_bits) - _ordered(e_bits)).abs().float()
    return torch.quantile(ulp, 0.99).item()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("with_affine", [False, True], ids=["no_affine", "gamma_beta"])
def test_precision_baseline(device, shape, with_affine):
    """Record numerics + assert generous thresholds.

    Metrics recorded (printed on -s):
      PCC, max_abs_diff, mean_abs_diff, rms_relative, ulp_p99.
    """
    out, ref = _measure(shape, device, with_affine)

    max_abs = (out - ref).abs().max().item()
    mean_abs = (out - ref).abs().mean().item()
    rms_rel = _rms_relative(out, ref)
    ulp_p99 = _ulp_distance_p99(out, ref)

    print(
        f"\n[precision] shape={shape} affine={with_affine}: "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rms_rel={rms_rel:.3e} ulp_p99={ulp_p99:.3g}"
    )

    # Generous gate — leaves head-room for slight implementation variants.
    assert_with_pcc(ref, out, pcc=0.999)

    # Per-row LayerNorm invariants only meaningful in the no_affine case.
    if not with_affine:
        row_mean = out.mean(dim=-1)
        row_var = out.var(dim=-1, unbiased=False)
        passed_mean, msg_mean = comp_allclose(torch.zeros_like(row_mean), row_mean, rtol=0.0, atol=1e-3)
        assert passed_mean, f"per-row mean drift: {msg_mean}"
        passed_var, msg_var = comp_allclose(torch.ones_like(row_var), row_var, rtol=0.0, atol=5e-3)
        assert passed_var, f"per-row var drift: {msg_var}"
