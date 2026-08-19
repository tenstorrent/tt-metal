# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""Authoritative precision characterization for ttnn.operations.rms_norm.

Refinement 1 (numerical configurability) opened three new precision axes:
`bfloat8_b` activations, `bfloat8_b` gamma, and `fp32_dest_acc_en=False`.  This is
the single matrix that pins all of them, per /numeric-formats-metal S10:

    shape (aligned / W-non-aligned / H-non-aligned / both / wide)
      x dtype (bfloat16, float32, bfloat8_b)
      x fp32_dest_acc_en (True, False)
      x math_fidelity (HiFi4, HiFi2)

Two assertions, because they fail on DIFFERENT bugs:

  * PCC - the shape of the output.

  * the ROW-SCALE BIAS - the mean relative error of the per-row 1/rms factor,
    recovered by a least-squares fit of `out ~ k * x` per row.  This is the gate
    that matters for an RMS norm and PCC is nearly blind to it: the 16-bit-DEST
    sum-of-squares overestimate Refinement 1 fixed showed PCC 0.99995 while every
    row was scaled 4.8% low (op_requirements.md Refinement 1, changelog).  A
    uniform scale error is exactly what folding tile padding into the denominator
    or losing an accumulator to rounding looks like.

Results table: precision_matrix_results.md (regenerate with -s).
"""

import pytest
import torch

import ttnn
from ttnn.operations.rms_norm import rms_norm

EPS = 1e-6

# PCC floor per input dtype (/numeric-formats-metal S11: the precision matrix runs
# every fidelity and both DEST widths, so it gates at 0.99).
PCC_FLOOR = {ttnn.bfloat16: 0.99, ttnn.float32: 0.99, ttnn.bfloat8_b: 0.99}

# A row scale is either right or structurally wrong; rounding never reaches 2%.
MAX_ROW_SCALE_BIAS = 0.02

SHAPES = [
    pytest.param((1, 1, 32, 64), id="32x64_aligned"),
    pytest.param((1, 1, 64, 128), id="64x128_aligned"),
    pytest.param((1, 1, 32, 50), id="32x50_W_non_aligned"),
    pytest.param((1, 1, 48, 64), id="48x64_H_non_aligned"),
    pytest.param((1, 1, 17, 50), id="17x50_both_non_aligned"),
    # Wide hidden: the regime where the accumulator precision actually bites
    # (the fixed bias grew with the reduced width, +10.4% at Wt=224).
    pytest.param((1, 1, 32, 7168), id="32x7168_wide"),
]


def _row_scale_bias(xg, out, s_ref):
    """Mean relative error of the per-row 1/rms factor.

    Least-squares fit of `out ~ k * xg` per row, where `xg = x * gamma` is
    everything the kernel applies APART from the row scale.  The regressor must
    include gamma: fitting against x alone cancels to ~0 for a random-sign gamma
    and the estimator reads k ~ 0 for a perfectly correct kernel.
    """
    gf = xg.reshape(-1, xg.shape[-1])
    of = out.reshape(-1, out.shape[-1])
    k = (of * gf).sum(-1) / (gf * gf).sum(-1).clamp_min(1e-30)
    return ((k / s_ref.reshape(-1)) - 1.0).mean().item()


@pytest.mark.parametrize("math_fidelity", [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi2], ids=["HiFi4", "HiFi2"])
@pytest.mark.parametrize("fp32_acc", [True, False], ids=["fp32_acc", "bf16_acc"])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b], ids=["bf16", "fp32", "bfp8"])
@pytest.mark.parametrize("shape", SHAPES)
def test_rms_norm_precision_matrix(device, shape, dtype, fp32_acc, math_fidelity):
    if dtype == ttnn.float32 and not fp32_acc:
        pytest.skip("EXCLUSIONS: {dtype: float32, fp32_dest_acc_en: False} — fp32 through a bf16 DEST accumulator")
    if dtype == ttnn.bfloat8_b and (shape[-1] % 32 or shape[-2] % 32):
        pytest.skip("feature_spec.INVALID: bfloat8_b x non-tile-aligned shape")

    torch.manual_seed(0)
    W = shape[-1]
    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    torch_x = torch.randn(shape).to(torch_dtype)
    torch_gamma = torch.randn(W).reshape(1, 1, 1, W).to(torch_dtype)

    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = math_fidelity
    cfg.fp32_dest_acc_en = fp32_acc
    cfg.math_approx_mode = False

    tt_x = ttnn.from_torch(torch_x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tt_gamma = ttnn.from_torch(torch_gamma, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # Reference uses the values the DEVICE actually holds, so block-float
    # quantization of the inputs is not charged to the kernel.
    x32 = ttnn.to_torch(tt_x).float()
    g32 = ttnn.to_torch(tt_gamma).float()[..., :W]
    s_ref = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + EPS)
    expected = x32 * s_ref * g32

    actual = ttnn.to_torch(rms_norm(tt_x, gamma=tt_gamma, epsilon=EPS, compute_kernel_config=cfg)).float()

    err = (actual - expected).abs()
    rel_rms = err.pow(2).mean().sqrt().item() / expected.std().clamp_min(1e-30).item()
    pcc = torch.corrcoef(torch.stack([expected.flatten().double(), actual.flatten().double()]))[0, 1].item()
    bias = _row_scale_bias(x32 * g32, actual, s_ref)

    print(
        f"\nPRECISION_MATRIX shape={tuple(shape)} dtype={dtype} fp32_acc={fp32_acc} "
        f"fidelity={math_fidelity} pcc={pcc:.6f} rel_rms={rel_rms:.5f} "
        f"row_scale_bias={bias:+.5f} max_abs={err.max().item():.5f} "
        f"median_abs={err.median().item():.6f} p99_abs={torch.quantile(err.flatten().float(), 0.99).item():.5f}"
    )

    assert abs(bias) < MAX_ROW_SCALE_BIAS, (
        f"row-scale bias {bias:+.5f} exceeds {MAX_ROW_SCALE_BIAS} — a UNIFORM scale error "
        f"(padding folded into the denominator, or an accumulator lost to rounding), not noise"
    )
    assert pcc >= PCC_FLOOR[dtype], f"PCC {pcc:.6f} < {PCC_FLOOR[dtype]} (rel_rms={rel_rms:.5f})"
