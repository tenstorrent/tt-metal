# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precise mode for the transformer ports (off by default: the graduated numerics stay as they were).

Why: a denoising run feeds every step's output into the next. Over 50 true-CFG steps (scale 4.0) the
per-forward error of the graduated ports (CFG noise PCC 0.99994, ~1% relative) drifts some samples'
trajectories apart (measured: final image PCC down to 0.52 while every stage passed per forward).
Two measured Wormhole floors drive it:
  * matmul inputs are rounded to bf16 before every projection (2^-9 relative per element);
  * a tile matmul accumulates its 32-term dot products at ~2^-12 of the largest term, whatever the
    input splitting. The joint-attention logits reach ~4e4, so QK^T carries absolute errors of whole
    units into a near-argmax softmax.
Precise mode carries matmul inputs as bf16 hi + lo (exact products against bf16 weights). For QK^T it
also splits K into 8 lanes (<= 4 nonzeros per 32-group, 8 apart: exact accumulation, measured 1.2e-6)
and takes the median of 3 rotated lane partitions (a rare power-of-two tile glitch is outvoted).
"""

from __future__ import annotations

import ttnn

ENABLED = False  # the pipeline switches this on; the per-component PCC tests keep the graduated path
EXACT_QK = True  # within precise mode: exact-lane QK^T (else the dense 3-term split)
EXACT_LANES = 8
PATTERNS = ("strided", "rot1", "rot3")
_MASKS = {}


def precise_config():
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


def split_bf16(x):
    x32 = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
    hi = ttnn.typecast(x32, ttnn.bfloat16)
    lo = ttnn.typecast(ttnn.subtract(x32, ttnn.typecast(hi, ttnn.float32)), ttnn.bfloat16)
    return hi, lo


def linear(x, w, bias=None, compute_kernel_config=None):
    """float32 x @ bf16 w -> float32, x as bf16 hi + lo; bias (float32) added separately."""
    cfg = precise_config()
    hi, lo = split_bf16(x)
    y = ttnn.add(
        ttnn.linear(hi, w, compute_kernel_config=cfg, dtype=ttnn.float32),
        ttnn.linear(lo, w, compute_kernel_config=cfg, dtype=ttnn.float32),
    )
    if bias is not None:
        y = ttnn.add(y, bias if bias.dtype == ttnn.float32 else ttnn.typecast(bias, ttnn.float32))
    return y


def _lane_of(k, pattern):
    j, g = k % 32, k // 32
    if pattern == "strided":
        return j % 8
    if pattern == "rot1":
        return (j % 8 + g) % 8
    return (j % 8 + 3 * g) % 8


def _masks(device, k, pattern):
    key = (id(device), k, pattern)
    if key not in _MASKS:
        lane = [_lane_of(i, pattern) for i in range(k)]
        ms = []
        for r in range(EXACT_LANES):
            vals = [1.0 if lane[i] == r else 0.0 for i in range(k)]
            ms.append(ttnn.typecast(ttnn.Tensor(vals, [1, k], ttnn.float32, ttnn.TILE_LAYOUT, device), ttnn.bfloat16))
        _MASKS[key] = ms
    return _MASKS[key]


def _median3(a, b, c):
    return ttnn.maximum(ttnn.minimum(a, b), ttnn.minimum(ttnn.maximum(a, b), c))


def exact_matmul_bt(a, b):
    """float32 a @ float32 b^T (reduction over the last dim of both) with exact accumulation:
    3-term bf16 split (ah.bh + ah.bl + al.bh), 8 exact lanes over K, median of 3 lane partitions."""
    cfg = precise_config()
    ah, al = split_bf16(a)
    bh, bl = split_bf16(b)
    k = a.shape[-1]
    ests = []
    for pattern in PATTERNS:
        y = None
        for m in _masks(a.device(), k, pattern):
            for pa, pb in ((ah, bh), (ah, bl), (al, bh)):
                t = ttnn.matmul(
                    ttnn.multiply(pa, m), pb, transpose_b=True, compute_kernel_config=cfg, dtype=ttnn.float32
                )
                y = t if y is None else ttnn.add(y, t)
        ests.append(y)
    return _median3(*ests)


def matmul(a, b):
    """float32 a @ float32 b -> float32 as the 3-term bf16 split (dense accumulation)."""
    cfg = precise_config()
    ah, al = split_bf16(a)
    bh, bl = split_bf16(b)
    mm = lambda p, q: ttnn.matmul(p, q, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
    return ttnn.add(ttnn.add(mm(ah, bh), mm(ah, bl)), mm(al, bh))


def matmul_bt(a, b):
    """float32 a @ float32 b^T as the dense 3-term bf16 split."""
    cfg = precise_config()
    ah, al = split_bf16(a)
    bh, bl = split_bf16(b)
    mm = lambda p, q: ttnn.matmul(p, q, transpose_b=True, compute_kernel_config=cfg, dtype=ttnn.float32)  # noqa: E731
    return ttnn.add(ttnn.add(mm(ah, bh), mm(ah, bl)), mm(al, bh))
