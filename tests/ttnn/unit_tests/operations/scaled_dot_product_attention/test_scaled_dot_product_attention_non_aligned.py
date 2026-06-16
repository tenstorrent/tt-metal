# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""R3 — non-tile-aligned shapes (S_q / S_kv / D) for scaled_dot_product_attention.

Exercises the in-kernel edge handling added in Refinement 3:
  - w_non_aligned (D % 32 != 0): head dim padded to a tile; padding lanes
    contribute 0 to QK^T / PV (rely on ttnn zero-fill).
  - h_non_aligned (S_q % 32 != 0, D aligned): query padding rows discarded;
    KV-sequence padding columns masked to -inf via the on-device edge mask.
  - both non-aligned (incl. cross-attention with non-aligned S_kv).

Mirrors the non-aligned INPUTS in
eval/golden_tests/scaled_dot_product_attention/feature_spec.py.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


PCC_TOLERANCE = {
    ttnn.float32: 0.999,
    ttnn.bfloat16: 0.995,
    ttnn.bfloat8_b: 0.99,
}

_TORCH_DTYPE = {
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
}


def pcc(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def reference_sdpa(Q, K, V, *, attn_mask=None, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    if Kf.shape[1] != Qf.shape[1]:  # GQA / MQA head broadcast
        r = Qf.shape[1] // Kf.shape[1]
        Kf = Kf.repeat_interleave(r, dim=1)
        Vf = Vf.repeat_interleave(r, dim=1)
    D = Qf.shape[-1]
    s = scale if scale is not None else 1.0 / math.sqrt(D)
    scores = torch.matmul(Qf, Kf.transpose(-2, -1)) * s
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, Vf).to(Q.dtype)


# (Q_shape, K_shape, V_shape) — the non-aligned INPUTS from the golden spec.
NON_ALIGNED_SHAPES = [
    pytest.param((1, 1, 32, 50), (1, 1, 32, 50), id="Q1x1x32x50_Dna"),
    pytest.param((1, 1, 47, 64), (1, 1, 47, 64), id="Q1x1x47x64_Sna"),
    pytest.param((1, 1, 50, 50), (1, 1, 50, 50), id="Q1x1x50x50_both"),
    pytest.param((1, 4, 47, 64), (1, 4, 47, 64), id="Q1x4x47x64_Sna_mh"),
    pytest.param((2, 4, 100, 64), (2, 4, 100, 64), id="Q2x4x100x64_Sna_mb"),
    pytest.param((1, 8, 64, 47), (1, 8, 64, 47), id="Q1x8x64x47_Dna_mh"),
    pytest.param((1, 12, 33, 50), (1, 12, 33, 50), id="Q1x12x33x50_both_mh"),
    pytest.param((1, 8, 47, 64), (1, 2, 47, 64), id="Q1x8x47x64_Sna_gqa"),
    pytest.param((1, 8, 47, 64), (1, 1, 47, 64), id="Q1x8x47x64_Sna_mqa"),
    pytest.param((1, 4, 100, 50), (1, 4, 47, 50), id="Q1x4x100x50_both_cross"),
]

MASK_MODES = [pytest.param("none", id="mask_none"), pytest.param("custom", id="mask_custom")]
SCALE_MODES = [pytest.param("auto", id="scale_auto"), pytest.param("explicit", id="scale_explicit")]
DTYPES = [pytest.param(ttnn.bfloat16, id="bf16"), pytest.param(ttnn.float32, id="fp32")]

EXPLICIT_SCALE = 0.125


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("q_shape,k_shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("mask_mode", MASK_MODES)
@pytest.mark.parametrize("scale_mode", SCALE_MODES)
def test_non_aligned(device, dtype, q_shape, k_shape, mask_mode, scale_mode):
    torch_dtype = _TORCH_DTYPE[dtype]
    B, H_q, S_q, D = q_shape
    _, H_kv, S_kv, _ = k_shape

    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(k_shape, dtype=torch_dtype)
    V = torch.randn(k_shape, dtype=torch_dtype)

    if mask_mode == "custom":
        torch_mask = torch.randn(B, 1, S_q, S_kv, dtype=torch_dtype)
    else:
        torch_mask = None

    scale = EXPLICIT_SCALE if scale_mode == "explicit" else None
    expected = reference_sdpa(Q, K, V, attn_mask=torch_mask, scale=scale)

    qt = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    kt = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    vt = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    mt = (
        ttnn.from_torch(torch_mask, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if torch_mask is not None
        else None
    )

    out = scaled_dot_product_attention(qt, kt, vt, attn_mask=mt, scale=scale)
    assert list(out.shape) == [B, H_q, S_q, D]
    result = ttnn.to_torch(out)

    correlation = pcc(result, expected)
    assert correlation >= PCC_TOLERANCE[dtype], f"PCC too low: {correlation:.6f} ({q_shape}, {mask_mode}, {scale_mode})"


def test_bfloat8_b_w_non_aligned_excluded(device):
    """bf8b + D-non-aligned is refused (block-float padding corrupts QK^T)."""
    Q = torch.randn(1, 1, 32, 50, dtype=torch.bfloat16)
    qt = ttnn.from_torch(Q, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(NotImplementedError):
        scaled_dot_product_attention(qt, qt, qt)
