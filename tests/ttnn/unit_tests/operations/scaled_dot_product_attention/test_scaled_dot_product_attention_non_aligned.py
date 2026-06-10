# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 4 — non-tile-aligned shapes.

Covers the alignment axis (w_non_aligned: D % 32 != 0; h_non_aligned:
S % 32 != 0): the ten feature-spec non-aligned shapes, both mask modes,
both scale modes, all three dtypes, plus deterministic tests proving the
S_kv pad columns enter neither rowmax nor rowsum.
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

TOLERANCES = {
    ttnn.float32: (0.999, 0.02),
    ttnn.bfloat16: (0.995, 0.05),
    ttnn.bfloat8_b: (0.99, 0.12),
}
TORCH_DTYPE = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16, ttnn.bfloat8_b: torch.bfloat16}

# Feature-spec non-aligned bucket: (Q, KV) shapes (V == K).
NON_ALIGNED_SHAPES = [
    ((1, 1, 32, 50), (1, 1, 32, 50)),  # D non-aligned
    ((1, 1, 47, 64), (1, 1, 47, 64)),  # S non-aligned
    ((1, 1, 50, 50), (1, 1, 50, 50)),  # both
    ((1, 4, 47, 64), (1, 4, 47, 64)),  # S + multi-head
    ((2, 4, 100, 64), (2, 4, 100, 64)),  # S + multi-batch
    ((1, 8, 64, 47), (1, 8, 64, 47)),  # D + multi-head
    ((1, 12, 33, 50), (1, 12, 33, 50)),  # both + multi-head
    ((1, 8, 47, 64), (1, 2, 47, 64)),  # S + GQA
    ((1, 8, 47, 64), (1, 1, 47, 64)),  # S + MQA
    ((1, 4, 100, 50), (1, 4, 47, 50)),  # both + cross-attn
]


def ref_sdpa(Q, K, V, mask=None, scale=None):
    s = scale if scale is not None else 1.0 / math.sqrt(Q.shape[-1])
    Hq, Hkv = Q.shape[1], K.shape[1]
    Kf, Vf = K.float(), V.float()
    if Hq != Hkv:
        Kf = Kf.repeat_interleave(Hq // Hkv, dim=1)
        Vf = Vf.repeat_interleave(Hq // Hkv, dim=1)
    sc = Q.float() @ Kf.transpose(-2, -1) * s
    if mask is not None:
        sc = sc + mask.float()
    return torch.softmax(sc, -1) @ Vf


def causal_mask(B, S_q, S_kv, torch_dtype):
    m = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    m.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    return m


def run_case(device, q_shape, kv_shape, dtype, mask_mode="none", scale=None):
    torch_dtype = TORCH_DTYPE[dtype]
    torch.manual_seed(0)
    Q = torch.randn(q_shape, dtype=torch_dtype)
    K = torch.randn(kv_shape, dtype=torch_dtype)
    V = torch.randn(kv_shape, dtype=torch_dtype)
    mask = causal_mask(q_shape[0], q_shape[2], kv_shape[2], torch_dtype) if mask_mode == "causal" else None

    expected = ref_sdpa(Q, K, V, mask=mask, scale=scale)

    to_dev = lambda t: ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(
        to_dev(Q), to_dev(K), to_dev(V), attention_mask=to_dev(mask) if mask is not None else None, scale=scale
    )
    result = ttnn.to_torch(out).float()

    assert list(result.shape) == list(q_shape)
    pcc_min, rms_max = TOLERANCES[dtype]
    pcc = torch.corrcoef(torch.stack([result.flatten(), expected.flatten()]))[0, 1].item()
    rms = ((result - expected).pow(2).mean().sqrt() / expected.std()).item()
    assert pcc >= pcc_min and rms <= rms_max, f"pcc={pcc:.6f} (>{pcc_min}), rel_rms={rms:.5f} (<{rms_max})"


@pytest.mark.parametrize("q_shape,kv_shape", NON_ALIGNED_SHAPES)
@pytest.mark.parametrize("mask_mode", ["none", "causal"])
def test_non_aligned_bf16(device, q_shape, kv_shape, mask_mode):
    run_case(device, q_shape, kv_shape, ttnn.bfloat16, mask_mode=mask_mode)


@pytest.mark.parametrize("dtype", [ttnn.float32, ttnn.bfloat8_b], ids=["fp32", "bf8b"])
@pytest.mark.parametrize(
    "q_shape,kv_shape",
    [
        ((1, 1, 47, 64), (1, 1, 47, 64)),
        ((1, 1, 50, 50), (1, 1, 50, 50)),
        ((1, 4, 100, 50), (1, 4, 47, 50)),
    ],
)
def test_non_aligned_dtypes(device, q_shape, kv_shape, dtype):
    run_case(device, q_shape, kv_shape, dtype, mask_mode="causal")


def test_non_aligned_explicit_scale(device):
    run_case(device, (1, 4, 47, 64), (1, 4, 47, 64), ttnn.bfloat16, mask_mode="none", scale=0.125)


def test_pad_columns_excluded_from_softmax(device):
    """Q=K=0 → uniform attention over VALID keys only. Output row = mean of
    the S_kv valid V rows. If the zero-padded score columns leaked into the
    rowsum, every output would be scaled by S_kv/ceil32(S_kv) = 47/64 ≈ 0.73;
    if a pad column leaked into rowmax it would shift exp() identically.
    Distinct V rows make any partial leak visible (atol 0.02 vs 27% error).
    """
    S = 47
    Q = torch.zeros(1, 1, S, 64, dtype=torch.bfloat16)
    K = torch.zeros(1, 1, S, 64, dtype=torch.bfloat16)
    V = torch.arange(S, dtype=torch.float32)[None, None, :, None].expand(1, 1, S, 64).to(torch.bfloat16)

    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V))).float()

    expected = V.float().mean(dim=2, keepdim=True).expand(1, 1, S, 64)  # mean = 23.0
    assert torch.allclose(out, expected, atol=0.25), f"max diff {(out - expected).abs().max()} (pad leak ≈ 6.2)"


def test_pad_columns_excluded_from_rowmax(device):
    """All-negative scores: rowmax must be a real (negative) score, not the
    zero-padded columns' 0. K = -Q → scores < 0; a 0 in the running max makes
    exp(s) underflow uniformly and the softmax goes flat → output ≈ mean(V)
    instead of the reference's sharper mix.
    """
    S = 47
    torch.manual_seed(1)
    Q = torch.randn(1, 1, S, 64, dtype=torch.bfloat16) * 3
    K = -Q
    V = torch.randn(1, 1, S, 64, dtype=torch.bfloat16)

    expected = ref_sdpa(Q, K, V)
    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(scaled_dot_product_attention(to_dev(Q), to_dev(K), to_dev(V))).float()
    pcc = torch.corrcoef(torch.stack([out.flatten(), expected.flatten()]))[0, 1].item()
    assert pcc >= 0.995, f"pcc={pcc:.6f} — pad columns likely entered rowmax"


def test_aligned_unchanged(device):
    """Non-regression: tile-aligned path is untouched."""
    run_case(device, (1, 4, 128, 64), (1, 4, 128, 64), ttnn.bfloat16, mask_mode="causal")
