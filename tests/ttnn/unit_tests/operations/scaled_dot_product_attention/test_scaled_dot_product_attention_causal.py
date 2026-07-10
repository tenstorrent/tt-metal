# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 3 — native causal masking (is_causal=True).

Exercises the on-device triangular-bias path directly:
- numerical equivalence of is_causal=True vs an explicit triangular additive
  mask (the two must produce the same output — same math, different mechanism);
- across dtypes (bf16 / fp32 / bf8b), MHA / GQA / MQA, auto / explicit scale,
  single- and multi-block sequences, and non-tile-aligned S_q;
- the {causal, cross} EXCLUSION (S_q != S_kv) is refused;
- is_causal + attn_mask is a ValueError (mutually exclusive).
"""

import math

import pytest
import torch
import ttnn

from ttnn.operations._op_contract import ExcludedCell
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _tri_mask(B, S_q, S_kv, torch_dtype=torch.bfloat16):
    m = torch.zeros(B, 1, S_q, S_kv, dtype=torch_dtype)
    m.masked_fill_(torch.triu(torch.ones(S_q, S_kv, dtype=torch.bool), diagonal=1), float("-inf"))
    return m


def _ref(Q, K, V, *, is_causal, scale):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
    return torch.nn.functional.scaled_dot_product_attention(
        Qf, Kf, Vf, attn_mask=None, is_causal=is_causal, scale=scale
    )


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _rel_rms(a, b):
    a, b = a.float(), b.float()
    return (torch.sqrt(torch.mean((a - b) ** 2)) / (b.std() + 1e-8)).item()


_DT = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32, ttnn.bfloat8_b: torch.bfloat16}
_TOL = {ttnn.bfloat16: (0.995, 0.05), ttnn.float32: (0.999, 0.02), ttnn.bfloat8_b: (0.99, 0.12)}


def _cfg(fp32_acc=True):
    return ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=fp32_acc, math_approx_mode=False
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 32, 32),
        (1, 1, 128, 64),
        (1, 1, 256, 64),
        (1, 4, 128, 64),
        (2, 4, 256, 128),
    ],
)
def test_causal_matches_reference(device, dtype, shape):
    B, H, S, D = shape
    # Large head_dim in fp32 doubles CB bytes and OOMs L1 — pre-existing
    # boundary owned by Refinement 4 (identical for custom-mask / none), not a
    # causal-specific limit. Skip that combo so this test isolates causal.
    if dtype == ttnn.float32 and D >= 128:
        pytest.skip("fp32 + large head_dim OOMs L1 (Refinement 4 scope)")
    torch_dtype = _DT[dtype]
    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=torch_dtype)
    K = torch.randn(shape, dtype=torch_dtype)
    V = torch.randn(shape, dtype=torch_dtype)
    scale = 1.0 / math.sqrt(D)

    expected = _ref(Q, K, V, is_causal=True, scale=scale)

    tq = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=_cfg())
    res = ttnn.to_torch(out)

    pcc_min, rms_max = _TOL[dtype]
    pcc, rms = _pcc(res, expected), _rel_rms(res, expected)
    assert pcc >= pcc_min, f"PCC {pcc:.5f} < {pcc_min} (rms {rms:.4f}) shape={shape} dtype={dtype}"
    assert rms <= rms_max, f"relRMS {rms:.4f} > {rms_max} (pcc {pcc:.5f}) shape={shape} dtype={dtype}"


@pytest.mark.parametrize("kv_heads", ["mha", "gqa", "mqa"])
def test_causal_gqa_mqa(device, kv_heads):
    B, H_q, S, D = 1, 8, 128, 64
    H_kv = {"mha": 8, "gqa": 2, "mqa": 1}[kv_heads]
    torch.manual_seed(0)
    Q = torch.randn(B, H_q, S, D, dtype=torch.bfloat16)
    K = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    V = torch.randn(B, H_kv, S, D, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)
    expected = _ref(Q, K, V, is_causal=True, scale=scale)

    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=_cfg())
    res = ttnn.to_torch(out)
    assert _pcc(res, expected) >= 0.995, f"{kv_heads}: PCC {_pcc(res, expected):.5f}"


def test_causal_explicit_scale(device):
    shape = (1, 2, 128, 64)
    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    expected = _ref(Q, K, V, is_causal=True, scale=0.125)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, scale=0.125, compute_kernel_config=_cfg())
    assert _pcc(ttnn.to_torch(out), expected) >= 0.995


def test_causal_equals_explicit_triangular_mask(device):
    """is_causal=True must equal the explicit triangular additive-mask path."""
    shape = (1, 2, 128, 64)
    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    out_causal = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=_cfg()))

    mask = _tri_mask(shape[0], shape[2], shape[2])
    tmask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_mask = ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv, attn_mask=tmask, compute_kernel_config=_cfg()))
    assert _pcc(out_causal, out_mask) >= 0.999, f"causal vs explicit-mask PCC {_pcc(out_causal, out_mask):.5f}"


@pytest.mark.parametrize("shape", [(1, 1, 47, 64), (1, 4, 100, 64)])
def test_causal_non_aligned_seq(device, shape):
    """h_non_aligned (S_q % 32 != 0) causal — best-effort corner."""
    B, H, S, D = shape
    torch.manual_seed(0)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    scale = 1.0 / math.sqrt(D)
    expected = _ref(Q, K, V, is_causal=True, scale=scale)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=_cfg())
    res = ttnn.to_torch(out)
    pcc, rms = _pcc(res, expected), _rel_rms(res, expected)
    assert pcc >= 0.99, f"non-aligned causal PCC {pcc:.5f} rms {rms:.4f} shape={shape}"


def test_causal_cross_excluded(device, expect_error):
    """causal + cross-attn (S_q != S_kv) must be refused (EXCLUSION)."""
    Q = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    K = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16)
    V = torch.randn(1, 4, 128, 64, dtype=torch.bfloat16)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(ExcludedCell, "unsupported combination"):
        scaled_dot_product_attention(tq, tk, tv, is_causal=True, compute_kernel_config=_cfg())


def test_causal_and_mask_mutually_exclusive(device, expect_error):
    Q = torch.randn(1, 1, 128, 64, dtype=torch.bfloat16)
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    mask = _tri_mask(1, 128, 128)
    tmask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(ValueError, "mutually exclusive"):
        scaled_dot_product_attention(tq, tq, tq, attn_mask=tmask, is_causal=True, compute_kernel_config=_cfg())
