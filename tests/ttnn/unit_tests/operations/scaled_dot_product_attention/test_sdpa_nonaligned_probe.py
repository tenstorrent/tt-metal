# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0
#
# Probe (kept as a debug artifact, DO NOT DELETE): characterize how
# ttnn.from_torch(TILE_LAYOUT) pads non-tile-aligned tensors, and verify the
# basic non-aligned SDPA paths in isolation.

import math

import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def _ref(Q, K, V, *, mask=None, scale=None):
    Qf, Kf, Vf = Q.float(), K.float(), V.float()
    H_q, H_kv = Qf.shape[1], Kf.shape[1]
    if H_q != H_kv:
        r = H_q // H_kv
        Kf, Vf = Kf.repeat_interleave(r, 1), Vf.repeat_interleave(r, 1)
    s = scale if scale is not None else 1.0 / math.sqrt(Qf.shape[-1])
    scores = (Qf @ Kf.transpose(-2, -1)) * s
    if mask is not None:
        scores = scores + mask.float()
    return (torch.softmax(scores, dim=-1) @ Vf).to(Q.dtype)


def test_padding_values_w_non_aligned(device):
    """Inspect the padded columns of a TILE_LAYOUT tensor with D=50 (pad to 64)."""
    t = torch.arange(1, 1 * 1 * 32 * 50 + 1, dtype=torch.float32).reshape(1, 1, 32, 50)
    tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    print("logical shape:", tt.shape, "padded:", tt.padded_shape)
    rm = ttnn.to_torch(ttnn.to_layout(tt, ttnn.ROW_MAJOR_LAYOUT))
    print("rm readback shape:", rm.shape)
    # readback is logical (32,50) — to see padding we need the physical tile.
    # Use the device tensor's tile-padded readback via tilized->untilize roundtrip
    # isn't enough; instead check that the op-relevant invariant holds via a
    # tiny matmul: pad cols must be zero so Q·Kᵀ over D is correct.


def test_w_non_aligned_skv_aligned(device):
    """D non-aligned (47), S_q and S_kv aligned (64). Isolates the D-padding edge
    — no key-padding mask needed. Should 'just work' once tile counts use ceil."""
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    torch.manual_seed(0)
    shape = (1, 1, 64, 47)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    expected = _ref(Q, K, V)
    qq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(qq, kk, vv)
    res = ttnn.to_torch(out)
    print("out shape:", res.shape)
    assert_with_pcc(expected.float(), res.float(), 0.99)


def test_h_non_aligned_and_skv_non_aligned(device):
    """S_q=47 and S_kv=47 both non-aligned. Exercises the key-padding mask edge."""
    from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

    torch.manual_seed(0)
    shape = (1, 1, 47, 64)
    Q = torch.randn(shape, dtype=torch.bfloat16)
    K = torch.randn(shape, dtype=torch.bfloat16)
    V = torch.randn(shape, dtype=torch.bfloat16)
    expected = _ref(Q, K, V)
    qq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    kk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    vv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = scaled_dot_product_attention(qq, kk, vv)
    res = ttnn.to_torch(out)
    print("out shape:", res.shape)
    assert_with_pcc(expected.float(), res.float(), 0.99)
