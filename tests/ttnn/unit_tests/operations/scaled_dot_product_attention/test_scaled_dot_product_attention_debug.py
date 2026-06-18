# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Debug regression test for the SDPA single_tile UNPACK-stall hang.

Root cause (fixed): Phase 0 pre-scaled Q with an *in-place* SFPU transform on
``cb_q_in`` (the reader-fed input CB):

    transform_in_place<cb_q_in>(B_q*DHt, MulUnary<>{scale_bits});

An in-place pop+push on a *remote-producer* (reader) CB nets to ZERO tiles
visible to the next same-thread ``cb_wait_front`` on the UNPACK thread. The
downstream QK ``matmul_block`` then blocks forever in
``cb_wait_front(cb_q_in, B_q*DHt)`` — the UNPACK thread never gets Q, while
MATH/PACK (whose ``cb_wait_front`` is a near-noop) sail through, giving the
misleading "UNPACK stuck inside matmul_block" symptom.

Bisection proof (DEVICE_PRINT on TR0): with the in-place pre-scale present,
``cb_wait_front(cb_q_in, 1)`` blocks while ``cb_wait_front(cb_k_in, ...)``
(K is untouched) returns all tiles. Disabling the pre-scale makes UNPACK see Q
and pass the matmul wait region.

Fix: do NOT fold scale into the reader-fed Q. Fold it into the scores instead —
``transform_in_place<cb_qk>(B_q*B_kv, MulUnary<>{scale_bits})`` right after the
QK matmul. cb_qk is locally matmul-produced, so the in-place transform is the
legal pattern (same as toy_variance's square_in_place on a locally-produced CB).
Mathematically identical: ``(Q*scale)·Kᵀ == scale*(Q·Kᵀ)``.

This test pins the single_tile (n_kv == 1) case that was hanging. It must
complete (not hang) and meet PCC.

NOTE: multi-KV-block cases (n_kv > 1: non_square, long_seq, cross_attn) have a
SEPARATE, independent defect in the online-softmax j>0 alpha-correction
recurrence — they are NOT covered here and were never reachable before the hang
was fixed.
"""

import math

import pytest
import torch

import ttnn

from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention


def _ref_sdpa(q, k, v, scale):
    # q,k,v: (B, H, S, D) torch float
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


@pytest.mark.parametrize("scale_mode", ["auto", "explicit"])
def test_sdpa_single_tile_no_hang(device, scale_mode):
    torch.manual_seed(0)
    B, H, S, D = 1, 1, 32, 64  # single_tile: n_kv == 1

    # Deterministic, small-magnitude inputs so softmax is well-conditioned and
    # the bf16 matmul stays within PCC tolerance.
    q = torch.randn(B, H, S, D, dtype=torch.float32) * 0.1
    k = torch.randn(B, H, S, D, dtype=torch.float32) * 0.1
    v = torch.randn(B, H, S, D, dtype=torch.float32) * 0.1

    scale = (1.0 / math.sqrt(D)) if scale_mode == "auto" else 0.25
    pass_scale = None if scale_mode == "auto" else scale

    ref = _ref_sdpa(q, k, v, scale)

    tq = ttnn.from_torch(q.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    tk = ttnn.from_torch(k.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    tv = ttnn.from_torch(v.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)

    out = scaled_dot_product_attention(tq, tk, tv, attention_mask=None, scale=pass_scale)
    got = ttnn.to_torch(out).to(torch.float32)

    pcc = torch.corrcoef(torch.stack([ref.flatten(), got.flatten()]))[0, 1].item()
    assert pcc >= 0.99, f"PCC {pcc:.5f} below 0.99 (scale_mode={scale_mode})"
