# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 1b — coarse-chunk + partial-remainder chunking.

DO NOT DELETE — documents the R1b verification surface.

R1 kept the `_chunk_size` largest-divisor trick, so every chunk was whole and no
partial-CHUNK kernel path existed (the only partial unit was the last S_kv tile's
columns, masked to −∞). R1b replaces the divisor trick with a coarse chunk
`min(axis_t, 4)` + a partial last chunk of `axis_t % 4` tiles, threaded through the
reader/compute/writer as a per-chunk runtime tile count `min(chunk_t, axis_t −
j·chunk_t)` — for both the Sq q-chunk (M extent) and the Skv loop (QKᵀ N / PV K
extent), with the matmul N-subblock decomposition re-derived on-device for the
partial N.

The divisor trick collapsed a prime tile-count > 4 to a 1-tile chunk (e.g.
Skv_t=101 → chunk 1, repaying per-chunk reconfig/init/fill-drain every tile). These
shapes force partial chunks and would run at chunk 1 under the old rule; they run at
chunk 4 now and must stay correct:

  * Sq_t / Skv_t prime > 4 (5, 7, 101) — the granularity-floor case.
  * Sq_t / Skv_t = 6 (coarse 4 + partial 2) — the common non-divisible-by-4 case
    the golden suite already exercises via (2,3,192,96); pinned here explicitly.
  * partial chunk + S_kv%32≠0 (KV column pad on the last partial chunk's boundary
    tile) — the R1 mask riding on a partial R1b chunk, uncovered by any golden cell.
  * multi-core partial q-chunks (some cores get the partial last q-chunk).

Reference matches torch F.sdpa (fp32) via the same helper as the R1 unit test.
"""

import math

import pytest
import torch

import ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

# Golden bf16 + fp32-DEST tolerances (helpers.TOLERANCES[(bfloat16, True)]).
PCC = 0.995
REL_RMS = 0.05


def _check(ref, out):
    ref = ref.float()
    out = out.float()
    rf, of = ref.flatten(), out.flatten()
    rf = rf - rf.mean()
    of = of - of.mean()
    pcc = (rf @ of / (rf.norm() * of.norm() + 1e-12)).item()
    rel_rms = (torch.sqrt(((ref - out) ** 2).mean()) / (ref.std() + 1e-12)).item()
    assert pcc >= PCC, f"PCC {pcc:.5f} < {PCC}  (rel_rms={rel_rms:.4f})"
    assert rel_rms <= REL_RMS, f"rel_rms {rel_rms:.4f} > {REL_RMS}  (pcc={pcc:.5f})"


def attention_reference(q, k, v, attn_mask=None, scale=None):
    B, H, Sq, D = q.shape
    Hkv = k.shape[1]
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    q, k, v = q.float(), k.float(), v.float()
    if Hkv != H:
        rep = H // Hkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask.float()
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v)


def _to_device(t, device, dtype):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def run_sdpa(device, q_shape, k_shape, v_shape, *, mask_mode="none", scale_mode="auto"):
    torch.manual_seed(1234)
    dtype = ttnn.bfloat16
    q = torch.randn(q_shape)
    k = torch.randn(k_shape)
    v = torch.randn(v_shape)

    scale = None if scale_mode == "auto" else 0.25

    attn_mask = None
    tt_mask = None
    if mask_mode == "custom":
        B, _, Sq, _ = q_shape
        Skv = k_shape[-2]
        attn_mask = torch.randn(B, 1, Sq, Skv) * 2.0
        tt_mask = _to_device(attn_mask, device, dtype)

    ref = attention_reference(q, k, v, attn_mask=attn_mask, scale=scale)

    tq = _to_device(q, device, dtype)
    tk = _to_device(k, device, dtype)
    tv = _to_device(v, device, dtype)

    out = scaled_dot_product_attention(tq, tk, tv, attn_mask=tt_mask, scale=scale)
    assert list(out.shape) == list(q_shape), f"shape {list(out.shape)} != {list(q_shape)}"
    _check(ref, ttnn.to_torch(out))


# Shapes whose tile-count forces a partial last chunk under the coarse-4 scheme
# (and would collapse to a 1-tile chunk under the old divisor trick when prime>4).
PARTIAL_CHUNK = {
    # Sq_t=Skv_t=5 (prime > 4): coarse chunk 4 + partial 1. Aligned.
    "prime5_self": ((1, 1, 160, 64), (1, 1, 160, 64), (1, 1, 160, 64)),
    # Sq_t=6 (coarse 4 + partial 2), multi-head — the common !%4 case.
    "six_self_mh": ((1, 2, 192, 64), (1, 2, 192, 64), (1, 2, 192, 64)),
    # Cross-attn, both prime: Sq_t=5 (chunk 4, rem 1), Skv_t=7 (chunk 3, rem 1 —
    # rem 3 would violate rem|chunk, so the coarse selector drops to 3).
    "prime_cross": ((1, 1, 160, 64), (1, 1, 224, 64), (1, 1, 224, 64)),
    # Multi-core partial q-chunks: total_work = 2*4*ceil(5/4)=16 units; some cores
    # get the partial last q-chunk (qc=1 -> sq_valid=1).
    "prime5_batch_mh": ((2, 4, 160, 64), (2, 4, 160, 64), (2, 4, 160, 64)),
    # GQA on a prime shape (KV-head reuse across the partial chunk).
    "prime5_gqa": ((1, 8, 160, 64), (1, 2, 160, 64), (1, 2, 160, 64)),
    # partial chunk + S_kv%32!=0: Sq_t=Skv_t=5 (partial), S=143 -> 143%32=15 (KV
    # column pad on the last partial chunk's boundary tile) + partial q rows.
    "prime5_kvpad_self": ((1, 1, 143, 64), (1, 1, 143, 64), (1, 1, 143, 64)),
    # w_non_aligned (D%32!=0) on a partial chunk: D=96 -> Dt=3, Skv_t=5.
    "prime5_wpad_self": ((1, 1, 160, 96), (1, 1, 160, 96), (1, 1, 160, 96)),
}

# The verifier's headline shape — large prime tile-count (Skv_t=101). Runs at
# chunk 4 (26 chunks) instead of the divisor trick's chunk 1 (101 chunks).
LARGE_PRIME = {
    "prime101_self": ((1, 1, 3232, 64), (1, 1, 3232, 64), (1, 1, 3232, 64)),
}


@pytest.mark.parametrize("key", list(PARTIAL_CHUNK.keys()))
def test_partial_chunk_none(device, key):
    q, k, v = PARTIAL_CHUNK[key]
    run_sdpa(device, q, k, v, mask_mode="none")


@pytest.mark.parametrize("key", list(PARTIAL_CHUNK.keys()))
def test_partial_chunk_custom(device, key):
    q, k, v = PARTIAL_CHUNK[key]
    run_sdpa(device, q, k, v, mask_mode="custom")


@pytest.mark.parametrize("key", list(PARTIAL_CHUNK.keys()))
def test_partial_chunk_explicit_scale(device, key):
    q, k, v = PARTIAL_CHUNK[key]
    run_sdpa(device, q, k, v, mask_mode="none", scale_mode="explicit")


@pytest.mark.parametrize("key", list(LARGE_PRIME.keys()))
def test_large_prime_tilecount(device, key):
    q, k, v = LARGE_PRIME[key]
    run_sdpa(device, q, k, v, mask_mode="none")
