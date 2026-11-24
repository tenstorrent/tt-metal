# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Device unit tests for the GLM-5.3-Flash NoPE MLA / DSA attention block.

Reference: the formula in ``tt/mla.py``'s docstring (GLM ``modeling_glm5_next.py:1100-1275``), which
is plain MLA in the absorbed form with **no rope tail** (``qk_rope_head_dim = 0``):

    q_r    = RMSNorm_1536(x W_qa);  q = (q_r W_qb).view(H, 256)
    c      = RMSNorm_512(x W_kva)                      -> the only KV cache entry (latent, 512/token)
    q_lat  = W_uk_h^T q_h   (512 per head)             (absorbed; exact only because there is no rope part)
    o_lat  = softmax(q_lat . c / sqrt(256)) c           (dense causal latent attention)
    out    = concat_h(W_uv_h o_lat_h) W_o

The absorber identity is what the test checks hardest: ``W_uk^T q`` then ``. c`` must equal
``q . (W_uk c)``. With a rope tail that equality fails; NoPE is what makes the fast path legal.

The DSA indexer is exercised through its contract only (``check_dense_exact``), because this decoder
implements the dense-exact regime and rejects longer contexts -- see ``test_config_precision.py``.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

import ttnn
from models.experimental.glm53_flash.tt.mla import MLA

pytestmark = pytest.mark.use_module_device({"l1_small_size": 24576})

DRAM = ttnn.DRAM_MEMORY_CONFIG

# Small but structurally faithful MLA instance.
H = 8  # heads
HIDDEN = 512
Q_LORA = 128
KV_LORA = 64
DQK = 64  # nope width (no rope part by construction)
DV = 64
BLOCK = 64
DECODE_H = 32  # MLA.decode slices heads in groups of decode_heads = 32


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten().float(), b.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def to_dev(t, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)


def dev_to_tor(t) -> torch.Tensor:
    return ttnn.to_torch(t).float()


def build_mla(device, seed=0, heads=H):
    torch.manual_seed(seed)
    w_qa_kva = torch.randn(HIDDEN, Q_LORA + KV_LORA) * 0.05  # [q_a | kv_a]
    qa_norm = torch.rand(Q_LORA) + 0.5
    kva_norm = torch.rand(KV_LORA) + 0.5
    w_qb = torch.randn(Q_LORA, heads * DQK) * 0.05
    w_uk = torch.randn(heads, DQK, KV_LORA) * 0.05
    w_uv = torch.randn(heads, KV_LORA, DV) * 0.05
    w_o = torch.randn(heads * DV, HIDDEN) * 0.05

    mla = MLA(
        device,
        w_qa_kva=to_dev(w_qa_kva, device, ttnn.bfloat8_b),
        qa_norm=to_dev(qa_norm, device),
        kva_norm=to_dev(kva_norm, device),
        w_qb=to_dev(w_qb, device, ttnn.bfloat8_b),
        w_uk=to_dev(w_uk, device),
        w_uv=to_dev(w_uv, device),
        w_o=to_dev(w_o, device, ttnn.bfloat8_b),
        num_heads=heads,
        q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA,
        qk_head_dim=DQK,
        v_head_dim=DV,
        eps=1e-5,
        block_size=BLOCK,
    )
    ref = dict(w_qa_kva=w_qa_kva, qa_norm=qa_norm, kva_norm=kva_norm, w_qb=w_qb, w_uk=w_uk, w_uv=w_uv, w_o=w_o)
    return mla, ref


def mla_reference(x, w_qa_kva, qa_norm, kva_norm, w_qb, w_uk, w_uv, w_o, eps, *, cache=None):
    """``x [T, HIDDEN]`` -> (out [T, HIDDEN], latent [T, KV_LORA]) with causal latent attention."""
    T = x.shape[0]
    qk = x @ w_qa_kva
    q_r, c = qk[:, :Q_LORA], qk[:, Q_LORA:]
    q_r = F.rms_norm(q_r, (Q_LORA,), qa_norm, eps)
    latent = F.rms_norm(c, (KV_LORA,), kva_norm, eps)  # [T, KV_LORA] = the cache entry

    # The full (non-absorbed) path: q = q_r W_qb -> [T, H, DQK]; k = W_uk^T c per head... but MLA's
    # k for head h is c @ W_uk[h] (KV_LORA -> DQK), and v is c @ W_uv[h] (KV_LORA -> DV).
    q = (q_r @ w_qb).view(T, H, DQK)
    # MLA's per-head k is c @ W_uk[h] (KV_LORA -> DQK) and v is c @ W_uv[h] (KV_LORA -> DV).
    k = torch.einsum("tk,hdk->thd", latent, w_uk)
    v = torch.einsum("tk,hkv->thv", latent, w_uv)

    if cache is not None:
        k = torch.cat([cache[0], k], 0)
        v = torch.cat([cache[1], v], 0)

    scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(DQK)  # [H, T, S]
    S = scores.shape[-1]
    # causal mask: query t may attend keys <= (S - T + t)
    offset = S - T
    causal = torch.ones(T, S, dtype=torch.bool).tril(diagonal=offset)
    scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores.float(), dim=-1)
    o = torch.einsum("hts,shv->thv", attn, v)  # [T, H, DV]
    out = o.reshape(T, H * DV) @ w_o
    return out, latent


def test_mla_allocate_cache_shape(device):
    """The latent cache is [blocks, 1, block_size, kv_lora] bf16: one 512-wide entry per token."""
    mla, _ = build_mla(device)
    cache = mla.allocate_cache(4)
    assert tuple(cache.shape) == (4, 1, BLOCK, KV_LORA)
    assert cache.dtype == ttnn.bfloat16


def test_mla_project_shapes(device):
    """``_project`` gives the absorbed query [1,H,T,kv_lora] and the latent [1,1,T,kv_lora]."""
    mla, _ = build_mla(device)
    T = 32
    x = torch.randn(1, 1, T, HIDDEN) * 0.5
    q_lat, latent = mla._project(to_dev(x, device))
    ttnn.synchronize_device(device)
    assert tuple(q_lat.shape) == (1, H, T, KV_LORA), q_lat.shape
    assert tuple(latent.shape) == (1, 1, T, KV_LORA), latent.shape


def test_mla_absorbed_query_matches_full_product(device):
    """The absorber identity: ``W_uk^T q`` dotted with the latent equals ``q`` dotted with ``W_uk c``.

    This is the assumption that makes the absorbed decode path legal, and it holds only because
    ``qk_rope_head_dim == 0`` (no rope tail to absorb).
    """
    mla, ref = build_mla(device, seed=1)
    T = 16
    x = torch.randn(T, HIDDEN) * 0.5

    q_lat, latent = mla._project(to_dev(x.reshape(1, 1, T, HIDDEN), device))
    ttnn.synchronize_device(device)

    q_lat_t = dev_to_tor(q_lat).reshape(H, T, KV_LORA)
    latent_t = dev_to_tor(latent).reshape(T, KV_LORA)

    # Absorbed score for head h, query t, key s:  q_lat[h,t] . latent[s]
    absorbed = torch.einsum("htk,sk->hts", q_lat_t, latent_t)

    # Full form: q[h,t] . (W_uk[h] c[s])
    qk = x @ ref["w_qa_kva"]
    q_r = F.rms_norm(qk[:, :Q_LORA], (Q_LORA,), ref["qa_norm"], 1e-5)
    q = (q_r @ ref["w_qb"]).view(T, H, DQK).permute(1, 0, 2)  # [H,T,DQK]
    kfull = torch.einsum("sk,hdk->hsd", latent_t, ref["w_uk"])  # [H,S,DQK]
    full = torch.einsum("htd,hsd->hts", q, kfull)

    assert pcc(absorbed, full) > 0.999, f"absorbed vs full PCC {pcc(absorbed, full):.6f}"


@pytest.mark.parametrize("T", [64])
def test_mla_prefill_matches_reference(device, T):
    """One prefill chunk end to end against the dense causal latent-attention reference.

    ``T`` must be a multiple of the page size (``BLOCK``), as the paged fill requires.
    """
    mla, ref = build_mla(device, seed=2)
    n_pages = T // BLOCK
    # paged_fill_cache/chunked SDPA require the page table's row to be a multiple of 32 columns.
    table_cols = max(32, ((n_pages + 31) // 32) * 32)
    cache = mla.allocate_cache(table_cols)
    x = torch.randn(T, HIDDEN) * 0.5

    chunk_table = ttnn.from_torch(
        torch.arange(table_cols, dtype=torch.int32).reshape(1, table_cols),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    user_table = ttnn.from_torch(
        torch.arange(table_cols, dtype=torch.int32).reshape(1, table_cols),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    chunk_start = ttnn.from_torch(
        torch.tensor([0], dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    out = mla.prefill(to_dev(x.reshape(1, 1, T, HIDDEN), device), cache, chunk_table, user_table, chunk_start)
    ttnn.synchronize_device(device)

    ref_out, _ = mla_reference(x, **ref, eps=1e-5)
    got = dev_to_tor(out).reshape(T, HIDDEN)
    assert pcc(got, ref_out) > 0.98, f"MLA prefill PCC {pcc(got, ref_out):.5f}"


def test_mla_decode_matches_reference(device):
    """Decode attends the cached latents plus the new token; the cache must grow by one per step.

    Uses a 32-head instance because ``MLA.decode`` slices heads in groups of ``decode_heads = 32``
    (64 real heads need two groups to fit one core's L1; the grouping is part of the contract).
    """
    mla, ref = build_mla(device, seed=3, heads=DECODE_H)
    n_blocks = 32  # page-table rows must be a multiple of 32 columns
    cache = mla.allocate_cache(n_blocks)
    page_table = torch.arange(n_blocks, dtype=torch.int32).reshape(1, n_blocks)

    T = 3
    x = torch.randn(T, HIDDEN) * 0.5
    outs = []
    for t in range(T):
        cur_pos = ttnn.from_torch(
            torch.tensor([t], dtype=torch.int32), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        pt = ttnn.from_torch(page_table, device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        o = mla.decode(to_dev(x[t : t + 1].reshape(1, 1, 1, HIDDEN), device), cache, pt, cur_pos)
        ttnn.synchronize_device(device)
        outs.append(dev_to_tor(o).reshape(HIDDEN))

    # Reference: recompute with the explicit prefix cache, one token at a time.
    ref_outs = []
    cached_k = cached_v = None
    for t in range(T):
        qk = x[t : t + 1] @ ref["w_qa_kva"]
        q_r = F.rms_norm(qk[:, :Q_LORA], (Q_LORA,), ref["qa_norm"], 1e-5)
        q = (q_r @ ref["w_qb"]).view(DECODE_H, DQK)  # [H, DQK]
        lat = F.rms_norm(qk[:, Q_LORA:], (KV_LORA,), ref["kva_norm"], 1e-5)
        kt = torch.einsum("sk,hdk->hsd", lat, ref["w_uk"])  # [H, 1, DQK]
        vt = torch.einsum("sk,hkv->hsv", lat, ref["w_uv"])  # [H, 1, DV]
        cached_k = kt if cached_k is None else torch.cat([cached_k, kt], 1)
        cached_v = vt if cached_v is None else torch.cat([cached_v, vt], 1)
        scores = torch.einsum("hd,hsd->hs", q, cached_k) / math.sqrt(DQK)  # [H, S]
        attn = torch.softmax(scores.float(), -1)
        o = torch.einsum("hs,hsv->hv", attn, cached_v).reshape(1, DECODE_H * DV) @ ref["w_o"]
        ref_outs.append(o.reshape(HIDDEN))

    got = torch.stack(outs, 0)
    ref_all = torch.stack(ref_outs, 0)
    assert pcc(got, ref_all) > 0.98, f"MLA decode PCC {pcc(got, ref_all):.5f}"
