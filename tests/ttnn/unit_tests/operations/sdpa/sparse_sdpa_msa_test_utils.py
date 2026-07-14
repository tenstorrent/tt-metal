# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for sparse_sdpa_msa tests.

The golden mirrors MiniMax-AI/MSA `sparse_attention_ref` (python/fmha_sm100/cute/test_sparse_atten.py):
pure ``softmax(q·scale·kᵀ over selected blocks)·v`` with separate K and V. Causality is encoded by `indices`;
the op has no token-level causal mask. RoPE and QK-norm are applied upstream.

Sentinel for a masked block id is -1 (0xFFFFFFFF as uint32 bits), used as a contiguous tail.
"""

import torch

import ttnn

SENTINEL = -1  # masked/invalid block id; contiguous tail per (group, query) row
BLK_KV = 128  # MSA block size in tokens (= 4 tile-rows)


def sparse_attention_ref_msa(q, k, v, indices, scale, *, blk_kv=BLK_KV, causal=False, chunk_start_idx=0):
    """MSA block-sparse reference: attend the selected blocks, softmax, then PV with separate V.

        q       [B, H, S, d]            (post-rope, post-qk-norm — done upstream)
        k, v    [B, n_kv, T, d]         (separate tensors; T % blk_kv == 0)
        indices [B, n_kv, S, topk]      block-ids per (group, query); SENTINEL (-1) = masked, contiguous tail
        -> out  [B, H, S, v_dim]        (v_dim = v.shape[-1])

    `causal=True` enables a token-level causality — required for correctness on the diagonal block,
    whose selected tokens after the query position are future and must not be attended.

    Query heads sharing a KV head also share that KV head's block selection. All-masked rows return 0.
    """
    B, H, S, d = q.shape
    n_kv, T = k.shape[1], k.shape[2]
    topk = indices.shape[-1]
    G = H // n_kv
    nblk = T // blk_kv
    assert T % blk_kv == 0 and H % n_kv == 0

    qf, kf, vf = q.float(), k.float(), v.float()
    v_dim = vf.shape[-1]

    # block_mask[b, g, s, blk] — set True only for valid (non-sentinel) selected blocks (flat index_put so a
    # sentinel clamped to block 0 can never overwrite a genuinely-selected block 0).
    block_mask = torch.zeros(B, n_kv, S, nblk, dtype=torch.bool)
    valid = indices >= 0
    idx_safe = torch.where(valid, indices, torch.zeros_like(indices)).long()
    flat_mask = block_mask.view(-1, nblk)
    flat_valid = valid.reshape(-1, topk)
    flat_idx = idx_safe.reshape(-1, topk)
    row = torch.arange(flat_mask.shape[0]).unsqueeze(1)
    flat_mask[row.expand_as(flat_idx)[flat_valid], flat_idx[flat_valid]] = True

    token_mask = block_mask.repeat_interleave(blk_kv, dim=3)  # [B,n_kv,S,T]
    token_mask = token_mask.repeat_interleave(G, dim=1)  # [B,H,S,T]
    kf = kf.repeat_interleave(G, dim=1)  # [B,H,T,d]
    vf = vf.repeat_interleave(G, dim=1)  # [B,H,T,d]

    scores = torch.einsum("bhsd,bhtd->bhst", qf * scale, kf)  # [B,H,S,T]
    scores = scores.masked_fill(~token_mask, float("-inf"))
    if causal:
        # Strictly-future keys (only ever inside the diagonal block; past blocks are all <= query pos).
        q_pos = (torch.arange(S) + chunk_start_idx).view(1, 1, S, 1)
        kv_pos = torch.arange(T).view(1, 1, 1, T)
        scores = scores.masked_fill(kv_pos > q_pos, float("-inf"))

    row_has_value = (scores > float("-inf")).any(dim=-1, keepdim=True)
    scores = torch.where(row_has_value, scores, torch.zeros_like(scores))
    attn = torch.where(row_has_value, scores.softmax(dim=-1, dtype=torch.float32), torch.zeros_like(scores))
    return torch.einsum("bhst,bhtd->bhsd", attn, vf[..., :v_dim])  # [B,H,S,v_dim]


def sparse_attention_ref_msa_sampled_tokens(
    q, k, v, indices, scale, sample_tokens, *, blk_kv=BLK_KV, causal=False, chunk_start_idx=0
):
    """Memory-light MSA reference for selected query tokens. `causal=True` adds the token-level causal
    mask (a selected key may be a future token only inside the diagonal block; mask those out)."""
    B, H, _, d = q.shape
    n_kv, v_dim = k.shape[1], v.shape[-1]
    assert B == 1 and H % n_kv == 0
    G = H // n_kv

    qf, kf, vf = q.float(), k.float(), v.float()
    outs = []
    for s in sample_tokens:
        out = torch.zeros(H, v_dim, dtype=torch.float32)
        p = s + chunk_start_idx
        for g in range(n_kv):
            row = indices[0, g, s]
            valid_blocks = row[row >= 0].long()
            if valid_blocks.numel() == 0:
                continue
            h0, h1 = g * G, (g + 1) * G
            k_sel = torch.cat([kf[0, g, int(blk) * blk_kv : (int(blk) + 1) * blk_kv] for blk in valid_blocks], dim=0)
            v_sel = torch.cat([vf[0, g, int(blk) * blk_kv : (int(blk) + 1) * blk_kv] for blk in valid_blocks], dim=0)
            scores = torch.matmul(qf[0, h0:h1, s] * scale, k_sel.transpose(0, 1))
            if causal:
                key_ids = torch.cat([torch.arange(int(blk) * blk_kv, (int(blk) + 1) * blk_kv) for blk in valid_blocks])
                scores = scores.masked_fill(key_ids.view(1, -1) > p, float("-inf"))
            attn = scores.softmax(dim=-1, dtype=torch.float32)
            out[h0:h1] = torch.matmul(attn, v_sel)
        outs.append(out)
    return torch.stack(outs, dim=1).unsqueeze(0)  # [1,H,len(sample_tokens),v_dim]


def dense_grouped_kv_attention(q, k, v, scale, *, causal=False, chunk_start_idx=0):
    """Dense grouped-KV reference used when all blocks are selected."""
    B, H, S, d = q.shape
    n_kv, T = k.shape[1], k.shape[2]
    G = H // n_kv
    qf = q.float()
    kf = k.float().repeat_interleave(G, dim=1)
    vf = v.float().repeat_interleave(G, dim=1)
    scores = torch.einsum("bhsd,bhtd->bhst", qf * scale, kf)
    if causal:
        q_pos = torch.arange(S) + chunk_start_idx
        kv_pos = torch.arange(T)
        scores = scores.masked_fill(kv_pos.view(1, 1, 1, T) > q_pos.view(1, 1, S, 1), float("-inf"))
    attn = scores.softmax(dim=-1, dtype=torch.float32)
    return torch.einsum("bhst,bhtd->bhsd", attn, vf)


def make_msa_inputs(H, n_kv, S, T, topk, d, *, blk_kv=BLK_KV, causal=False, force_local=True, seed=0):
    """Build q, k, v, and sorted block-id indices with a sentinel tail."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, S, d, generator=gen)
    k = torch.randn(1, n_kv, T, d, generator=gen)
    v = torch.randn(1, n_kv, T, d, generator=gen)
    nblk = T // blk_kv
    assert topk <= nblk
    indices = torch.full((1, n_kv, S, topk), SENTINEL, dtype=torch.int32)
    for g in range(n_kv):
        for s in range(S):
            if causal:
                local = s // blk_kv  # block containing the query (chunk-local)
                visible = min(local + 1, nblk)
                if visible <= topk:
                    chosen = torch.arange(visible)
                else:
                    pool = torch.randperm(visible, generator=gen)[:topk]
                    if force_local and local not in pool.tolist():
                        pool[-1] = local
                    chosen = pool.sort().values
            else:
                chosen = torch.randperm(nblk, generator=gen)[:topk].sort().values
            indices[0, g, s, : chosen.numel()] = chosen.to(torch.int32)
    return q, k, v, indices


def pcc(out, golden_t):
    return torch.corrcoef(torch.stack([out.flatten().float(), golden_t.flatten().float()]))[0, 1].item()


# Composition baseline: map separate K/V block selection onto DSA sparse_sdpa for n_kv=1.
def run_op_msa_composed(q, k, v, indices, device, *, k_chunk_size=128):
    assert k.shape[1] == 1 and v.shape[1] == 1, "composition baseline is n_kv==1 only"
    _, H, S, d = q.shape
    T = k.shape[2]
    v_dim = v.shape[-1]
    topk = indices.shape[-1]
    B = BLK_KV
    kv_packed = torch.cat([v[0, 0], k[0, 0]], dim=-1).unsqueeze(0).unsqueeze(0)  # [1,1,T,v_dim+d], V first
    qz = torch.cat([torch.zeros(1, H, S, v_dim), q[0:1]], dim=-1)  # [1,H,S,v_dim+d]
    Hp = ((H + 31) // 32) * 32
    if Hp != H:
        qz = torch.cat([qz, torch.zeros(1, Hp - H, S, v_dim + d)], dim=1)  # dummy heads
    exp = torch.full((1, 1, S, topk * B), 0xFFFFFFFF, dtype=torch.int64)  # token-ids; sentinel = 0xFFFFFFFF
    for s in range(S):
        row = []
        for j in range(topk):
            blk = int(indices[0, 0, s, j])
            if blk < 0:
                continue
            for r in range(B):
                row.append(blk * B + r)  # all B tokens of the selected block
        if row:
            exp[0, 0, s, : len(row)] = torch.tensor(row, dtype=torch.int64)
    scale = d**-0.5

    def dev(t, dt):
        return ttnn.from_torch(
            t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    tt_out = ttnn.transformer.sparse_sdpa(
        dev(qz.to(torch.bfloat16), ttnn.bfloat16),
        dev(kv_packed.to(torch.bfloat16), ttnn.bfloat16),
        dev(exp.to(torch.int32), ttnn.uint32),
        v_dim,
        scale=scale,
        k_chunk_size=k_chunk_size,
    )
    return ttnn.to_torch(tt_out)[:, :H]  # drop dummy heads


def run_op_msa_native(
    q,
    k,
    v,
    indices,
    device,
    *,
    block_size=BLK_KV,
    kv_dtype=ttnn.bfloat16,
    q_dtype=ttnn.bfloat16,
    chunk_start_idx=None,
    cluster_axis=None,
):
    """Run the native op. K/V are pre-tiled; q/indices/output stay row-major."""
    _, H, _, d = q.shape
    scale = d**-0.5

    def dev_rm(t, dt):  # row-major: q, indices, output (ttnn quantizes float->fp8 when dt is fp8_e4m3)
        return ttnn.from_torch(
            t, dtype=dt, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def dev_tile(t, dt):  # pre-tiled K/V cache (block-aligned; bf16 or bfp8_b)
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=dt,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    tt_out = ttnn.transformer.sparse_sdpa_msa(
        dev_rm(q.to(torch.float32), q_dtype),  # float32 host (ttnn requires fp32 source when target is fp8)
        dev_tile(k, kv_dtype),
        dev_tile(v, kv_dtype),
        dev_rm(indices.to(torch.int32), ttnn.uint32),  # -1 sentinel -> 0xFFFFFFFF bit pattern
        scale=scale,
        block_size=block_size,
        chunk_start_idx=chunk_start_idx,  # set -> enable token-level diagonal-block causal mask
        cluster_axis=cluster_axis,
    )
    # Output dtype matches q. fp8 can't be to_torch'd directly, so typecast fp8 -> bf16 on device first.
    if tt_out.dtype == ttnn.fp8_e4m3:
        tt_out = ttnn.typecast(tt_out, ttnn.bfloat16)
    return ttnn.to_torch(tt_out)[:, :H]
