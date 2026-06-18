# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Reference implementation (torch golden) of the MiniMax-M3 **GQA** sparse-attention
prefill op — the correctness oracle for the GQA variant of ``ttnn.transformer.sparse_sdpa``.

This mirrors ``models/demos/deepseek_v32/reference_cpu/sparse_sdpa_prefill.py`` (the
MLA golden) in style and contract, but for M3's attention shape. It is the "give me
the shapes + a torch test case" deliverable: a drop-in golden the kernel can be
validated against, plus an input builder matching the producer contract.

WHAT CHANGES VS THE MLA OP (``sparse_sdpa`` on the sparse-MLA-prefill branch)
----------------------------------------------------------------------------------
The MLA op is absorbed-MQA: ONE latent KV head, and V is a *prefix slice of K*
(``kv [1,1,T,576]``; K = full 576, V = ``kv[..., :512]``). M3 is plain **GQA**
(VERIFIED below) — no latent compression:
  * 64 query heads, **4 KV heads** (GQA group = 64/4 = 16), head_dim **128**.
  * K and V are **disjoint, equal-width (128)** tensors — V is NOT a slice of K.
  * **block selection is PER GQA GROUP** (4 independent top-k lists), NOT one shared
    list (see VERIFIED below — this differs from DeepSeek DSA's summed single list).

ARCHITECTURE VERIFIED (don't assume — checked against authoritative sources)
----------------------------------------------------------------------------------
  * HF config ``MiniMaxAI/MiniMax-M3`` (text_config): num_attention_heads=64,
    num_key_value_heads=4, head_dim=128; **no MLA keys** (no kv_lora_rank/q_lora_rank/
    qk_nope_head_dim/qk_rope_head_dim/v_head_dim) ⇒ plain GQA, K and V disjoint 128-wide.
  * MSA paper (arXiv:2606.13392) "MiniMax Sparse Attention": "Base Attention uses GQA,
    not MLA"; "a lightweight Index Branch ... independently selects a Top-k subset **for
    each GQA group**"; "one lightweight query head per GQA group, one shared key head".
  * MSA reference (github.com/MiniMax-AI/MSA, ``python/fmha_sm100/api.py``):
    ``sparse_topk_select`` output ``(total_qo_len, num_qo_heads, topk)``; ``max_score``
    ``(num_qo_heads, max_k_tiles, total_qo_len)``; ``kv_block_indexes``
    ``[total_qo_len, num_kv_heads | num_qo_heads, kv_block_num]`` — i.e. the selected-block
    index tensor carries a per-(kv-)head axis. Confirms per-group selection, not shared.
  * config: sparse_num_index_heads=4 == num_key_value_heads=4 ⇒ one index head / GQA group.

SHARED CONTRACT (kept identical to the MLA op, so the kernel skeleton carries over)
----------------------------------------------------------------------------------
  * Masking is **fully baked into the indices** (FlashMLA sparse contract): a slot
    with ``index == 0xFFFFFFFF`` (``MASKED_INDEX``) scores ``-inf`` (0 weight). There
    is **no causal/position math in the op** — the indexer→topk producer emits the
    sentinel for future/pad slots. Sentinels are a contiguous tail; every row has
    >= 1 valid key; all non-sentinel indices are < T.
  * Output dtype matches q; ``scale`` defaults to ``head_dim**-0.5`` (= 128**-0.5).

REAL M3 SHAPES (per chip, under the §6.4 TP->SP CCL bracket: sparse path runs SP, so
each chip holds ALL heads and an S-shard — H is full 64, a multiple of 32 ✓)
----------------------------------------------------------------------------------
  q       [1, 64, S, 128]    bf16   (Hq = 64 query heads)
  k       [1,  4, T, 128]    bf16   (Hkv = 4 KV heads)
  v       [1,  4, T, 128]    bf16   (Hkv = 4 KV heads; disjoint from k)
  indices [1,  4, S, TOPK]   uint32 (ONE LIST PER GQA GROUP; 0xFFFFFFFF = masked)
  ->  out [1, 64, S, 128]    bf16
  TOPK = sparse_topk_blocks(16) * sparse_block_size(128) = 2048 ; scale = 128**-0.5
  (block-0 sink + local block are folded into each group's index list by the producer.)

  NOTE on the index head-axis: this golden accepts indices with head-dim
  ``{1, Hkv, Hq}`` and broadcasts to per-q-head: 1 = one shared list (the OLD MLA-style
  assumption — kept for compatibility/ablation), **Hkv = per-GQA-group (the M3 truth)**,
  Hq = per-query-head (the reference's widest form). M3 default = Hkv.

OPEN QUESTIONS FOR THE KERNEL OWNER (the design points this golden makes concrete)
----------------------------------------------------------------------------------
  1. K/V packing: pass **two tensors** ``k``,``v`` ([1,4,T,128] each), or **one packed**
     ``kv [1,4,T,256]`` = concat(K,V)? The MLA op's "V = leading v_dim of one cache"
     doesn't fit GQA (K!=V). This golden takes two tensors (k, v) — the unambiguous form.
  2. Index head-axis the op will consume: per-GQA-group ``[1,4,S,2048]`` (matches the
     MSA reference's ``kv_block_indexes [.., num_kv_heads, ..]`` and is the compact M3
     form) vs per-q-head ``[1,64,S,2048]`` (``sparse_topk_select`` output form). The MLA
     op took ``[1,1,S,k]``; GQA needs a head axis. Default proposed here: per-group (4).
  3. KV-head dim placement: ``[1, Hkv, T, 128]`` (head as dim1, like q) vs flattened.
     Golden uses dim1 = Hkv.

Two equivalent forms (mirrors the MLA golden):
  * ``sparse_gqa``        — the op (gather the k selected slots). Validate the kernel
                            against this.
  * ``sparse_gqa_masked`` — dense-mask golden (equivalence oracle only).
"""

import torch

MASKED_INDEX = 0xFFFFFFFF  # sentinel: slot is masked out (scores -inf)

# Real M3 defaults (text_config.sparse_attention_config + verified projection shapes).
HEAD_DIM = 128  # head_dim — K and V are both this wide (no latent split)
NUM_Q_HEADS = 64  # num_attention_heads
NUM_KV_HEADS = 4  # num_key_value_heads  -> GQA group = 64/4 = 16, and #GQA groups = 4
SPARSE_BLOCK_SIZE = 128  # sparse_block_size
SPARSE_TOPK_BLOCKS = 16  # sparse_topk_blocks
TOPK = SPARSE_TOPK_BLOCKS * SPARSE_BLOCK_SIZE  # 2048 selected keys per query, per group


def _idx_to_per_qhead(indices, B, Hq, Hkv, S):
    """Normalize the indices head-axis to per-query-head [B, Hq, S, topk].

    Accepts head-dim 1 (shared), Hkv (per-GQA-group, the M3 default), or Hq (per-q-head).
    Reshapes [B,Hidx,S,topk] / [B,1,S,k]-style inputs and broadcasts to Hq.
    """
    topk = indices.shape[-1]
    idx = indices.reshape(B, -1, S, topk)
    Hidx = idx.shape[1]
    group = Hq // Hkv
    if Hidx == 1:  # one shared list (old MLA-style assumption / ablation)
        return idx.expand(B, Hq, S, topk)
    if Hidx == Hkv:  # per-GQA-group (M3): each group's 16 q-heads share its list
        return idx.repeat_interleave(group, dim=1)
    if Hidx == Hq:  # per-query-head (reference's widest form)
        return idx
    raise ValueError(f"indices head-dim {Hidx} must be 1, Hkv={Hkv}, or Hq={Hq}")


def _prep(q, k, v):
    B, Hq, S, D = q.shape
    Hkv, T = k.shape[1], k.shape[2]
    assert v.shape == k.shape, f"k {k.shape} and v {v.shape} must match (disjoint GQA K/V, equal width)"
    assert Hq % Hkv == 0, f"Hq ({Hq}) must be a multiple of Hkv ({Hkv})"
    return B, Hq, Hkv, S, T, D, Hq // Hkv


def sparse_gqa(
    q: torch.Tensor,  # [1, Hq, S, 128] query heads
    k: torch.Tensor,  # [1, Hkv, T, 128] KV-head keys
    v: torch.Tensor,  # [1, Hkv, T, 128] KV-head values (disjoint from k)
    indices: torch.Tensor,  # [1, {1|Hkv|Hq}, S, topk] uint32; 0xFFFFFFFF = masked. M3: Hkv (per group)
    scale: float = HEAD_DIM**-0.5,
    start_pos: int = 0,  # accepted for signature parity; IGNORED (indices encode causality)
) -> torch.Tensor:  # [1, Hq, S, 128]
    """
    GQA sparse attention over the per-GQA-group top-k selected keys. Each query head h
    belongs to group ``h // group`` (== its KV head); it scores against that group's
    gathered keys, ``-inf`` on sentinel slots, softmax over the topk axis, weighted-sum
    the gathered V. Masking is baked into ``indices``; no causal/position math here.
    """
    del start_pos  # causality is encoded in `indices` (sentinel), not positions
    B, Hq, Hkv, S, T, D, group = _prep(q, k, v)
    topk = indices.shape[-1]
    idx_qh = _idx_to_per_qhead(indices, B, Hq, Hkv, S)  # [B,Hq,S,topk]

    masked = idx_qh == MASKED_INDEX  # [B,Hq,S,topk]
    idx_safe = torch.where(masked, torch.zeros_like(idx_qh), idx_qh).to(torch.int64)

    # Each q-head reads its GQA group's KV head: expand Hkv -> Hq.
    k_q = k.repeat_interleave(group, dim=1)  # [B,Hq,T,D]
    v_q = v.repeat_interleave(group, dim=1)

    # Gather the selected keys/values per q-head: [B,Hq,T,D] -> [B,Hq,S,topk,D].
    idx_g = idx_safe.unsqueeze(-1).expand(B, Hq, S, topk, D)
    sel_k = torch.gather(k_q.unsqueeze(2).expand(B, Hq, S, T, D), 3, idx_g)
    sel_v = torch.gather(v_q.unsqueeze(2).expand(B, Hq, S, T, D), 3, idx_g)

    scores = torch.einsum("bhsd,bhsjd->bhsj", q, sel_k) * scale  # [B,Hq,S,topk]
    scores = scores.masked_fill(masked, float("-inf"))
    probs = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum("bhsj,bhsjd->bhsd", probs, sel_v)  # [B,Hq,S,D]
    return out


def sparse_gqa_masked(
    q: torch.Tensor,  # [1, Hq, S, 128]
    k: torch.Tensor,  # [1, Hkv, T, 128]
    v: torch.Tensor,  # [1, Hkv, T, 128]
    indices: torch.Tensor,  # [1, {1|Hkv|Hq}, S, topk]; 0xFFFFFFFF = masked
    scale: float = HEAD_DIM**-0.5,
    start_pos: int = 0,
) -> torch.Tensor:  # [1, Hq, S, 128]
    """Dense-mask golden (equivalence oracle). Per q-head: softmax over the full T axis
    with an additive ``-inf`` mask everywhere except that head's selected (non-sentinel) keys."""
    del start_pos
    B, Hq, Hkv, S, T, D, group = _prep(q, k, v)
    topk = indices.shape[-1]
    idx_qh = _idx_to_per_qhead(indices, B, Hq, Hkv, S)  # [B,Hq,S,topk]

    masked = idx_qh == MASKED_INDEX
    idx_safe = torch.where(masked, torch.full_like(idx_qh, T), idx_qh).to(torch.int64)  # sentinels -> col T

    k_q = k.repeat_interleave(group, dim=1)  # [B,Hq,T,D]
    v_q = v.repeat_interleave(group, dim=1)
    scores = torch.einsum("bhsd,bhtd->bhst", q, k_q) * scale  # [B,Hq,S,T]

    # Per-(b,h,s) selected-key mask (indices differ per group, so the mask is per-head).
    index_mask = torch.full((B, Hq, S, T + 1), float("-inf"), device=q.device)
    index_mask.scatter_(-1, idx_safe, 0.0)
    scores = scores + index_mask[..., :T]  # drop throwaway col
    probs = scores.softmax(dim=-1, dtype=torch.float32).to(q.dtype)
    out = torch.einsum("bhst,bhtd->bhsd", probs, v_q)  # [B,Hq,S,D]
    return out


def make_gqa_inputs(
    S,
    T,
    topk=TOPK,
    Hq=NUM_Q_HEADS,
    Hkv=NUM_KV_HEADS,
    D=HEAD_DIM,
    index_heads=None,  # None -> Hkv (per-GQA-group, the M3 default); or 1 / Hq
    n_valid_fn=None,
    seed=0,
    dtype=torch.float32,
):
    """Build (q, k, v, indices) matching the producer contract — per-group sentinels are a
    contiguous tail, every row has >= 1 valid key, all valid indices < T.

    ``index_heads`` selects the indices head-axis: Hkv = one independent top-k list per GQA
    group (default, the M3 truth), 1 = a single shared list, Hq = per-query-head.
    ``n_valid_fn(s)`` -> #valid keys for query row s (default: all topk, clamped to T).
    Returns indices as int64 (cast to uint32 at the device boundary, as the op test does).
    """
    gen = torch.Generator().manual_seed(seed)
    Hidx = Hkv if index_heads is None else index_heads
    q = torch.randn(1, Hq, S, D, generator=gen, dtype=dtype)
    k = torch.randn(1, Hkv, T, D, generator=gen, dtype=dtype)
    v = torch.randn(1, Hkv, T, D, generator=gen, dtype=dtype)
    indices = torch.full((1, Hidx, S, topk), MASKED_INDEX, dtype=torch.int64)
    if n_valid_fn is None:
        n_valid_fn = lambda s: topk  # noqa: E731
    for h in range(Hidx):  # each index head/group selects independently
        for s in range(S):
            nv = max(1, min(topk, min(T, n_valid_fn(s))))
            indices[0, h, s, :nv] = torch.randperm(T, generator=gen)[:nv]
    return q, k, v, indices


def _selfcheck() -> None:
    """Prove sparse_gqa == dense-mask golden on real-M3-shaped, sentinel-padded inputs,
    across all three index head-axis conventions (per-group is the M3 default).

    Uses SMALL S/T/topk (the gather materializes sel[B,Hq,S,topk,D]) but the REAL head
    geometry: Hq=64, Hkv=4, group=16, head_dim=128.
    """
    torch.manual_seed(0)
    S, T, topk = 16, 256, 64  # topk < T => non-trivial sparsity
    scale = HEAD_DIM**-0.5

    for ih_id, ih in [("per_group(Hkv=4)", NUM_KV_HEADS), ("shared(1)", 1), ("per_qhead(Hq=64)", NUM_Q_HEADS)]:
        for nv_id, nv_fn in [
            ("all_valid", lambda s: topk),
            ("few_valid", lambda s: 1 + (s % 7)),
            ("boundary", lambda s: 1 + (s * 3) % 20),
        ]:
            q, k, v, indices = make_gqa_inputs(S, T, topk, index_heads=ih, n_valid_fn=nv_fn)
            a = sparse_gqa(q, k, v, indices, scale)
            b = sparse_gqa_masked(q, k, v, indices, scale)
            assert a.shape == (1, NUM_Q_HEADS, S, HEAD_DIM), a.shape
            torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
        print(f"OK [{ih_id:16s}]: sparse_gqa == dense-mask (all_valid/few_valid/boundary)")

    print(
        f"\nGQA sparse_sdpa golden OK — Hq={NUM_Q_HEADS} Hkv={NUM_KV_HEADS} group={NUM_Q_HEADS//NUM_KV_HEADS} "
        f"head_dim={HEAD_DIM}, TOPK(prod)={TOPK}, scale={scale:.6f}; M3 default index head-axis = Hkv (per group)"
    )


if __name__ == "__main__":
    _selfcheck()
