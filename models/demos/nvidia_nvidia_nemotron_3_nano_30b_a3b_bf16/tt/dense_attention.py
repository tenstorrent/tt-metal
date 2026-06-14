# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""DenseAttention block — TP=4 column/row-parallel on QB 4-chip Blackhole.

GQA (32 Q / 2 KV heads, head_dim=128).  No RoPE (HF has it
commented out with a TODO for this model variant).

TP strategy:
  wq [4096, 2688]: column-parallel → [1024, 2688]/device → 8 Q heads/device
  wk [256,  2688]: replicated (2 KV heads < TP=4)
  wv [256,  2688]: replicated
  SDPA: ttnn.transformer.scaled_dot_product_attention (prefill path, S≥1)
        q [B,8,S,128], k [B,2,S,128], v [B,2,S,128] per device — GQA handled.
        For production decode (with KV cache) use scaled_dot_product_attention_decode.
  wo [2688, 4096]: row-parallel → [2688, 1024]/device
  all_reduce after wo; residual add on device.
"""

import torch

import ttnn
from ttnn import MeshDevice

from .tp import _col, _rep, _rep_keyed, _row, all_reduce

NUM_HEADS = 32
NUM_KV_HEADS = 2
HEAD_DIM = 128
TP = 4
HIDDEN_SIZE = 2688
NORM_EPS = 1e-5


def dense_attention_forward(
    mesh_device: MeshDevice,
    hidden_states: ttnn.Tensor,  # [B, S, 2688] bf16 on device
    norm_weight: torch.Tensor,  # [2688] bf16 CPU
    wq: torch.Tensor,  # [4096, 2688] bf16 CPU
    wk: torch.Tensor,  # [256,  2688] bf16 CPU
    wv: torch.Tensor,  # [256,  2688] bf16 CPU
    wo: torch.Tensor,  # [2688, 4096] bf16 CPU
    norm_eps: float = NORM_EPS,
    kv_cache: tuple | None = None,  # (k_cache, v_cache) [num_blocks, n_kv, block_size, head_dim]
    page_table: ttnn.Tensor | None = None,  # [B, max_blocks_per_seq]
    current_pos: ttnn.Tensor | None = None,  # [B] device tensor (current seq position)
) -> ttnn.Tensor:
    """Returns [B, S, 2688] bfloat16 on device (replicated) with residual applied.

    When kv_cache/page_table/current_pos are all provided, uses paged Flash-Decode.
    Otherwise falls back to prefill SDPA (S can be >1).
    """
    residual = hidden_states
    B = hidden_states.shape[0]
    S = hidden_states.shape[1]
    paged = kv_cache is not None and page_table is not None and current_pos is not None

    # 1. Pre-norm
    w_tt = _rep_keyed(id(norm_weight), norm_weight.bfloat16().unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(hidden_states, epsilon=norm_eps, weight=w_tt)

    # 2. Q: column-parallel → [B, S, 1024]/device (8 heads/device)
    wq_tt = _col(wq, mesh_device)
    q_tt = ttnn.linear(normed_tt, wq_tt, transpose_b=True)  # [B, S, 1024]/device

    # 3. K: replicated (2 KV heads < TP=4)
    wk_tt = _rep(wk, mesh_device)
    k_tt = ttnn.linear(normed_tt, wk_tt, transpose_b=True)  # [B, S, 256]/device

    # 4. V: replicated
    wv_tt = _rep(wv, mesh_device)
    v_tt = ttnn.linear(normed_tt, wv_tt, transpose_b=True)  # [B, S, 256]/device

    # 5. Reshape to [B, heads, S, head_dim] for SDPA
    q_4d = ttnn.reshape(q_tt, [B, S, NUM_HEADS // TP, HEAD_DIM])  # [B, S, 8, 128]
    q_4d = ttnn.permute(q_4d, [0, 2, 1, 3])  # [B, 8, S, 128]

    k_4d = ttnn.reshape(k_tt, [B, S, NUM_KV_HEADS, HEAD_DIM])  # [B, S, 2, 128]
    k_4d = ttnn.permute(k_4d, [0, 2, 1, 3])  # [B, 2, S, 128]

    v_4d = ttnn.reshape(v_tt, [B, S, NUM_KV_HEADS, HEAD_DIM])  # [B, S, 2, 128]
    v_4d = ttnn.permute(v_4d, [0, 2, 1, 3])  # [B, 2, S, 128]

    # 6. Attention — paged Flash-Decode or prefill SDPA.
    if paged:
        k_cache, v_cache = kv_cache
        # Reformat K/V: [B, n_kv, S, D] → [S, B, n_kv, D] for paged_update_cache.
        k_upd = ttnn.permute(k_4d, [2, 0, 1, 3])  # [1, B, 2, 128]
        v_upd = ttnn.permute(v_4d, [2, 0, 1, 3])
        # paged_update_cache requires L1 HEIGHT_SHARDED with num_cores == B and
        # tile-aligned shard height.  n_kv_heads=2 → ceil(2/32)*32 = 32.
        _TILE = 32
        upd_shard_h = ((-(-NUM_KV_HEADS // _TILE)) * _TILE) // B  # ceil-div then per-core
        upd_mem = ttnn.create_sharded_memory_config(
            [upd_shard_h, HEAD_DIM],
            ttnn.CoreGrid(x=1, y=B),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_upd = ttnn.to_memory_config(k_upd, upd_mem)
        v_upd = ttnn.to_memory_config(v_upd, upd_mem)
        ttnn.experimental.paged_update_cache(k_cache, k_upd, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v_upd, update_idxs_tensor=current_pos, page_table=page_table)
        # Q: [B, n_q, S, D] → [S, B, n_q, D] = [1, B, 8, 128]
        q_sdpa = ttnn.permute(q_4d, [2, 0, 1, 3])
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_sdpa,
            k_cache,
            v_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
        )  # [1, B, 8, 128]
        # Permute back: [S, B, n_q, D] → [B, S, n_q, D]
        attn_out = ttnn.permute(attn_out, [1, 0, 2, 3])
    else:
        is_causal = S > 1
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, is_causal=is_causal
        )  # [B, 8, S, 128]/device
        # Permute: [B, n_q, S, D] → [B, S, n_q, D]
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])

    # 7. Reshape to [B, S, 1024] for row-parallel O projection
    attn_out = ttnn.reshape(attn_out, [B, S, (NUM_HEADS // TP) * HEAD_DIM])  # [B, S, 1024]

    # 8. O projection: row-parallel → partial [B, S, 2688]/device
    wo_tt = _row(wo, mesh_device)
    out_tt = ttnn.linear(attn_out, wo_tt, transpose_b=True)  # [B, S, 2688] partial/device

    # 9. All-reduce → full [B, S, 2688] + residual
    result_tt = all_reduce(out_tt)
    return ttnn.add(residual, result_tt)
