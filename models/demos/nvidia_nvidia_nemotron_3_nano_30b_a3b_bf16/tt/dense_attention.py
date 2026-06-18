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

# Blackhole QB grid: 8×8 = 64 cores.  Chunk sizes follow tt_transformers precedent:
# 256 tiles for S≥2048 (each tile = 32 tokens, so 8192-token blocks), 64 tiles for S<2048.
# Multicore flash-attention avoids materialising the full S×S attention matrix in DRAM.
_SDPA_GRID = (8, 8)


def _sdpa_cfg(S: int):
    if S < 64:
        return None  # too small to chunk; single-core SDPA is fine
    chunk = 256 if S >= 2048 else 64
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=_SDPA_GRID,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )


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
    # Keep residual in DRAM — hidden_states was written by a compute kernel (safe write).
    # The DRAM defect is write-side only; reads from valid DRAM are safe.
    residual = hidden_states
    B = residual.shape[0]
    S = residual.shape[1]
    has_cache = kv_cache is not None and page_table is not None

    # 1. Pre-norm (residual in DRAM; reads are safe)
    w_tt = _rep_keyed(id(norm_weight), norm_weight.bfloat16().unsqueeze(0), mesh_device)
    normed_tt = ttnn.rms_norm(residual, epsilon=norm_eps, weight=w_tt)

    # 2. Q: column-parallel → [B, S, 1024]/device (8 heads/device)
    wq_tt = _col(wq, mesh_device)
    q_tt = ttnn.linear(normed_tt, wq_tt, transpose_b=True)  # [B, S, 1024]/device

    # 3. K: replicated (2 KV heads < TP=4)
    wk_tt = _rep(wk, mesh_device)
    k_tt = ttnn.linear(normed_tt, wk_tt, transpose_b=True)  # [B, S, 256]/device

    # 4. V: replicated
    wv_tt = _rep(wv, mesh_device)
    v_tt = ttnn.linear(normed_tt, wv_tt, transpose_b=True)  # [B, S, 256]/device

    # 5. Reshape to [B, heads, S, head_dim] for SDPA.
    # Prefill path (has_cache and S > 1): the tiled reshape kernel allocates a DRAM
    # mapping tensor on each shape cache-miss.  After prior layers recycle DRAM, that
    # allocation can land on device-2's defective page and hang.  RM-detour
    # (untilize → RM reshape → retilize, all to DRAM) avoids it; RM layout has no
    # mapping tensor so DRAM is safe for these intermediates.
    if has_cache and S > 1:
        _DRAM = ttnn.DRAM_MEMORY_CONFIG
        # Q: [B, S, 1024] TILE → RM(DRAM) → [B, S, 8, 128] RM(DRAM) → TILE(DRAM)
        _q_rm = ttnn.to_layout(q_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)
        _q_rm4 = ttnn.reshape(_q_rm, [B, S, NUM_HEADS // TP, HEAD_DIM], memory_config=_DRAM)
        del _q_rm
        q_4d = ttnn.to_layout(_q_rm4, ttnn.TILE_LAYOUT)  # → DRAM, tilize ≠ mapping tensor
        del _q_rm4
        q_4d = ttnn.permute(q_4d, [0, 2, 1, 3])
        # K: [B, S, 256] TILE → RM(DRAM) → [B, S, 2, 128] RM(DRAM) → TILE(DRAM)
        _k_rm = ttnn.to_layout(k_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)
        _k_rm4 = ttnn.reshape(_k_rm, [B, S, NUM_KV_HEADS, HEAD_DIM], memory_config=_DRAM)
        del _k_rm
        k_4d = ttnn.to_layout(_k_rm4, ttnn.TILE_LAYOUT)
        del _k_rm4
        k_4d = ttnn.permute(k_4d, [0, 2, 1, 3])
        # V: same pattern as K
        _v_rm = ttnn.to_layout(v_tt, ttnn.ROW_MAJOR_LAYOUT, memory_config=_DRAM)
        _v_rm4 = ttnn.reshape(_v_rm, [B, S, NUM_KV_HEADS, HEAD_DIM], memory_config=_DRAM)
        del _v_rm
        v_4d = ttnn.to_layout(_v_rm4, ttnn.TILE_LAYOUT)
        del _v_rm4
        v_4d = ttnn.permute(v_4d, [0, 2, 1, 3])
    else:
        # Decode (S=1) or no-cache: standard tiled reshape is safe.
        q_4d = ttnn.reshape(q_tt, [B, S, NUM_HEADS // TP, HEAD_DIM])
        q_4d = ttnn.permute(q_4d, [0, 2, 1, 3])
        k_4d = ttnn.reshape(k_tt, [B, S, NUM_KV_HEADS, HEAD_DIM])
        k_4d = ttnn.permute(k_4d, [0, 2, 1, 3])
        v_4d = ttnn.reshape(v_tt, [B, S, NUM_KV_HEADS, HEAD_DIM])
        v_4d = ttnn.permute(v_4d, [0, 2, 1, 3])

    # 6. Attention: select path based on S and whether a KV cache is present.
    if has_cache and S == 1 and current_pos is not None:
        # --- Paged Flash-Decode (single-token decode path) ---
        k_cache, v_cache = kv_cache
        k_upd = ttnn.permute(k_4d, [2, 0, 1, 3])  # [1, B, 2, 128]
        v_upd = ttnn.permute(v_4d, [2, 0, 1, 3])
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
        q_sdpa = ttnn.permute(q_4d, [2, 0, 1, 3])
        attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_sdpa,
            k_cache,
            v_cache,
            page_table_tensor=page_table,
            cur_pos_tensor=current_pos,
        )  # [1, B, 8, 128]
        attn_out = ttnn.permute(attn_out, [1, 0, 2, 3])
    elif has_cache and S > 1:
        # --- Chunked prefill: causal SDPA + bulk KV cache fill ---
        k_cache, v_cache = kv_cache
        _cfg = _sdpa_cfg(S)
        _sdpa_kwargs = {"program_config": _cfg} if _cfg is not None else {}
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, is_causal=True, **_sdpa_kwargs
        )  # [B, 8, S, 128]/device
        del q_4d  # not needed after SDPA; free to reduce peak DRAM
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])  # → [B, S, 8, 128] DRAM
    else:
        # --- No-cache path (stateless tests / reference) ---
        is_causal = S > 1
        _cfg = _sdpa_cfg(S)
        _sdpa_kwargs = {"program_config": _cfg} if _cfg is not None else {}
        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_4d, k_4d, v_4d, is_causal=is_causal, **_sdpa_kwargs
        )  # [B, 8, S, 128]/device
        attn_out = ttnn.permute(attn_out, [0, 2, 1, 3])

    # 7. Reshape [B, S, H, D] → [B, S, H*D] for row-parallel O projection.
    # Prefill RM-detour: avoids the TILE reshape's internal DRAM mapping tensor
    # that hangs on device-2's defective pages.  RM layout has no mapping tensor,
    # so DRAM is safe for the RM intermediates.
    if has_cache and S > 1:
        _attn_rm = ttnn.to_layout(attn_out, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        del attn_out
        _attn_rm3 = ttnn.reshape(_attn_rm, [B, S, (NUM_HEADS // TP) * HEAD_DIM], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        del _attn_rm
        attn_out = ttnn.to_layout(_attn_rm3, ttnn.TILE_LAYOUT)  # → DRAM (tilize ≠ mapping tensor)
        del _attn_rm3
    else:
        # Decode path (S=1) or no-cache: standard reshape is safe.
        attn_out = ttnn.reshape(attn_out, [B, S, (NUM_HEADS // TP) * HEAD_DIM])

    # 8. O projection: row-parallel → partial [B, S, 2688]/device
    wo_tt = _row(wo, mesh_device)
    out_tt = ttnn.linear(attn_out, wo_tt, transpose_b=True)  # [B, S, 2688] partial/device

    # 9. All-reduce → full [B, S, 2688] + residual.
    # Device-2 DRAM defect workaround: copy to L1 before CCL so reads are from safe SRAM.
    # Cap at S <= 8192: at 16K tokens the tensor is ~86 MB which exceeds L1 capacity
    # (~50 MB available).  Above the cap, all_reduce runs directly from DRAM; the output
    # of ttnn.linear is a fresh DRAM allocation that lands on safe pages.
    if S <= 8192:
        out_tt = ttnn.to_memory_config(out_tt, ttnn.L1_MEMORY_CONFIG)
        result_tt = all_reduce(out_tt)
        del out_tt  # free L1 before result_tt moves to L1 (avoids two peaks coexisting)
        result_tt = ttnn.to_memory_config(result_tt, ttnn.L1_MEMORY_CONFIG)
    else:
        result_tt = all_reduce(out_tt)
        del out_tt
    # Both inputs in L1 (safe reads for S<=8192); output to DRAM (fresh allocation).
    ret = ttnn.add(residual, result_tt)
    del result_tt
    del residual  # async-safe: device executes add before processing this dealloc

    # 10. Fill KV cache for prefill after CCL.
    # k_4d/v_4d move to L1 so paged_fill_cache reads from safe on-chip SRAM.
    if has_cache and S > 1:
        k_4d_l1 = ttnn.to_memory_config(k_4d, ttnn.L1_MEMORY_CONFIG)
        v_4d_l1 = ttnn.to_memory_config(v_4d, ttnn.L1_MEMORY_CONFIG)
        ttnn.experimental.paged_fill_cache(k_cache, k_4d_l1, page_table, batch_idx=0)
        ttnn.experimental.paged_fill_cache(v_cache, v_4d_l1, page_table, batch_idx=0)
        del k_4d_l1
        del v_4d_l1

    return ret
