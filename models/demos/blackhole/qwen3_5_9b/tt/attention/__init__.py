# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel (TP>1) full-attention path for Qwen3.5.

Ported from models/demos/qwen35_27b/tt/attention.py forward_decode.

Long-context (64k+) correctness choices, validated at 64k on Qwen3.5-27B-FP8 and Qwen3.6-27B:
  - HYBRID q/k-norm scaling: PREFILL uses the HF-correct (1+weight) scale (sharp attention,
    required for retrieval — without it long-context attention is uniform and retrieval is zero);
    DECODE uses the raw weights (flat attention, robust to the small per-step decode noise that
    sharp attention amplifies into loops). See load_attention_weights_tp / forward_decode.
  - Q stays bf16 into the chunked SDPA (forward_prefill), NOT bf8. Casting Q to bf8 was
    the real long-context degeneration cause (bf16-Q → coherent 64k/256k summary; bf8-Q →
    loops/gibberish), matching the 9B's deliberately-bf16 path (ttnn_gated_attention.py:277).
    env QWEN_SDPA_BF8_Q=1 restores the old bf8 cast for comparison.
  - weights are kept INTERLEAVED per device (no DRAM-width-sharding) and matmuls
    use ttnn's auto program config — same robust pattern validated for the MLP.

Decode input/output use the framework layout: x [1,1,B,dim] replicated in; output
fractured along dim=3 (reduce-scatter). Column-parallel q/k/v, row-parallel wo.
"""
import os

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt.attention.kv_cache import init_kv_cache
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.demos.blackhole.qwen3_5_9b.tt.attention.weights import load_attention_weights
from models.demos.blackhole.qwen3_5_9b.tt.rms_norm import qwen35_rms_norm
from models.demos.blackhole.qwen3_5_9b.tt.attention.operations import apply_qkvg_projection
from models.tt_transformers.tt.ccl import tt_all_reduce
from models.common.lightweightmodule import LightweightModule


class Qwen35Attention(LightweightModule):
    """Standalone TP full-attention with internal per-head KV caches (decode)."""

    def __init__(self, mesh_device, state_dict, args, tt_ccl, create_kv_cache=False, tensor_cache_path=None):
        self.mesh_device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.weights = load_attention_weights(
            mesh_device=mesh_device, state_dict=state_dict, args=args, tensor_cache_path=tensor_cache_path
        )

        # Single KV-cache slot holding the [k_cache, v_cache] list straight from init_kv_cache
        # (mirrors gemma4's self.kv_cache). One attribute keeps the demo (contiguous) and vLLM
        # (paged) caches in the same handle; callers unpack k_cache, v_cache = self.kv_cache.
        if create_kv_cache:
            self.kv_cache = init_kv_cache(
                mesh_device=mesh_device,
                args=args,
                max_batch_size=args.max_batch_size,
                max_seq_len=args.max_seq_len,
                paged_attention_config=None,
                cache_dtype=ttnn.bfloat16,
                tensor_cache_path=tensor_cache_path,
            )
        else:
            self.kv_cache = None

        self.B = args.max_batch_size
        self.NH = args.n_local_heads
        self.NKV = args.n_local_kv_heads
        self.HD = args.head_dim
        self.scale = self.HD**-0.5
        self.rope_dim = args.rope_head_dim
        self.eps = args.norm_eps

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Bind an externally-allocated paged KV cache into the single cache slot.

        Overwrites self.kv_cache (the [k_cache, v_cache] list) — the same slot forward_decode
        and forward_prefill_paged read. The internal contiguous cache and the external paged
        cache are mutually exclusive per deployment, so one slot serves both; the op variant
        is selected by page_table at call time, not by which cache is bound (mirrors gemma4).
        One call after allocate_kv_caches.
        """
        self.kv_cache = [k_cache, v_cache]

    def forward_prefill(
        self,
        hidden_states,
        cos_tt,
        sin_tt,
        page_table=None,
        user_id=0,
    ):
        weights = self.weights
        NH, NKV, HD = self.NH, self.NKV, self.HD
        S = hidden_states.shape[-2]

        # 1. apply q/k/v/g projections
        q, k, v, gate = apply_qkvg_projection(hidden_states, weights)

        # 2. reshape to heads format / shape for attention
        # reshape from [1, 1, S, NH * HD] -> [1, NH, S, HD] for q and gate
        # reshape from [1, 1, S, NKV * HD] -> [1, NKV, S, HD] for k and v
        q = ttnn.transpose(ttnn.reshape(q, (1, S, NH, HD)), 1, 2)
        gate = ttnn.transpose(ttnn.reshape(gate, (1, S, NH, HD)), 1, 2)
        k = ttnn.transpose(ttnn.reshape(k, (1, S, NKV, HD)), 1, 2)
        v = ttnn.transpose(ttnn.reshape(v, (1, S, NKV, HD)), 1, 2)

        # 3. apply RMS norms
        q = qwen35_rms_norm(q, weight=weights.w_q_norm, eps=self.eps, scale=True)
        k = qwen35_rms_norm(k, weight=weights.w_k_norm, eps=self.eps, scale=True)

        # 4. apply RoPE
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache  # unpack the single [k_cache, v_cache] slot
            if page_table is not None:
                ttnn.experimental.paged_fill_cache(k_cache, k, page_table, batch_idx=user_id)
                ttnn.experimental.paged_fill_cache(v_cache, v, page_table, batch_idx=user_id)
            else:
                ttnn.fill_cache(k_cache, k, batch_idx=user_id)
                ttnn.fill_cache(v_cache, v, batch_idx=user_id)

        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        gated = attn * ttnn.sigmoid(gate)  # [1,NH,S,HD]

        # [1,NH,S,HD] -> [1,S,NH,HD] -> [1,1,S,NH*HD]
        gated = ttnn.transpose(gated, 1, 2)
        gated = ttnn.reshape(gated, (1, 1, S, NH * HD))
        out = ttnn.linear(gated, weights.wo)

        return tt_all_reduce(
            out,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_decode(self, hidden_states, cur_pos_tt, cos_tt, sin_tt, page_table=None):
        assert self.kv_cache is not None, "forward_decode requires allocated KV caches"
        k_cache, v_cache = self.kv_cache

        weights = self.weights
        B, NH, NKV, HD = self.B, self.NH, self.NKV, self.HD

        q, k, v, gate = apply_qkvg_projection(hidden_states, weights)

        q = ttnn.reshape(q, (1, B, NH, HD))
        gate = ttnn.reshape(gate, (1, B, NH, HD))

        k = ttnn.reshape(k, (1, B, NKV, HD))
        v = ttnn.reshape(v, (1, B, NKV, HD))

        q = qwen35_rms_norm(q, weight=weights.w_q_norm, eps=self.eps, scale=True)
        k = qwen35_rms_norm(k, weight=weights.w_k_norm, eps=self.eps, scale=True)

        q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)

        # Single KV-cache slot; the op variant is chosen by page_table, not by a separate
        # cache attribute (mirrors gemma4 decode.py). The demo path binds an internal
        # contiguous cache (init_kv_cache); the vLLM path binds an external paged cache
        # (set_paged_kv_cache). Both are [B, n_local_kv_heads, max_seq, HD] handles here, so
        # the pad-to-32 (TILE_SIZE) + height-sharded update is identical — only the SDPA op
        # differs (paged variant takes the page_table). ONE in-place update of all local
        # heads at cur_pos, then SDPA-decode straight off the cache.
        k = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
        v = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0)

        k = ttnn.to_memory_config(k, self.args.kv_update_shard_cfg)
        v = ttnn.to_memory_config(v, self.args.kv_update_shard_cfg)

        # The cache update is identical for both paths: paged_update_cache's page_table
        # kwarg defaults to None, so passing page_table=None is exactly the contiguous
        # (demo) update. SDPA cannot merge the same way — the paged op is a separate ttnn
        # entry point that takes page_table_tensor as a REQUIRED positional, so the op
        # choice stays a one-line fork on whether a page table was supplied.
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos_tt, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos_tt, page_table=page_table)

        # SDPA-decode needs an explicit program config. With program_config=None the op falls
        # back to the full device grid, whose static circular buffers (a) clash with the
        # height-sharded K/V still resident in L1 from the cache update above, and (b) exceed
        # the 64-cores/head reduction-tree cap (MAX_TREE_REDUCTION_ROUNDS=6) when n_local_kv_heads
        # is small (1 at TP=4). SDPAProgramConfig's default max_cores_per_head_batch=16 caps the
        # tree; the q/k chunk sizes mirror gemma4's validated decode path. Output to DRAM since
        # the caches it reads are DRAM-resident.
        sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=64,
            exp_approx_mode=False,
        )

        if page_table is not None:
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                k_cache,
                v_cache,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=sdpa_program_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                k_cache,
                v_cache,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=sdpa_program_config,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        gated = attn_out * ttnn.sigmoid(gate)
        gated = ttnn.reshape(gated, (1, B, NH * HD))
        wo_partial = ttnn.linear(gated, weights.wo)
        wo_partial = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))

        return tt_all_reduce(
            wo_partial,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
