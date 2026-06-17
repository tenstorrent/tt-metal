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
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.blackhole.qwen3_5_9b.tt.attention.kv_cache import init_kv_cache
from models.demos.blackhole.qwen3_5_9b.tt.attention.operations import (
    apply_gate_projection,
    apply_qkv_projection,
    split_qkv_heads_decode,
    split_qkv_heads_prefill,
)
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.demos.blackhole.qwen3_5_9b.tt.attention.weights import load_attention_weights
from models.demos.blackhole.qwen3_5_9b.tt.rms_norm import qwen35_rms_norm
from models.tt_transformers.tt.ccl import tt_all_reduce


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
        """
        hidden_states: [B, 1, S, dim]
        cos_tt, sin_tt: [max_seq_len, rope_dim] RoPE tensors
        """
        weights = self.weights
        NH, NKV, HD = self.NH, self.NKV, self.HD
        B, _, S, _ = hidden_states.shape

        # 1. fused Q+K+V projection + separate gate projection
        xqkv = apply_qkv_projection(hidden_states, weights)
        gate = apply_gate_projection(hidden_states, weights)

        # 2. create QKV heads. nlp_create_qkv_heads does the [B,1,S,*] -> [B,H,S,HD]
        # The gate is a Qwen3.5-specific 4th projection (NH heads) the QKV op can't emit,
        # so it keeps the explicit reshape+transpose and lands in q's [B, NH, S, HD] layout.
        q, k, v = split_qkv_heads_prefill(xqkv, NH, NKV)
        gate = ttnn.transpose(ttnn.reshape(gate, (B, S, NH, HD)), 1, 2)

        # 3. apply RMS norms
        q = qwen35_rms_norm(q, weight=weights.w_q_norm, eps=self.eps, scale=True)
        k = qwen35_rms_norm(k, weight=weights.w_k_norm, eps=self.eps, scale=True)

        # 4. apply RoPE
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # 5. update kv cache. Both fill_cache (contiguous) and paged_fill_cache (paged) write
        #    one batch row per call in this scalar-batch_idx form, so a B>1 prefill fills users
        #    one at a time — the same per-user loop gemma4's batched prefill uses
        #    (gemma4/tt/attention/prefill.py fills valid_slots with batch_idx=slot). User b's
        #    [1,NKV,S,HD] slice lands in slot user_id+b; for the paged cache that batch_idx also
        #    selects user b's page_table row, so the caller must pass a page_table whose rows
        #    cover [user_id, user_id+B). B==1 keeps the no-slice fast path. Paged vs contiguous
        #    differ only in which fill op writes the slice — the batched SDPA below is batch-general.
        if self.kv_cache is not None:
            k_cache, v_cache = self.kv_cache  # unpack the single [k_cache, v_cache] slot
            for b in range(B):
                k_b = k if B == 1 else ttnn.slice(k, (b, 0, 0, 0), (b + 1, NKV, S, HD))
                v_b = v if B == 1 else ttnn.slice(v, (b, 0, 0, 0), (b + 1, NKV, S, HD))
                if page_table is not None:
                    ttnn.experimental.paged_fill_cache(k_cache, k_b, page_table, batch_idx=user_id + b)
                    ttnn.experimental.paged_fill_cache(v_cache, v_b, page_table, batch_idx=user_id + b)
                else:
                    ttnn.fill_cache(k_cache, k_b, batch_idx=user_id + b)
                    ttnn.fill_cache(v_cache, v_b, batch_idx=user_id + b)

        # 6. global attention
        attn = ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # 7. apply gate and output projection
        gated = attn * ttnn.sigmoid(gate)  # [B,NH,S,HD]

        # [B,NH,S,HD] -> [B,S,NH,HD] -> [B,1,S,NH*HD]
        gated = ttnn.transpose(gated, 1, 2)
        gated = ttnn.reshape(gated, (B, 1, S, NH * HD))
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
        """
        hidden_states: [1, 1, B, dim] (framework layout for decode)
        cur_pos_tt: [1] scalar tensor with the current decode position (0-based)
        cos_tt, sin_tt: [max_seq_len, rope_dim] RoPE tensors
        """
        assert self.kv_cache is not None, "forward_decode requires allocated KV caches"
        k_cache, v_cache = self.kv_cache

        weights = self.weights
        B, NH, NKV, HD = self.B, self.NH, self.NKV, self.HD

        # 1. fused Q+K+V projection + separate gate projection. xqkv: [1, B, H, HD]
        # The gate is projected + reshaped separately into q's [1, B, NH, HD] layout.
        xqkv = apply_qkv_projection(hidden_states, weights)
        gate = apply_gate_projection(hidden_states, weights)

        # 2. Split QKV heads.
        q, k, v = split_qkv_heads_decode(xqkv, NH, NKV)
        gate = ttnn.reshape(gate, (1, B, NH, HD))
        # 3. apply RMS norms
        q = qwen35_rms_norm(q, weight=weights.w_q_norm, eps=self.eps, scale=True)
        k = qwen35_rms_norm(k, weight=weights.w_k_norm, eps=self.eps, scale=True)

        # 4. apply RoPE.
        q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)

        # TODO This is not trace compatible
        k = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
        v = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0)

        k = ttnn.to_memory_config(k, self.args.kv_update_shard_cfg)
        v = ttnn.to_memory_config(v, self.args.kv_update_shard_cfg)

        # The cache update is identical for both paths: paged_update_cache's page_table
        # kwarg defaults to None, so passing page_table=None is exactly the contiguous tensor cache update.
        ttnn.experimental.paged_update_cache(k_cache, k, update_idxs_tensor=cur_pos_tt, page_table=page_table)
        ttnn.experimental.paged_update_cache(v_cache, v, update_idxs_tensor=cur_pos_tt, page_table=page_table)

        if page_table is not None:
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                k_cache,
                v_cache,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                k_cache,
                v_cache,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
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
