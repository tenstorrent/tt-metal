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
from models.tt_transformers.tt.ccl import tt_all_reduce


class TPAttention:
    """Standalone TP full-attention with internal per-head KV caches (decode)."""

    def __init__(self, mesh_device, state_dict, args, tt_ccl, create_kv_cache=False, tensor_cache_path=None):
        self.mesh = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.weights = load_attention_weights(
            mesh_device=mesh_device, state_dict=state_dict, tensor_cache_path=tensor_cache_path
        )

        if create_kv_cache:
            self.k_caches, self.v_caches = init_kv_cache(
                mesh_device=mesh_device,
                args=args,
                max_batch_size=args.max_batch_size,
                max_seq_len=args.max_seq_len,
                paged_attention_config=None,
                cache_dtype=ttnn.bfloat16,
                tensor_cache_path=tensor_cache_path,
            )
        else:
            self.k_caches, self.v_caches = None, None

        self.B = args.max_batch_size
        self.NH = args.n_local_heads
        self.NKV = args.n_local_kv_heads
        self.HD = args.head_dim
        self.scale = self.HD**-0.5
        self.rope_dim = args.rope_head_dim

    def __call__(
        self,
        hidden_states,
        rope_mats=None,
        position_idx=None,
        page_table=None,
        kv_cache=None,
        is_decode=True,
    ):
        if is_decode:
            return
        else:
            return

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach an externally-allocated paged KV cache (one call after allocate_kv_caches)."""
        self.paged_k = k_cache
        self.paged_v = v_cache
        self.use_paged = True

    def forward_prefill(
        self,
        hidden_states,
        cos_tt,
        sin_tt,
    ):
        weights = self.weights
        NH, NKV, HD = self.NH, self.NKV, self.HD
        S = hidden_states.shape[-2]

        q_and_gate = ttnn.linear(hidden_states, weights.wq)  # wq is the fused q+gate proj (out = NH*2*HD)
        k = ttnn.linear(hidden_states, weights.wk)
        v = ttnn.linear(hidden_states, weights.wv)

        # [1,1,S,NH*HD*2] -> [1,S,NH,2*HD] -> split -> [1,NH,S,HD]
        q_and_gate = ttnn.reshape(q_and_gate, (1, S, NH, 2 * HD))
        q = ttnn.transpose(ttnn.slice(q_and_gate, (0, 0, 0, 0), (1, S, NH, HD)), 1, 2)
        gate = ttnn.transpose(ttnn.slice(q_and_gate, (0, 0, 0, HD), (1, S, NH, 2 * HD)), 1, 2)
        ttnn.deallocate(q_and_gate)

        k = ttnn.transpose(
            ttnn.reshape(k, (1, S, NKV, HD)), 1, 2
        )  # [1,S,NKV,HD] -> [1,NKV,S,HD] -> transpose -> [1,NKV,S,HD]
        v = ttnn.transpose(
            ttnn.reshape(v, (1, S, NKV, HD)), 1, 2
        )  # [1,S,NKV,HD] -> [1,NKV,S,HD] -> transpose -> [1,NKV,S,HD]

        q = ttnn.rms_norm(q, epsilon=1e-6) * (1 + weights.w_q_norm)
        k = ttnn.rms_norm(k, epsilon=1e-6) * (1 + weights.w_k_norm)

        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # Fill the KV cache with the prompt's (post-RoPE) K/V so decode continues from
        # position S, only when caches are allocated (stateful demo path). self.k_caches/
        # v_caches are single [B, n_local_kv_heads, max_seq, HD] tensors (init_kv_cache) and
        # k/v are already [1, NKV, S, HD], so ONE fill_cache writes every local head at once —
        # the gemma4 single-write pattern, replacing the old per-head slice+fill loop.
        # batch_idx=0: prefill fills a single user.
        if self.k_caches is not None:
            ttnn.fill_cache(self.k_caches, k, batch_idx=0)
            ttnn.fill_cache(self.v_caches, v, batch_idx=0)

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
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_decode(self, x, cur_pos_tt, cos_tt, sin_tt, page_table=None):
        tw, B, NH, NKV, HD = self.tw, self.B, self.NH, self.NKV, self.HD
        use_paged = self.use_paged and page_table is not None
        if not use_paged and self.k_caches is None:
            self.reset_state()

        qg = ttnn.linear(x, tw["wqkv"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kp = ttnn.linear(x, tw["wk"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vp = ttnn.linear(x, tw["wv"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        qg_r = ttnn.reshape(qg, (1, B, NH, HD * 2))
        ttnn.deallocate(qg)
        q = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD))
        gate = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, HD * 2))
        ttnn.deallocate(qg_r)
        k = ttnn.reshape(kp, (1, B, NKV, HD))
        ttnn.deallocate(kp)
        v = ttnn.reshape(vp, (1, B, NKV, HD))
        ttnn.deallocate(vp)

        # QK RMSNorm — DEFAULT raw (no-+1) at DECODE only (hybrid scaling): sharp +1 prefill
        # retrieves the long context; flat decode averages over keys so the per-step decode noise
        # cannot flip retrieval (the loop/junk failure mode). Validated 64k on 3.5-FP8 AND 3.6
        # (coherent Frankenstein summaries; sharp decode loops/junks both). QWEN35_QKNORM_DECODE_SHARP=1
        # reverts decode to +1.
        _sharp = os.environ.get("QWEN35_QKNORM_DECODE_SHARP") == "1"
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6), tw["q_norm" if _sharp else "q_norm_flat"])
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6), tw["k_norm" if _sharp else "k_norm_flat"])

        q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)

        # Cap the SDPA-decode grid to 64 cores (tree-reduction limit); auto-grid
        # grabs all 110 P150 cores for a single user (B=1) and overflows.
        sdpa_dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
        if use_paged:
            # External paged KV cache (vLLM/contract path): update at cur_pos via the
            # page_table, then paged SDPA-decode. Mirrors qwen35_27b attention.py:188-218.
            keys, values = self.paged_k, self.paged_v
            k_p = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
            v_p = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k_sh = ttnn.to_memory_config(k_p, self.args.kv_update_shard_cfg)
            v_sh = ttnn.to_memory_config(v_p, self.args.kv_update_shard_cfg)
            ttnn.deallocate(k_p)
            ttnn.deallocate(v_p)
            ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            ttnn.deallocate(k_sh)
            ttnn.deallocate(v_sh)
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=sdpa_dec_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)
        else:
            # Internal combined KV cache (demo/standalone): ONE in-place update of all local
            # heads at cur_pos, then SDPA-decode straight off the cache — mirrors the paged
            # branch above and gemma4's single-write path, dropping the old per-head update
            # loop + concat. self.k_caches/v_caches are single [B, n_local_kv_heads, max_seq,
            # HD] tensors; pad the head dim to 32 (TILE_SIZE) for the height-sharded update.
            k_p = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
            v_p = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k_sh = ttnn.to_memory_config(k_p, self.args.kv_update_shard_cfg)
            v_sh = ttnn.to_memory_config(v_p, self.args.kv_update_shard_cfg)
            ttnn.deallocate(k_p)
            ttnn.deallocate(v_p)
            ttnn.experimental.paged_update_cache(self.k_caches, k_sh, update_idxs_tensor=cur_pos_tt)
            ttnn.experimental.paged_update_cache(self.v_caches, v_sh, update_idxs_tensor=cur_pos_tt)
            ttnn.deallocate(k_sh)
            ttnn.deallocate(v_sh)

            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                self.k_caches,
                self.v_caches,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=sdpa_dec_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(q)

        gated = ttnn.multiply(attn_out, ttnn.sigmoid(gate))
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate)

        gated_flat = ttnn.reshape(gated, (1, B, NH * HD))
        ttnn.deallocate(gated)
        wo_partial = ttnn.linear(
            gated_flat, tw["wo"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(gated_flat)
        wo_partial = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))
        return tt_all_reduce(
            wo_partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_prefill_paged(
        self,
        x,
        cos_tt,
        sin_tt,
        page_table,
        chunk_page_table=None,
        chunk_start_idx=0,
        chunk_start_idx_tensor=None,
        user_id=0,
    ):
        """Paged-KV prefill for one chunk of a sequence (vLLM / model-contract path).

        Fills the external paged KV cache (self.paged_k/v) for this chunk via
        paged_fill_cache, then runs chunked SDPA over the paged cache so the chunk
        attends to all prior chunks. x: [1,1,S,dim] replicated; cos/sin sliced to this
        chunk; chunk_start_idx is the chunk's absolute token offset. Output fractured
        along dim=3. Mirrors qwen35_27b/tt/attention.py forward_prefill_paged, adapted
        to the integrated interleaved-matmul / reshape conventions used by forward_prefill.

        chunk_start_idx_tensor: optional device tensor [1] int32. When supplied, the
        chunked SDPA uses the FLEXIBLE path (runtime device offset + a fixed q/k_chunk=64
        program config), so a single captured trace / compiled program serves every chunk
        position — the masked-bucket + chunk-outer-trace path (mirrors the single-device
        gated_attention_forward_ttnn flexible branch). chunk_start_idx (int) is still used
        host-side to size the page table; the op consumes the tensor.
        """
        assert self.use_paged and self.paged_k is not None, "forward_prefill_paged requires a bound paged KV cache"
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        if chunk_start_idx is None:
            chunk_start_idx = 0
        S = x.shape[-2]

        qg = ttnn.linear(x, tw["wqkv"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kp = ttnn.linear(x, tw["wk"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vp = ttnn.linear(x, tw["wv"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        qg = ttnn.reshape(qg, (1, S, NH, 2 * HD))
        q = ttnn.transpose(ttnn.slice(qg, (0, 0, 0, 0), (1, S, NH, HD)), 1, 2)
        gate = ttnn.transpose(ttnn.slice(qg, (0, 0, 0, HD), (1, S, NH, 2 * HD)), 1, 2)
        ttnn.deallocate(qg)
        k = ttnn.transpose(ttnn.reshape(kp, (1, S, NKV, HD)), 1, 2)
        ttnn.deallocate(kp)
        v = ttnn.transpose(ttnn.reshape(vp, (1, S, NKV, HD)), 1, 2)
        ttnn.deallocate(vp)

        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6), tw["q_norm"])
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6), tw["k_norm"])
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # Fill only this chunk's positions into the paged cache.
        k_paged, v_paged = self.paged_k, self.paged_v
        block_size = k_paged.shape[2]
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[1] * block_size
        if page_len < S:
            k_fill = ttnn.slice(k, (0, 0, 0, 0), (1, NKV, page_len, HD))
            v_fill = ttnn.slice(v, (0, 0, 0, 0), (1, NKV, page_len, HD))
        else:
            k_fill, v_fill = k, v
        ttnn.experimental.paged_fill_cache(k_paged, k_fill, fill_page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_paged, v_fill, fill_page_table, batch_idx=user_id)
        if page_len < S:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Chunked SDPA over the paged cache (attends to prior chunks via page_table).
        # Keep Q in bf16 (matching the 9B's working chunked-SDPA path, ttnn_gated_attention.py:277
        # "Q/K/V stay bfloat16"). Casting Q to bf8 was the long-context degeneration cause: at 64k
        # the bf8-Q output collapsed (looping/gibberish) while bf16-Q produced a coherent themed
        # summary — confirmed by A/B on bf16 Qwen3.5-27B (thinking-off, greedy). Q is per-chunk
        # (small), so bf16 costs negligible L1/DRAM and showed no perf delta. QWEN_SDPA_BF8_Q=1
        # restores the old bf8 cast for comparison. When bf16, q IS q8 (freed once at deallocate(q8)).
        if os.environ.get("QWEN_SDPA_BF8_Q") == "1":
            q8 = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
            ttnn.deallocate(q)
        else:
            q8 = q

        # chunked SDPA requires chunk_start_idx % q_chunk_size == 0. The FLEXIBLE path
        # (device-tensor offset) fixes q/k_chunk=64 so ONE program serves every chunk
        # position (64 divides any 2048-multiple chunk_start) — required for a single
        # captured trace / pre-warmed bucket program. The int path picks the largest
        # power-of-two chunk dividing chunk_start_idx (any size divides 0).
        if chunk_start_idx_tensor is not None:
            qk_chunk = 64
        else:
            cap = 256 if S >= 2048 else 64
            qk_chunk = cap if not chunk_start_idx else min(cap, chunk_start_idx & -chunk_start_idx)
        # Use the FULL Blackhole worker grid (13x10=130 on P150) to match the reference 27B and
        # avoid the WH-era (8,8)=64-core cap (~49% SDPA utilization on BH). NOTE: this is a PERF
        # alignment only — verified bit-identical to (8,8) at long context (flash attention is
        # grid-invariant), so it does NOT affect correctness. See test_tp_chunked_prefill_pcc_sweep.
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=qk_chunk,
            k_chunk_size=qk_chunk,
        )

        # Pad the SDPA page table with zero blocks so (a) K covers a padded/short Q
        # (K < Q + chunk_start_idx — padded slots are masked out / never filled), and
        # (b) the stick size is a multiple of 32, which the (flexible) chunked SDPA kernel
        # requires. The extra logical blocks map to physical block 0 but sit at K positions
        # beyond the prompt, so causality masks them out of every real query.
        sdpa_page_table = page_table
        needed_blocks = (S + chunk_start_idx + block_size - 1) // block_size
        target_blocks = max(needed_blocks, page_table.shape[-1])
        target_blocks = ((target_blocks + 31) // 32) * 32
        if page_table.shape[-1] < target_blocks:
            zeros_pad = ttnn.zeros(
                (page_table.shape[0], target_blocks - page_table.shape[-1]),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            sdpa_page_table = ttnn.concat([page_table, zeros_pad], dim=-1)
            ttnn.deallocate(zeros_pad)

        if chunk_start_idx_tensor is not None:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q8,
                input_tensor_k=k_paged,
                input_tensor_v=v_paged,
                page_table_tensor=sdpa_page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                compute_kernel_config=self.compute_cfg,
                program_config=sdpa_cfg,
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q8,
                input_tensor_k=k_paged,
                input_tensor_v=v_paged,
                page_table_tensor=sdpa_page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=self.compute_cfg,
                program_config=sdpa_cfg,
            )
        if sdpa_page_table is not page_table:
            ttnn.deallocate(sdpa_page_table)
        ttnn.deallocate(q8)

        gated = ttnn.multiply(attn, ttnn.sigmoid(gate))
        ttnn.deallocate(attn)
        ttnn.deallocate(gate)
        gated = ttnn.transpose(gated, 1, 2)
        gated = ttnn.reshape(gated, (1, 1, S, NH * HD))
        partial = ttnn.linear(
            gated, tw["wo"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(gated)
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
