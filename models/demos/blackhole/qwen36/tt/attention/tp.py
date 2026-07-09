# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel (TP>1) full-attention path for Qwen3.5.

Long-context (64k+) correctness choices, validated at 64k on Qwen3.6-27B:
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

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.tt_transformers.tt.ccl import tt_all_reduce


def load_attention_weights_tp(mesh, state_dict, args, cache_dir=None):
    """Shard one full-attention layer's weights across the mesh.

    state_dict keys (from the FP8 loader / 9B substate): q_proj/k_proj/v_proj/
    o_proj/q_norm/k_norm. q_proj is the fused per-head [Q,gate] projection.
    """
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    tw = {}
    # Column-parallel: shard output dim (contiguous heads per device, gate kept with Q)
    tw["wqkv"] = tpc.shard_w(
        state_dict["q_proj.weight"],
        mesh,
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("wqkv"),
        dtype=ttnn.bfloat8_b,
    )
    tw["wk"] = tpc.shard_w(
        state_dict["k_proj.weight"],
        mesh,
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("wk"),
        dtype=ttnn.bfloat8_b,
    )
    tw["wv"] = tpc.shard_w(
        state_dict["v_proj.weight"],
        mesh,
        dim=-1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("wv"),
        dtype=ttnn.bfloat8_b,
    )
    # Row-parallel: shard input dim → reduce-scatter after
    tw["wo"] = tpc.shard_w(
        state_dict["o_proj.weight"],
        mesh,
        dim=0,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("wo"),
        dtype=ttnn.bfloat8_b,
    )
    # Per-head QK norms, replicated. HF Qwen3_5RMSNorm computes output*(1+weight) and the ckpts
    # store raw zero-centered weights (means ~0.32-0.58), so +1 is the correct scale. Without it
    # Q·K logits are ~14x too small and long-context attention is UNIFORM -> zero retrieval at
    # 64k. Don't cache (tiny tensors, read-only ckpt dir).
    tw["q_norm"] = tpc.replicate(state_dict["q_norm.weight"].to(torch.float32) + 1.0, mesh, None)
    tw["k_norm"] = tpc.replicate(state_dict["k_norm.weight"].to(torch.float32) + 1.0, mesh, None)
    # Flat (raw, no +1) twins — the decode scale (hybrid): flat decode attention averages over
    # keys so per-step decode noise can't flip retrieval (loops/junk). Negligible cost.
    tw["q_norm_flat"] = tpc.replicate(state_dict["q_norm.weight"].to(torch.float32), mesh, None)
    tw["k_norm_flat"] = tpc.replicate(state_dict["k_norm.weight"].to(torch.float32), mesh, None)
    return tw


class TPAttention:
    """Standalone TP full-attention with internal per-head KV caches (decode)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        self.B = args.max_batch_size
        self.NH = args.n_local_heads
        self.NKV = args.n_local_kv_heads
        self.HD = args.head_dim
        self.scale = self.HD**-0.5
        self.rope_dim = args.rope_head_dim
        self.compute_cfg = tpc.COMPUTE_HIFI2
        self.k_caches = None
        self.v_caches = None
        # Paged KV cache (vLLM / model-contract path). Bound via set_paged_kv_cache;
        # when use_paged, decode/prefill read+write this external paged cache through
        # page_table instead of the internal concat caches above (which stay for the demo).
        self.paged_k = None
        self.paged_v = None
        self.use_paged = False

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach an externally-allocated paged KV cache (one call after allocate_kv_caches)."""
        self.paged_k = k_cache
        self.paged_v = v_cache
        self.use_paged = True

    def reset_state(self):
        def z():
            return ttnn.from_torch(
                torch.zeros(self.B, 1, self.args.max_seq_len, self.HD, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.k_caches = [z() for _ in range(self.NKV)]
        self.v_caches = [z() for _ in range(self.NKV)]

    def forward_prefill(self, x, cos_tt, sin_tt):
        """Causal prefill over a full sequence. x: [1,1,S,dim] replicated;
        cos/sin: [1,1,S,rope_dim]. Output fractured along dim=3 (reduce-scatter)."""
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        S = x.shape[-2]

        qg = ttnn.linear(x, tw["wqkv"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kp = ttnn.linear(x, tw["wk"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        vp = ttnn.linear(x, tw["wv"], compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # [1,1,S,NH*HD*2] -> [1,S,NH,2*HD] -> split -> [1,NH,S,HD]
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

        # Fill the per-head KV cache with the prompt's (post-RoPE) K/V so decode
        # continues from position S. Only when caches are allocated (stateful path).
        if self.k_caches is not None:
            # Don't deallocate the slices — for NKV==1 they alias k/v, which are
            # still needed for SDPA below.
            for h in range(NKV):
                ttnn.fill_cache(self.k_caches[h], ttnn.slice(k, (0, h, 0, 0), (1, h + 1, S, HD)), 0)
                ttnn.fill_cache(self.v_caches[h], ttnn.slice(v, (0, h, 0, 0), (1, h + 1, S, HD)), 0)

        # Keep Q/K/V bf16 into SDPA. The unconditional bf8 cast here was inconsistent with the documented bf16-Q fix and degraded
        # the bespoke prefill_tp oracle vs the paged path. QWEN_SDPA_BF8_Q=1 restores the old bf8.
        if os.environ.get("QWEN_SDPA_BF8_Q") == "1":
            q8 = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
            k8 = ttnn.typecast(k, dtype=ttnn.bfloat8_b)
            v8 = ttnn.typecast(v, dtype=ttnn.bfloat8_b)
            ttnn.deallocate(q)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        else:
            q8, k8, v8 = q, k, v
        padded = max(32, ((S + 31) // 32) * 32)
        ch = min(256 if S >= 2048 else 64, padded)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=ch, k_chunk_size=ch
        )
        attn = ttnn.transformer.scaled_dot_product_attention(
            q8, k8, v8, is_causal=True, scale=self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=sdpa_cfg
        )
        ttnn.deallocate(q8)
        ttnn.deallocate(k8)
        ttnn.deallocate(v8)

        gated = ttnn.multiply(attn, ttnn.sigmoid(gate))  # [1,NH,S,HD]
        ttnn.deallocate(attn)
        ttnn.deallocate(gate)
        # [1,NH,S,HD] -> [1,S,NH,HD] -> [1,1,S,NH*HD]
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

        # QK RMSNorm — raw (no-+1) at DECODE only (hybrid scaling): sharp +1 prefill retrieves the
        # long context; flat decode averages over keys so the per-step decode noise cannot flip
        # retrieval (the loop/junk failure mode).
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6), tw["q_norm_flat"])
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6), tw["k_norm_flat"])

        q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)

        # Cap the SDPA-decode grid to 64 cores (tree-reduction limit); auto-grid
        # grabs all 110 P150 cores for a single user (B=1) and overflows.
        sdpa_dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
        if use_paged:
            # External paged KV cache (vLLM/contract path): update at cur_pos via the
            # page_table, then paged SDPA-decode
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
            # Internal per-head KV caches (demo/standalone). Pad NKV head dim to 32 for
            # the tile-aligned sharded update.
            for h in range(NKV):
                k_h = ttnn.slice(k, (0, 0, h, 0), (1, B, h + 1, HD))
                v_h = ttnn.slice(v, (0, 0, h, 0), (1, B, h + 1, HD))
                k_hp = ttnn.pad(k_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                v_hp = ttnn.pad(v_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                ttnn.deallocate(k_h)
                ttnn.deallocate(v_h)
                k_sh = ttnn.to_memory_config(k_hp, self.args.kv_update_shard_cfg)
                v_sh = ttnn.to_memory_config(v_hp, self.args.kv_update_shard_cfg)
                ttnn.deallocate(k_hp)
                ttnn.deallocate(v_hp)
                ttnn.experimental.paged_update_cache(self.k_caches[h], k_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.experimental.paged_update_cache(self.v_caches[h], v_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.deallocate(k_sh)
                ttnn.deallocate(v_sh)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            if NKV == 1:
                k_full, v_full = self.k_caches[0], self.v_caches[0]
            else:
                k_full = ttnn.concat(self.k_caches, dim=1)
                v_full = ttnn.concat(self.v_caches, dim=1)

            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                k_full,
                v_full,
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

        # Chunked SDPA over the paged cache (attends to prior chunks via page_table). Keep Q in bf16
        # (see the module docstring: bf8-Q was the long-context degeneration cause). QWEN_SDPA_BF8_Q=1
        # restores the old bf8 cast. When bf16, q IS q8 (freed once at deallocate(q8)).
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
