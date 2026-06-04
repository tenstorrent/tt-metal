# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel (TP>1) full-attention path for Qwen3.5.

Ported from models/demos/qwen35_27b/tt/attention.py forward_decode, with two
corrections vs that reference:
  - q_norm/k_norm are zero-centered (HF Qwen3_5RMSNorm = output*(1+weight)); the
    27B omitted the +1.
  - weights are kept INTERLEAVED per device (no DRAM-width-sharding) and matmuls
    use ttnn's auto program config — same robust pattern validated for the MLP.

Decode input/output use the framework layout: x [1,1,B,dim] replicated in; output
fractured along dim=3 (reduce-scatter). Column-parallel q/k/v, row-parallel wo.
"""
import torch

import ttnn
from models.demos.blackhole.qwen3_5_9b.tt import tp_common as tpc
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.tt_transformers.tt.ccl import tt_all_reduce


def load_attention_weights_tp(mesh, state_dict, args, cache_dir=None):
    """Shard one full-attention layer's weights across the mesh.

    state_dict keys (from the FP8 loader / 9B substate): q_proj/k_proj/v_proj/
    o_proj/q_norm/k_norm. q_proj is the fused per-head [Q,gate] projection.
    """
    if cache_dir is not None:
        import os

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
    # Zero-centered (+1) per-head QK norms, replicated
    tw["q_norm"] = tpc.replicate(state_dict["q_norm.weight"].to(torch.float32) + 1.0, mesh, c("q_norm"))
    tw["k_norm"] = tpc.replicate(state_dict["k_norm.weight"].to(torch.float32) + 1.0, mesh, c("k_norm"))
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

        q8 = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
        k8 = ttnn.typecast(k, dtype=ttnn.bfloat8_b)
        v8 = ttnn.typecast(v, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
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

    def forward_decode(self, x, cur_pos_tt, cos_tt, sin_tt):
        tw, B, NH, NKV, HD = self.tw, self.B, self.NH, self.NKV, self.HD
        if self.k_caches is None:
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

        # zero-centered QK RMSNorm (weight already has +1 baked in)
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6), tw["q_norm"])
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6), tw["k_norm"])

        q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
        k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)

        # Update per-head KV caches (pad NKV head dim to 32 for tile-aligned sharded update)
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

        # Cap the SDPA-decode grid to 64 cores (tree-reduction limit); auto-grid
        # grabs all 110 P150 cores for a single user (B=1) and overflows.
        sdpa_dec_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=0
        )
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
