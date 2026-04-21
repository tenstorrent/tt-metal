# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Qwen3.5 Full Attention layer.

Differences from standard tt_transformers Attention:
1. Partial RoPE: 64 of 256 head dims
2. QK L2 norms (not RMSNorm): x / (||x||_2 + eps) * learned_scale
3. Sigmoid output gating: out = attn_output * sigmoid(gate)
4. Fused Q+gate projection: wqkv projects to [Q, gate] interleaved
5. Separate K/V projections (not fused QKV)
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.qwen35_27b.tt.model_config import create_prefill_matmul_program_config
from models.demos.qwen35_27b.tt.rope import apply_partial_rope_decode, apply_partial_rope_prefill, get_prefill_rot_mats
from models.tt_transformers.tt.ccl import tt_all_reduce


def _shard_linear(x_tt, weight, act_shard_cfg, prog_cfg, compute_cfg):
    x_sharded = ttnn.to_memory_config(x_tt, act_shard_cfg)
    return ttnn.linear(
        x_sharded,
        weight,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        program_config=prog_cfg,
        compute_kernel_config=compute_cfg,
    )


def _unshard(t):
    if t.memory_config().memory_layout != ttnn.TensorMemoryLayout.INTERLEAVED:
        return ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG)
    return t


def _rms_norm_dev(x, eps=1e-6):
    """RMSNorm along last dim: x / sqrt(mean(x^2) + eps). Used for QK norms."""
    return ttnn.rms_norm(x, epsilon=eps)


class Qwen35Attention(LightweightModule):
    """Qwen3.5 full attention with partial RoPE, L2 QK norms, and sigmoid gating."""

    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.args = args
        self.layer_num = layer_num
        self.dtype = dtype

        self.batch_size = args.max_batch_size
        self.max_seq_len = args.max_seq_len
        self.head_dim = args.head_dim
        self.n_local_heads = args.n_local_heads
        self.n_local_kv_heads = getattr(args, "n_local_kv_heads", args.n_kv_heads // args.num_devices)
        self.scale = self.head_dim**-0.5
        self.is_sliding = False

        self.compute_cfg = args.compute_kernel_config_hifi2

        # KV cache
        self.layer_past = None
        self.k_caches = None
        self.v_caches = None
        self._kv_update_shard_cfg = args.kv_update_shard_cfg

        # Weights set later via set_weights()
        self.tw = self._load_weights(state_dict, layer_num, mesh_device, weight_cache_path)

    def _load_weights(self, state_dict, layer_num, mesh_device, weight_cache_path):
        if isinstance(state_dict, dict) and "wqkv" in state_dict:
            return state_dict
        return {}

    def set_weights(self, layer_weights):
        self.tw = layer_weights

    def reset_state(self):
        B = self.batch_size
        mesh = self.mesh_device

        def _to_mesh(t):
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.k_caches = [
            _to_mesh(torch.zeros(B, 1, self.max_seq_len, self.head_dim, dtype=torch.bfloat16))
            for _ in range(self.n_local_kv_heads)
        ]
        self.v_caches = [
            _to_mesh(torch.zeros(B, 1, self.max_seq_len, self.head_dim, dtype=torch.bfloat16))
            for _ in range(self.n_local_kv_heads)
        ]

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        is_prefill = mode == "prefill" or (hasattr(mode, "value") and mode.value == "prefill")
        if is_prefill:
            if kv_cache is not None:
                return self.forward_prefill_paged(
                    x,
                    rot_mats=rot_mats,
                    page_table=page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx=chunk_start_idx,
                    kv_cache=kv_cache,
                    user_id=user_id,
                )
            return self.forward_prefill(x, current_pos, rot_mats)
        return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)

    def forward_decode(self, x, cur_pos_tt, rot_mats, page_table=None, kv_cache=None):
        tw = self.tw
        B = self.batch_size
        NH = self.n_local_heads
        NKV = self.n_local_kv_heads
        HD = self.head_dim
        act_shard = self.args.act_shard_hidden

        use_paged = kv_cache is not None
        if not use_paged and self.k_caches is None:
            self.reset_state()

        cos_tt, sin_tt = rot_mats if isinstance(rot_mats, (list, tuple)) else (rot_mats, None)

        # Projections (DRAM-sharded matmuls)
        qg_tt = _unshard(_shard_linear(x, tw["wqkv"], act_shard, self.args.attn_qg_progcfg, self.compute_cfg))
        kp_tt = _unshard(_shard_linear(x, tw["wk"], act_shard, self.args.attn_k_progcfg, self.compute_cfg))
        vp_tt = _unshard(_shard_linear(x, tw["wv"], act_shard, self.args.attn_v_progcfg, self.compute_cfg))

        # Reshape to [1, B, heads, hd]
        qg_r = ttnn.reshape(qg_tt, (1, B, NH, HD * 2))
        ttnn.deallocate(qg_tt)
        q = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD))
        gate = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, HD * 2))
        ttnn.deallocate(qg_r)

        k = ttnn.reshape(kp_tt, (1, B, NKV, HD))
        ttnn.deallocate(kp_tt)
        v = ttnn.reshape(vp_tt, (1, B, NKV, HD))
        ttnn.deallocate(vp_tt)

        # QK RMSNorm with learned scale
        q = ttnn.multiply(_rms_norm_dev(q), tw["q_norm"])
        k = ttnn.multiply(_rms_norm_dev(k), tw["k_norm"])

        # Partial RoPE (64 dims)
        if cos_tt is not None and sin_tt is not None:
            q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B)
            k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B)

        # KV cache update + SDPA
        if use_paged:
            # External paged KV cache from vLLM
            keys = kv_cache[0]
            values = kv_cache[1]

            # paged_update_cache requires a sharded input tensor. Mirror the non-paged
            # branch: pad the head dim from NKV (1 on 4x P150) to 32 for tile alignment,
            # then convert to the L1-sharded layout the op expects.
            k_padded = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
            v_padded = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
            k_sh = ttnn.to_memory_config(k_padded, self._kv_update_shard_cfg)
            v_sh = ttnn.to_memory_config(v_padded, self._kv_update_shard_cfg)
            ttnn.deallocate(k_padded)
            ttnn.deallocate(v_padded)

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
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            # Internal per-head KV caches (demo/standalone mode)
            for h in range(NKV):
                k_h = ttnn.slice(k, (0, 0, h, 0), (1, B, h + 1, HD))
                v_h = ttnn.slice(v, (0, 0, h, 0), (1, B, h + 1, HD))
                k_h_padded = ttnn.pad(k_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                v_h_padded = ttnn.pad(v_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                ttnn.deallocate(k_h)
                ttnn.deallocate(v_h)
                k_sh = ttnn.to_memory_config(k_h_padded, self._kv_update_shard_cfg)
                v_sh = ttnn.to_memory_config(v_h_padded, self._kv_update_shard_cfg)
                ttnn.deallocate(k_h_padded)
                ttnn.deallocate(v_h_padded)
                ttnn.experimental.paged_update_cache(self.k_caches[h], k_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.experimental.paged_update_cache(self.v_caches[h], v_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.deallocate(k_sh)
                ttnn.deallocate(v_sh)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            if NKV == 1:
                k_full, v_full = self.k_caches[0], self.v_caches[0]
            else:
                k_full = ttnn.concat([self.k_caches[h] for h in range(NKV)], dim=1)
                v_full = ttnn.concat([self.v_caches[h] for h in range(NKV)], dim=1)

            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                k_full,
                v_full,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        ttnn.deallocate(q)

        # Sigmoid gating
        gate_val = ttnn.sigmoid(gate)
        ttnn.deallocate(gate)
        gated = ttnn.multiply(attn_out, gate_val)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate_val)

        # Output projection + all-reduce
        gated_flat = ttnn.reshape(gated, (1, B, NH * HD))
        ttnn.deallocate(gated)
        act_shard_out = self.args.act_shard_attn_out
        wo_partial = _unshard(
            _shard_linear(gated_flat, tw["wo"], act_shard_out, self.args.attn_wo_progcfg, self.compute_cfg)
        )
        ttnn.deallocate(gated_flat)

        wo_partial = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))
        return self._all_reduce(wo_partial)

    def replicate_kv_cache_to_batch(self):
        """Copy user 0's KV cache entries to all batch_size slots.

        After prefilling 1 user, this replicates the KV cache so all 32 decode
        users start with the same cached keys/values.

        Each device holds a different KV head shard, so we replicate per-device
        (not across devices).
        """
        if self.k_caches is None:
            return
        B = self.batch_size
        mesh = self.mesh_device

        for h in range(self.n_local_kv_heads):
            # Get per-device tensors: each has shape [B, 1, max_seq_len, HD]
            k_per_device = ttnn.get_device_tensors(self.k_caches[h])
            v_per_device = ttnn.get_device_tensors(self.v_caches[h])

            k_torch_per_dev = []
            v_torch_per_dev = []
            for k_dev, v_dev in zip(k_per_device, v_per_device):
                k_cpu = ttnn.to_torch(k_dev)  # [B, 1, max_seq_len, HD]
                v_cpu = ttnn.to_torch(v_dev)
                # Take user 0 and replicate to all B slots
                k_user0 = k_cpu[0:1]  # [1, 1, max_seq_len, HD]
                v_user0 = v_cpu[0:1]
                k_torch_per_dev.append(k_user0.expand(B, -1, -1, -1).contiguous())
                v_torch_per_dev.append(v_user0.expand(B, -1, -1, -1).contiguous())

            # Write back: each device gets its own replicated data
            self.k_caches[h] = ttnn.from_torch(
                torch.cat(k_torch_per_dev, dim=0),  # [num_devices*B, 1, max_seq_len, HD]
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )
            self.v_caches[h] = ttnn.from_torch(
                torch.cat(v_torch_per_dev, dim=0),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )

    def _all_reduce(self, partial_mesh):
        return tt_all_reduce(
            partial_mesh,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward_prefill(self, x, current_pos, rot_mats):
        """Batched prefill with flash attention.

        Processes full sequence in parallel:
        1. QKV projections via 2D matmul (compute-bound)
        2. Partial RoPE for all positions
        3. KV cache fill
        4. Flash attention (ttnn.transformer.scaled_dot_product_attention, is_causal=True)
        5. Sigmoid gating + output projection
        """
        tw = self.tw
        NH = self.n_local_heads
        NKV = self.n_local_kv_heads
        HD = self.head_dim
        dim = self.args.dim

        # Ensure 4D: [1, 1, seq_len, dim]
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (1, 1, x.shape[1], x.shape[2]))
        seq_len = x.shape[2]

        if self.k_caches is None:
            self.reset_state()

        # Get prefill RoPE cos/sin: [1, 1, seq_len, 64]
        rope_setup = None
        if isinstance(rot_mats, (list, tuple)):
            # rot_mats are decode-mode cos/sin; we need prefill tables from rope_setup
            # Fall back to decode-per-token if we can't get prefill tables
            pass

        # ---- QKV Projections (2D matmul, DRAM interleaved) ----
        x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        qg_progcfg = create_prefill_matmul_program_config(seq_len, dim, NH * HD * 2)
        qg_tt = ttnn.linear(
            x_dram,
            tw["wqkv"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qg_progcfg,
            compute_kernel_config=self.compute_cfg,
        )

        k_progcfg = create_prefill_matmul_program_config(seq_len, dim, NKV * HD)
        kp_tt = ttnn.linear(
            x_dram,
            tw["wk"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=k_progcfg,
            compute_kernel_config=self.compute_cfg,
        )

        vp_tt = ttnn.linear(
            x_dram,
            tw["wv"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=k_progcfg,
            compute_kernel_config=self.compute_cfg,
        )
        ttnn.deallocate(x_dram)

        # ---- Reshape to head format for SDPA ----
        # Structural reshapes (splitting heads from flat dims) done on CPU because
        # ttnn.reshape in TILE_LAYOUT does not follow PyTorch semantics — a direct
        # reshape from [1,1,S,NH*HD] to [1,NH,S,HD] interleaves heads and tokens.
        # The correct operation is view(S,NH,HD).permute(1,0,2) → [NH,S,HD].
        mesh = self.mesh_device
        num_devices = mesh.get_num_devices()

        # K/V: [1, 1, seq_len, NKV*HD] → [1, NKV, seq_len, HD]
        kp_cpu = ttnn.to_torch(kp_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        vp_cpu = ttnn.to_torch(vp_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        ttnn.deallocate(kp_tt)
        ttnn.deallocate(vp_tt)

        k_parts, v_parts = [], []
        for d in range(num_devices):
            kd = kp_cpu[d].squeeze(0).float().reshape(seq_len, NKV, HD).permute(1, 0, 2)  # [NKV, S, HD]
            vd = vp_cpu[d].squeeze(0).float().reshape(seq_len, NKV, HD).permute(1, 0, 2)
            k_parts.append(kd.unsqueeze(0))  # [1, NKV, S, HD]
            v_parts.append(vd.unsqueeze(0))
        k_stacked = torch.cat(k_parts, dim=0).to(torch.bfloat16).contiguous()
        v_stacked = torch.cat(v_parts, dim=0).to(torch.bfloat16).contiguous()
        del kp_cpu, vp_cpu, k_parts, v_parts

        k = ttnn.from_torch(
            k_stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        v = ttnn.from_torch(
            v_stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        del k_stacked, v_stacked

        # Q+gate: [1, 1, seq_len, NH*HD*2] → split into q [1, NH, S, HD] and gate [1, NH, S, HD]
        qg_cpu = ttnn.to_torch(qg_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        ttnn.deallocate(qg_tt)

        q_parts, gate_parts = [], []
        for d in range(num_devices):
            qgd = qg_cpu[d].squeeze(0).float().reshape(seq_len, NH, HD * 2).permute(1, 0, 2)  # [NH, S, HD*2]
            q_parts.append(qgd[:, :, :HD].unsqueeze(0))  # [1, NH, S, HD]
            gate_parts.append(qgd[:, :, HD:].unsqueeze(0))  # [1, NH, S, HD]
        q_stacked = torch.cat(q_parts, dim=0).to(torch.bfloat16).contiguous()
        gate_stacked = torch.cat(gate_parts, dim=0).to(torch.bfloat16).contiguous()
        del qg_cpu, q_parts, gate_parts

        q = ttnn.from_torch(
            q_stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        gate = ttnn.from_torch(
            gate_stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        del q_stacked, gate_stacked

        # ---- QK L2 norm with learned scale ----
        q = ttnn.multiply(_rms_norm_dev(q), tw["q_norm"])
        k = ttnn.multiply(_rms_norm_dev(k), tw["k_norm"])

        # ---- Partial RoPE for all positions ----
        # Get cos/sin tables: [1, 1, seq_len, 64]
        cos_tt, sin_tt = (
            get_prefill_rot_mats(self.args._rope_setup_ref, seq_len, self.mesh_device)
            if hasattr(self.args, "_rope_setup_ref")
            else (None, None)
        )

        if cos_tt is None:
            # Build cos/sin from scratch for prefill
            from models.demos.qwen35_27b.tt.model_config import ROPE_DIM

            inv_freq = 1.0 / (self.args.rope_theta ** (torch.arange(0, ROPE_DIM, 2).float() / ROPE_DIM))
            t = torch.arange(seq_len, dtype=torch.float32)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos_tt = ttnn.from_torch(
                emb.cos().unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            sin_tt = ttnn.from_torch(
                emb.sin().unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV)
        ttnn.deallocate(cos_tt)
        ttnn.deallocate(sin_tt)

        # ---- Fill KV cache ----
        # Don't deallocate slices — they share buffer with k/v which we need later for SDPA
        for h in range(NKV):
            k_h = ttnn.slice(k, (0, h, 0, 0), (1, h + 1, seq_len, HD))
            v_h = ttnn.slice(v, (0, h, 0, 0), (1, h + 1, seq_len, HD))
            ttnn.fill_cache(self.k_caches[h], k_h, 0)
            ttnn.fill_cache(self.v_caches[h], v_h, 0)

        # ---- Flash Attention (causal) ----
        # SDPA expects bfloat8_b inputs and requires program config
        q_8b = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
        k_8b = ttnn.typecast(k, dtype=ttnn.bfloat8_b)
        v_8b = ttnn.typecast(v, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # Chunk sizes must be power of 2, multiple of 32, and <= seq_len (padded to tile)
        padded_seq = max(32, ((seq_len + 31) // 32) * 32)
        q_chunk = min(256 if seq_len >= 2048 else 64, padded_seq)
        k_chunk = min(256 if seq_len >= 2048 else 64, padded_seq)
        sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

        attn_out = ttnn.transformer.scaled_dot_product_attention(
            q_8b,
            k_8b,
            v_8b,
            is_causal=True,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=sdpa_config,
        )
        ttnn.deallocate(q_8b)
        ttnn.deallocate(k_8b)
        ttnn.deallocate(v_8b)

        # ---- Sigmoid gating ----
        gate_val = ttnn.sigmoid(gate)
        ttnn.deallocate(gate)
        gated = ttnn.multiply(attn_out, gate_val)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate_val)

        # ---- Output projection ----
        # [1, NH, seq_len, HD] -> [1, 1, seq_len, NH*HD]
        # Must use CPU reshape+permute (inverse of head-splitting) for correctness.
        gated_cpu = ttnn.to_torch(gated, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        ttnn.deallocate(gated)
        flat_parts = []
        for d in range(num_devices):
            gd = gated_cpu[d].float()  # [NH, seq_len, HD]
            flat_parts.append(gd.permute(1, 0, 2).reshape(1, seq_len, NH * HD).unsqueeze(0))  # [1, 1, S, NH*HD]
        gated_flat_cpu = torch.cat(flat_parts, dim=0).to(torch.bfloat16).contiguous()
        del gated_cpu, flat_parts
        gated_flat = ttnn.from_torch(
            gated_flat_cpu,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        del gated_flat_cpu

        wo_progcfg = create_prefill_matmul_program_config(seq_len, NH * HD, dim)
        wo_out = ttnn.linear(
            gated_flat,
            tw["wo"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=wo_progcfg,
            compute_kernel_config=self.compute_cfg,
        )
        ttnn.deallocate(gated_flat)

        # ---- All-reduce ----
        return self._all_reduce(wo_out)

    def forward_prefill_paged(
        self,
        x,
        *,
        rot_mats,
        page_table,
        chunk_page_table,
        chunk_start_idx,
        kv_cache,
        user_id=0,
    ):
        """Prefill with paged KV cache for one chunk of a long sequence.

        Called once per attention chunk from model.prefill_layer_chunked. The
        caller provides:
          - x: chunk input [1, 1, chunk_len, dim]
          - rot_mats: [cos_chunk, sin_chunk], each sliced to the chunk's token range
          - page_table: FULL page table (ttnn int32, ROW_MAJOR), passed to
            chunked_scaled_dot_product_attention so it can attend to prior chunks.
          - chunk_page_table: slice of page_table covering only this chunk's blocks;
            passed to paged_fill_cache so it writes only this chunk's positions.
          - chunk_start_idx: absolute token offset of this chunk within the sequence.
          - kv_cache: (k_paged, v_paged) tuple allocated by allocate_paged_kv_caches;
            both shape (max_num_blocks, n_local_kv_heads, block_size, head_dim).
          - user_id: int batch row index in the page table.
        """
        assert kv_cache is not None, "forward_prefill_paged requires kv_cache"
        assert page_table is not None, "forward_prefill_paged requires page_table"
        assert chunk_start_idx is not None, "forward_prefill_paged requires chunk_start_idx"
        assert rot_mats is not None and len(rot_mats) == 2, "forward_prefill_paged requires rot_mats=[cos, sin]"

        tw = self.tw
        NH = self.n_local_heads
        NKV = self.n_local_kv_heads
        HD = self.head_dim
        dim = self.args.dim

        # Ensure 4D: [1, 1, chunk_len, dim]
        if len(x.shape) == 3:
            x = ttnn.reshape(x, (1, 1, x.shape[1], x.shape[2]))
        seq_len = x.shape[2]

        # ---- QKV Projections (verbatim from forward_prefill) ----
        x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        qg_progcfg = create_prefill_matmul_program_config(seq_len, dim, NH * HD * 2)
        qg_tt = ttnn.linear(
            x_dram,
            tw["wqkv"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=qg_progcfg,
            compute_kernel_config=self.compute_cfg,
        )

        k_progcfg = create_prefill_matmul_program_config(seq_len, dim, NKV * HD)
        kp_tt = ttnn.linear(
            x_dram,
            tw["wk"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=k_progcfg,
            compute_kernel_config=self.compute_cfg,
        )
        vp_tt = ttnn.linear(
            x_dram,
            tw["wv"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=k_progcfg,
            compute_kernel_config=self.compute_cfg,
        )
        ttnn.deallocate(x_dram)

        # ---- CPU reshape to head-major (verbatim structure from forward_prefill) ----
        mesh = self.mesh_device
        num_devices = mesh.get_num_devices()

        kp_cpu = ttnn.to_torch(kp_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        vp_cpu = ttnn.to_torch(vp_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        ttnn.deallocate(kp_tt)
        ttnn.deallocate(vp_tt)

        k_parts, v_parts = [], []
        for d in range(num_devices):
            kd = kp_cpu[d].squeeze(0).float().reshape(seq_len, NKV, HD).permute(1, 0, 2)
            vd = vp_cpu[d].squeeze(0).float().reshape(seq_len, NKV, HD).permute(1, 0, 2)
            k_parts.append(kd.unsqueeze(0))
            v_parts.append(vd.unsqueeze(0))
        k_stacked = torch.cat(k_parts, dim=0).to(torch.bfloat16).contiguous()
        v_stacked = torch.cat(v_parts, dim=0).to(torch.bfloat16).contiguous()
        del kp_cpu, vp_cpu, k_parts, v_parts

        k = ttnn.from_torch(
            k_stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        v = ttnn.from_torch(
            v_stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        del k_stacked, v_stacked

        qg_cpu = ttnn.to_torch(qg_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        ttnn.deallocate(qg_tt)

        q_parts, gate_parts = [], []
        for d in range(num_devices):
            qgd = qg_cpu[d].squeeze(0).float().reshape(seq_len, NH, HD * 2).permute(1, 0, 2)
            q_parts.append(qgd[:, :, :HD].unsqueeze(0))
            gate_parts.append(qgd[:, :, HD:].unsqueeze(0))
        q_stacked = torch.cat(q_parts, dim=0).to(torch.bfloat16).contiguous()
        gate_stacked = torch.cat(gate_parts, dim=0).to(torch.bfloat16).contiguous()
        del qg_cpu, q_parts, gate_parts

        q = ttnn.from_torch(
            q_stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        gate = ttnn.from_torch(
            gate_stacked,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        del q_stacked, gate_stacked

        # ---- QK L2 norm with learned scale ----
        q = ttnn.multiply(_rms_norm_dev(q), tw["q_norm"])
        k = ttnn.multiply(_rms_norm_dev(k), tw["k_norm"])

        # ---- Partial RoPE using caller-provided chunk-sliced cos/sin ----
        cos_tt, sin_tt = rot_mats[0], rot_mats[1]
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV)

        # ---- Paged KV cache fill ----
        k_paged, v_paged = kv_cache
        block_size = k_paged.shape[2]
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[1] * block_size

        # Slice K/V down to page_len if this chunk overhangs its allocated blocks.
        if page_len < seq_len:
            k_fill = ttnn.slice(k, (0, 0, 0, 0), (1, NKV, page_len, HD))
            v_fill = ttnn.slice(v, (0, 0, 0, 0), (1, NKV, page_len, HD))
        else:
            k_fill = k
            v_fill = v

        ttnn.experimental.paged_fill_cache(k_paged, k_fill, fill_page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_paged, v_fill, fill_page_table, batch_idx=user_id)

        if page_len < seq_len:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)
        ttnn.deallocate(k)
        ttnn.deallocate(v)

        # ---- Chunked SDPA over paged cache ----
        # bfloat8_b cast to match the dtype used by tt_transformers' chunked SDPA path.
        q_8b = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q)

        # SDPA has a hard requirement: chunk_start_idx % q_chunk_size == 0.
        # For chunk_start_idx=0 any q_chunk divides; for chunk_start_idx > 0
        # the highest power of 2 dividing it is `chunk_start_idx & -chunk_start_idx`.
        # Formulation lifted from tt_transformers/tt/model_config.py:1452-1470 so the
        # rule holds at every chunk boundary across 8k-128k ISL. (Also tracks the
        # k_chunk workaround for tt-metal#35225.)
        def _pick_chunk_size(seq_len_inner, chunk_start):
            if chunk_start is None or chunk_start == 0:
                return 256 if seq_len_inner >= 2048 else 64
            cap = 256 if seq_len_inner >= 2048 else 64
            return min(cap, chunk_start & -chunk_start)

        q_chunk = _pick_chunk_size(seq_len, chunk_start_idx)
        k_chunk = _pick_chunk_size(seq_len, chunk_start_idx)

        # Blackhole P150 has a 13x10 (130-core) worker grid vs WH's 8x8. Pulling the
        # grid from the device avoids the WH-era 64-core cap that would pin SDPA at
        # ~49% utilization on BH. Existing helper idiom (see rope.py:472).
        sdpa_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            exp_approx_mode=False,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
        )

        attn_out = ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=q_8b,
            input_tensor_k=k_paged,
            input_tensor_v=v_paged,
            page_table_tensor=page_table,
            chunk_start_idx=chunk_start_idx,
            compute_kernel_config=self.compute_cfg,
            program_config=sdpa_config,
        )
        ttnn.deallocate(q_8b)

        # ---- Sigmoid gating (verbatim) ----
        gate_val = ttnn.sigmoid(gate)
        ttnn.deallocate(gate)
        gated = ttnn.multiply(attn_out, gate_val)
        ttnn.deallocate(attn_out)
        ttnn.deallocate(gate_val)

        # ---- Output projection + all-reduce (verbatim from forward_prefill) ----
        gated_cpu = ttnn.to_torch(gated, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
        ttnn.deallocate(gated)
        flat_parts = []
        for d in range(num_devices):
            gd = gated_cpu[d].float()
            flat_parts.append(gd.permute(1, 0, 2).reshape(1, seq_len, NH * HD).unsqueeze(0))
        gated_flat_cpu = torch.cat(flat_parts, dim=0).to(torch.bfloat16).contiguous()
        del gated_cpu, flat_parts
        gated_flat = ttnn.from_torch(
            gated_flat_cpu,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
        )
        del gated_flat_cpu

        wo_progcfg = create_prefill_matmul_program_config(seq_len, NH * HD, dim)
        wo_out = ttnn.linear(
            gated_flat,
            tw["wo"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=wo_progcfg,
            compute_kernel_config=self.compute_cfg,
        )
        ttnn.deallocate(gated_flat)

        return self._all_reduce(wo_out)
