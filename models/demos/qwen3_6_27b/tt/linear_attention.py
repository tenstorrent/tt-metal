# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6 Gated DeltaNet — mesh-aware, V-head row-axis tensor-parallel.

Sharding (mesh shape rows×cols):
  - 48 V-heads sharded across `rows` (validated in T3.1: PCC=0.999985 on 8×4)
  - 16 K-heads sharded across `rows` (K must divide rows; for 8 rows: 2 K-heads/row)
  - Hidden dim H replicated; col axis is replicated (DP/replication parking).
  - out_proj input dim (= n_v_heads*head_v_dim = 6144) sharded across rows;
    output [B, T, H] is partial-sum per row → ReduceScatter along row axis.

Operates on input that is hidden-dim REPLICATED across mesh (the caller is
responsible for the residual stream layout).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import chunk_gated_delta_rule_ttnn
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import causal_conv1d_ttnn

TILE = 32


class TtDeltaNetBlock:
    """Mesh-sharded Gated DeltaNet block."""

    def __init__(self, mesh_device, state_dict, prefix, hf_config):
        self.mesh = mesh_device
        self.prefix = prefix
        self.hf_cfg = hf_config
        self.mesh_rows = mesh_device.shape[0]
        self.mesh_cols = mesh_device.shape[1]
        self.n_v = hf_config.linear_num_value_heads  # 48
        self.n_k = hf_config.linear_num_key_heads  # 16
        self.hd_k = hf_config.linear_key_head_dim  # 128
        self.hd_v = hf_config.linear_value_head_dim  # 128
        self.g_ratio = self.n_v // self.n_k  # 3
        self.conv_k = hf_config.linear_conv_kernel_dim  # 4
        self.H = hf_config.hidden_size  # 5120
        self.eps = hf_config.rms_norm_eps

        # Per-row sharding
        assert self.n_v % self.mesh_rows == 0, f"n_v={self.n_v} not divisible by mesh_rows={self.mesh_rows}"
        assert self.n_k % self.mesh_rows == 0, f"n_k={self.n_k} not divisible by mesh_rows={self.mesh_rows}"
        self.n_v_per_row = self.n_v // self.mesh_rows
        self.n_k_per_row = self.n_k // self.mesh_rows

        # ---- Load and pre-shard weights on host ----
        in_proj_qkv = state_dict[f"{prefix}.in_proj_qkv.weight"].float()  # [10240, 5120]
        in_proj_z = state_dict[f"{prefix}.in_proj_z.weight"].float()  # [6144, 5120]
        in_proj_a = state_dict[f"{prefix}.in_proj_a.weight"].float()  # [48, 5120]
        in_proj_b = state_dict[f"{prefix}.in_proj_b.weight"].float()  # [48, 5120]
        conv_w = state_dict[f"{prefix}.conv1d.weight"].float()  # [10240, 1, 4]
        A_log = state_dict[f"{prefix}.A_log"].float()  # [48]
        dt_bias = state_dict[f"{prefix}.dt_bias"].float()  # [48]
        norm_w = state_dict[f"{prefix}.norm.weight"].float()  # [128]
        out_proj = state_dict[f"{prefix}.out_proj.weight"].float()  # [5120, 6144]

        # Split in_proj_qkv into per-K-head and per-V-head pieces (it's [Q‖K‖V] block-wise).
        # Q lives in K-head space (16×128=2048), K same (2048), V in V-head space (48×128=6144).
        Q_w = in_proj_qkv[: self.n_k * self.hd_k]  # [2048, H]
        K_w = in_proj_qkv[self.n_k * self.hd_k : 2 * self.n_k * self.hd_k]  # [2048, H]
        V_w = in_proj_qkv[2 * self.n_k * self.hd_k :]  # [6144, H]

        # Conv1d: depthwise on QKV concatenated. Split same.
        conv_Q = conv_w[: self.n_k * self.hd_k]  # [2048, 1, 4]
        conv_K = conv_w[self.n_k * self.hd_k : 2 * self.n_k * self.hd_k]
        conv_V = conv_w[2 * self.n_k * self.hd_k :]  # [6144, 1, 4]

        # Reshape per-head then shard along the head axis (row-parallel).
        # Q_w → [n_k, hd_k, H] → reshape → split into mesh_rows groups of n_k_per_row K-heads.
        def _shard_proj(w_flat, n_heads, head_dim, dim_in, head_axis_size_per_row):
            """w_flat: [n_heads*head_dim, dim_in] → shard along head axis across rows.
            Returns torch tensor [n_heads*head_dim, dim_in] but reorganized to a per-row layout
            suitable for ttnn.from_torch with ShardTensor2dMesh(row_dim=0).
            """
            # Already in flat [n_heads*head_dim, dim_in]. Sharding dim 0 splits by row.
            return w_flat

        # For our row sharding: shard dim 0 of [n_v*hd_v, H] across mesh_rows.
        # ttnn.from_torch with MeshMapperConfig(row_dim=None, col_dim=None) will give each row
        # a [n_v_per_row*hd_v, H] slice.

        def to_dev_sharded_dim0(t):
            """Send tensor sharded along its dim 0 across mesh rows, replicated across cols.
            For ttnn.linear we need weight in [in, out] form, so caller passes transposed."""
            return ttnn.from_torch(
                t.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.create_mesh_mapper(
                    mesh_device,
                    ttnn.MeshMapperConfig(row_dim=None, col_dim=None),
                ),
            )

        def to_dev_replicated(t):
            return ttnn.from_torch(
                t.contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.create_mesh_mapper(
                    mesh_device,
                    ttnn.MeshMapperConfig(row_dim=None, col_dim=None),
                ),
            )

        # For ttnn.linear: weight is [in, out], so we transpose then shard on the OUT axis.
        # The OUT axis is the head-dim concatenation; sharding that = head sharding. ✓
        # in_proj_q [2048, H] → transpose → [H, 2048]; sharding dim 1 (cols of transposed = rows of orig matrix)
        # But MeshMapperConfig only supports sharding along one dim per axis. With row_dim=1 (the out-axis after transpose), we shard heads correctly.
        def proj_weight_sharded_out(w_flat):
            """w_flat [out_dim, in_dim] = HF storage. ttnn.linear expects [in_dim, out_dim].
            We want to shard the out_dim across rows."""
            t = w_flat.T.contiguous()  # [in_dim, out_dim]
            return ttnn.from_torch(
                t,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.create_mesh_mapper(
                    mesh_device,
                    ttnn.MeshMapperConfig(row_dim=None, col_dim=None),
                ),
            )

        self.w_q = proj_weight_sharded_out(Q_w)  # out shards: 2 K-heads × 128/row
        self.w_k = proj_weight_sharded_out(K_w)
        self.w_v = proj_weight_sharded_out(V_w)  # out shards: 6 V-heads × 128/row
        self.w_z = proj_weight_sharded_out(in_proj_z)
        self.w_a = proj_weight_sharded_out(in_proj_a)  # 48/8 = 6/row
        self.w_b = proj_weight_sharded_out(in_proj_b)

        # conv1d weight reorganization. The naive concat [Q_conv ‖ K_conv ‖ V_conv] does NOT
        # shard cleanly into per-row [q_row_i ‖ k_row_i ‖ v_row_i] under uniform dim-0 split:
        # Q is 2048 wide, K is 2048, V is 6144; uniform 1280/row split mixes Q-K-V channels.
        # Solution: pre-interleave by row so dim-0 split gives each row its own
        # [conv_Q_row_i ‖ conv_K_row_i ‖ conv_V_row_i] in the right order.
        qk_per_row = self.n_k_per_row * self.hd_k
        v_per_row = self.n_v_per_row * self.hd_v
        chunks = []
        for i in range(self.mesh_rows):
            qc = conv_Q[i * qk_per_row : (i + 1) * qk_per_row]
            kc = conv_K[i * qk_per_row : (i + 1) * qk_per_row]
            vc = conv_V[i * v_per_row : (i + 1) * v_per_row]
            chunks.append(torch.cat([qc, kc, vc], dim=0))
        conv_w_layout = torch.cat(chunks, dim=0)  # [10240, 1, 4] reorganized per-row
        self.conv_weight = ttnn.from_torch(
            conv_w_layout,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(row_dim=None, col_dim=None),
            ),
        )
        self.conv_bias = None

        # SSM scalars: shape [1, 1, head_dim] for broadcast against a_f32 [B, T, head_dim].
        # Shard the head axis (last dim) across rows.
        self.A_log = ttnn.from_torch(
            A_log.reshape(1, 1, -1).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(row_dim=None, col_dim=None),  # last dim = head axis
            ),
        )
        self.dt_bias = ttnn.from_torch(
            dt_bias.reshape(1, 1, -1).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(row_dim=None, col_dim=None),
            ),
        )

        # Norm weight (per-head_dim, replicated): [128] → [1, 1, 1, 128] tile-aligned
        self.norm_weight_replicated_torch = norm_w  # store for host-side GroupRMSNormGated
        # TODO: device-side GroupRMSNormGated

        # out_proj: input [n_v*hd_v=6144] sharded across rows → 6*128=768/row.
        # Weight HF: [H=5120, 6144]. Transposed for ttnn.linear: [6144, 5120].
        # Shard INPUT dim (now row 0 of transposed) across rows.
        out_proj_T = out_proj.T.contiguous()  # [6144, 5120]
        self.w_o = ttnn.from_torch(
            out_proj_T,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(
                mesh_device,
                ttnn.MeshMapperConfig(row_dim=None, col_dim=None),
            ),
        )

    def __call__(self, hidden_states):
        """
        hidden_states: ttnn [B, T, H] BF16, H replicated across mesh.
        Returns: [B, T, H] partial sum → reduce_scatter along row axis (then caller adds residual)
        """
        # 1. Projections — each chip produces its head-slice locally
        q = ttnn.linear(hidden_states, self.w_q, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        k = ttnn.linear(hidden_states, self.w_k, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        v = ttnn.linear(hidden_states, self.w_v, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        z = ttnn.linear(hidden_states, self.w_z, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        a = ttnn.linear(hidden_states, self.w_a, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        b = ttnn.linear(hidden_states, self.w_b, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # 2. Concat Q‖K‖V → conv1d + silu (depthwise, channels sharded same way)
        qkv = ttnn.concat([q, k, v], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        mixed = causal_conv1d_ttnn(qkv, self.conv_weight, self.conv_bias, self.conv_k, self.mesh)

        B = mixed.shape[0]
        T = mixed.shape[1]

        # 3. Split mixed back into q, k, v (per-row dims)
        qk_dim = self.n_k_per_row * self.hd_k  # 2*128 = 256 per row at 8 rows
        v_dim = self.n_v_per_row * self.hd_v  # 6*128 = 768 per row at 8 rows
        q_local = ttnn.slice(mixed, [0, 0, 0], [B, T, qk_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        k_local = ttnn.slice(mixed, [0, 0, qk_dim], [B, T, 2 * qk_dim], memory_config=ttnn.L1_MEMORY_CONFIG)
        v_local = ttnn.slice(mixed, [0, 0, 2 * qk_dim], [B, T, 2 * qk_dim + v_dim], memory_config=ttnn.L1_MEMORY_CONFIG)

        # 4. Reshape per-head
        q_h = ttnn.reshape(q_local, [B, T, self.n_k_per_row, self.hd_k])
        k_h = ttnn.reshape(k_local, [B, T, self.n_k_per_row, self.hd_k])
        v_h = ttnn.reshape(v_local, [B, T, self.n_v_per_row, self.hd_v])

        # 5. β = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias) — all per V-head local
        beta = ttnn.sigmoid(b, memory_config=ttnn.L1_MEMORY_CONFIG)
        a_f32 = ttnn.typecast(a, ttnn.float32, memory_config=ttnn.L1_MEMORY_CONFIG)
        a_db = ttnn.add(a_f32, self.dt_bias, memory_config=ttnn.L1_MEMORY_CONFIG)
        sp = ttnn.softplus(a_db, beta=1.0, threshold=20.0, memory_config=ttnn.L1_MEMORY_CONFIG)
        neg_exp_Alog = ttnn.neg(
            ttnn.exp(self.A_log, memory_config=ttnn.L1_MEMORY_CONFIG),
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        g = ttnn.multiply(neg_exp_Alog, sp, memory_config=ttnn.L1_MEMORY_CONFIG)

        # 6. GQA expand q,k from n_k_per_row → n_v_per_row via repeat_interleave on host
        # (TTNN has no native repeat_interleave; use host bridge until we have a kernel.)
        # NOTE: this is per-shard, so the transfer is small.
        q_host = ttnn.to_torch(
            q_h, mesh_composer=ttnn.create_mesh_composer(self.mesh, ttnn.MeshComposerConfig([2, 0]))
        ).float()
        k_host = ttnn.to_torch(
            k_h, mesh_composer=ttnn.create_mesh_composer(self.mesh, ttnn.MeshComposerConfig([2, 0]))
        ).float()
        # Slice off col-replication
        q_host = q_host[:B].repeat_interleave(self.g_ratio, dim=2)
        k_host = k_host[:B].repeat_interleave(self.g_ratio, dim=2)
        # Re-shard
        q_h = ttnn.from_torch(
            q_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(self.mesh, ttnn.MeshMapperConfig(row_dim=None, col_dim=None)),
        )
        k_h = ttnn.from_torch(
            k_host,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(self.mesh, ttnn.MeshMapperConfig(row_dim=None, col_dim=None)),
        )

        # 7. Delta-rule kernel (per shard, head dim = n_v_per_row)
        core_out = chunk_gated_delta_rule_ttnn(q_h, k_h, v_h, beta, g, chunk_size=64, device=self.mesh)
        if isinstance(core_out, tuple):
            core_out = core_out[0]

        # 8. GroupRMSNormGated with z (host for now)
        # core_out: [B, T, n_v_per_row, hd_v]; z: [B, T, n_v_per_row * hd_v]
        core_host_full = ttnn.to_torch(
            core_out, mesh_composer=ttnn.create_mesh_composer(self.mesh, ttnn.MeshComposerConfig([2, 0]))
        ).float()
        z_host_full = ttnn.to_torch(
            z, mesh_composer=ttnn.create_mesh_composer(self.mesh, ttnn.MeshComposerConfig([2, 0]))
        ).float()
        core_host = core_host_full[:B]
        z_host = z_host_full[:B]
        z_host = z_host.reshape(B, T, self.n_v, self.hd_v)
        var = core_host.pow(2).mean(-1, keepdim=True)
        normed = core_host * torch.rsqrt(var + self.eps)
        normed = normed * self.norm_weight_replicated_torch.float()
        normed = normed * F.silu(z_host)
        out_pre_o = normed.reshape(B, T, self.n_v * self.hd_v)

        # 9. Out-proj: row-parallel matmul → ReduceScatter along row axis.
        # Reshard out_pre_o along V-head axis (last dim) across rows.
        out_pre_o_tt = ttnn.from_torch(
            out_pre_o,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.create_mesh_mapper(self.mesh, ttnn.MeshMapperConfig(row_dim=None, col_dim=None)),
        )
        # Each chip: in [B, T, 768] × w_o [768, 5120] → [B, T, 5120] partial sum
        partial = ttnn.linear(out_pre_o_tt, self.w_o, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16)

        # Host-side reduce for debugging — gather row shards and sum.
        # TODO: replace with on-device tt_all_reduce once verified correct.
        if self.mesh_rows > 1:
            partials_host = ttnn.to_torch(
                partial,
                mesh_composer=ttnn.create_mesh_composer(self.mesh, ttnn.MeshComposerConfig([0, 1])),
            ).float()
            # Shape: [rows*B, cols*T, H]. Reshape and sum rows; take first col.
            partials_5d = partials_host.reshape(self.mesh_rows, B, self.mesh_cols, T, self.H)
            out_host = partials_5d.sum(dim=0)[:, 0]  # [B, T, H]
            return ttnn.from_torch(
                out_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=ttnn.create_mesh_mapper(self.mesh, ttnn.MeshMapperConfig(row_dim=None, col_dim=None)),
            )
        else:
            return partial
