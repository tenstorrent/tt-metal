# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batched paged full-attention decode (all rows in one SDPA call) for the Qwen3.6 TP4 decoder.

Per chip: 4 query heads, 1 KV head, head dim 256, partial RoPE on the first 64 dims, output gate.
Reuses the layer's packed QKV+gate projection and o_proj; replaces the per-row SDPA loop.
"""

import torch

import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG


class FastAttention:
    def __init__(self, layer, sd, mesh, rows=32):
        self.layer = layer
        self.mesh = mesh
        self.rows = rows
        self.eps = layer.config.rms_norm_eps
        rep = lambda t, dtype=ttnn.bfloat16: ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.q_norm = rep((sd["self_attn.q_norm.weight"].float() + 1).reshape(1, 1, 1, 256))
        self.k_norm = rep((sd["self_attn.k_norm.weight"].float() + 1).reshape(1, 1, 1, 256))
        # rotate_half for the 64 rotary dims as a matmul: x @ P = cat(-x[32:], x[:32])
        p = torch.zeros(64, 64)
        p[32:, :32] = -torch.eye(32)
        p[:32, 32:] = torch.eye(32)
        self.rot = rep(p.reshape(1, 1, 64, 64))
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.num_cores_to_corerangeset(rows, grid, row_wise=True)
        self.kv_shard = ttnn.create_sharded_memory_config(
            shape=(32, 256),
            core_grid=cores,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, math_approx_mode=False
        )
        self.sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=256)

    def _rope(self, x, cos, sin):
        r = x[:, :, :, :64]
        # one batch per core (default config put the 32 row-batches on 2 cores: 36 us -> ~10 us, bit-identical)
        rot = ttnn.matmul(r, self.rot, compute_kernel_config=self.compute, core_grid=ttnn.CoreGrid(y=4, x=8))
        r = ttnn.add(ttnn.multiply(r, cos), ttnn.multiply(rot, sin))
        return ttnn.concat([r, x[:, :, :, 64:]], dim=-1)

    def decode(self, norm, kv_state, *, cur_pos, page_table, cos, sin, attn_mask, update_pos=None):
        """norm [1, R, 2048] normalized hidden (replicated). cur_pos int32 [R] (-1 = inactive),
        page_table int32 [R, P], cos/sin [1, R, 1, 64] bf16, attn_mask [1, R, 4, 256] bf16 0/1.
        Returns the local o_proj partial [1, R, 2048] (caller reduces)."""
        layer = self.layer
        r = self.rows
        projected = layer.linear(norm, "attention_packed", batch_decode=True)  # [1, R, 2560]
        q = ttnn.reshape(projected[:, :, 0:1024], (1, r, 4, 256))
        k = ttnn.reshape(projected[:, :, 1024:1280], (1, r, 1, 256))
        v = ttnn.reshape(projected[:, :, 1280:1536], (1, r, 1, 256))
        q = ttnn.rms_norm(q, epsilon=self.eps, weight=self.q_norm, compute_kernel_config=self.compute)
        k = ttnn.rms_norm(k, epsilon=self.eps, weight=self.k_norm, compute_kernel_config=self.compute)
        q = self._rope(q, cos, sin)
        k = self._rope(k, cos, sin)
        # [1, R, 1, 256] -> one row per core for the paged cache update
        k = ttnn.to_memory_config(
            ttnn.typecast(k, kv_state["k"].dtype) if k.dtype != kv_state["k"].dtype else k, self.kv_shard
        )
        v = ttnn.to_memory_config(
            ttnn.typecast(v, kv_state["v"].dtype) if v.dtype != kv_state["v"].dtype else v, self.kv_shard
        )
        # update_pos (speculative verify): one cache update per candidate column, each with the other columns' rows
        # at -1, so rows of one request never write the same page tile concurrently.
        for upd in update_pos if update_pos is not None else (cur_pos,):
            ttnn.experimental.paged_update_cache(kv_state["k"], k, update_idxs_tensor=upd, page_table=page_table)
            ttnn.experimental.paged_update_cache(kv_state["v"], v, update_idxs_tensor=upd, page_table=page_table)
        o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q,
            kv_state["k"],
            kv_state["v"],
            cur_pos_tensor=cur_pos,
            page_table_tensor=page_table,
            scale=1 / 16,
            program_config=self.sdpa_cfg,
            compute_kernel_config=self.compute,
            memory_config=DRAM,
        )  # [1, R, 4, 256]
        o = ttnn.where(attn_mask, o, 0.0)
        gate = ttnn.reshape(projected[:, :, 1536:2560], (1, r, 4, 256))
        o = ttnn.multiply(o, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
        merged = ttnn.reshape(o, (1, r, 1024))
        return layer._full_output(merged, batch_decode=True)
