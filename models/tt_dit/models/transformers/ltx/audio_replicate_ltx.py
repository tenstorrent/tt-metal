# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""LTX_AUDIO_REPLICATE: run audio self-attention and the audio FFN with every TP chip holding the full weights.

The audio stream is 256 tokens, so its TP-sharded matmuls are tiny and the block time goes to the
~8 TP collectives around them (two distributed norms, the QK norms, QKV/to_out/ff1 all-gather-matmuls
and the ff2 reduce-scatter). Each island instead gathers the residual once, runs norm, projection,
SDPA and residual on full rows with replicated weights, and keeps its TP slice with a local
`mesh_partition`, so one all-gather per island replaces those collectives.

Replicated weights are built on device from the loaded TP shards after the weight cache is read, so
the cache layout and key do not change; the sharded modules stay loaded but unused.
"""

from __future__ import annotations

import os

import torch

import ttnn

from ....layers.linear import ColParallelLinear, maybe_cast_activation
from ....utils.tensor import bf16_tensor


def audio_replicate_enabled() -> bool:
    return os.environ.get("LTX_AUDIO_REPLICATE", "0") not in ("0", "", "false", "False")


def qkv_regroup_columns(tp_factor: int, local: int) -> list[list[tuple[int, int]]]:
    """Column ranges of Q, K and V in a TP-gathered QKV weight, in head order.

    Each device's shard is [q_d | k_d | v_d] for its own heads, so the gathered weight is device-major.
    """
    return [[((3 * d + c) * local, (3 * d + c + 1) * local) for d in range(tp_factor)] for c in range(3)]


def _tp_shards(tensor: ttnn.Tensor, mesh_shape: tuple[int, int], tp_axis: int) -> list[torch.Tensor]:
    """Host copies of a TP-sharded tensor's shards, one per TP index (the other axis holds replicas)."""
    per_device = ttnn.get_device_tensors(tensor)
    cols = mesh_shape[1]
    shards = []
    for tp_idx in range(mesh_shape[tp_axis]):
        coord = (tp_idx, 0) if tp_axis == 0 else (0, tp_idx)
        shards.append(ttnn.to_torch(per_device[coord[0] * cols + coord[1]]))
    return shards


class ReplicatedAudioIslands:
    """Replicated audio self-attention + FFN for one LTX block, built from its loaded TP-sharded modules."""

    def __init__(self, block) -> None:
        attn = block.audio_attn1
        ff = block.audio_ff
        pc = block.parallel_config
        self.attn = attn
        self.ccl_manager = block.ccl_manager
        self.mesh_device = block.mesh_device
        self.tp_axis = pc.tensor_parallel.mesh_axis
        self.tp_factor = pc.tensor_parallel.factor
        self.sp_axis = pc.sequence_parallel.mesh_axis
        self.sp_factor = pc.sequence_parallel.factor
        self.eps = block.audio_norm1.norm_eps
        self.norm_compute_config = block.audio_norm1.compute_kernel_config
        self.ff_compute_config = block.ff_compute_kernel_config
        self.num_heads = attn.num_heads
        self.head_dim = attn.head_dim
        assert self.tp_factor > 1, "LTX_AUDIO_REPLICATE needs TP > 1"
        assert not attn.fuse_gate, "LTX_AUDIO_REPLICATE does not support LTX_FUSE_GATE=1"
        assert attn.to_qkv.weight._data is not None, "replicate before fold_gates_on_device releases the shards"
        mesh_shape = tuple(self.mesh_device.shape)
        dim = attn.dim
        local = dim // self.tp_factor

        def replicate(t: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
            return ttnn.from_torch(
                t,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        def gather_cols(param) -> ttnn.Tensor:
            return self._gather(param.data, dim=1)

        def host_cols(param) -> torch.Tensor:
            return torch.cat(_tp_shards(param.data, mesh_shape, self.tp_axis), dim=-1)

        regroup = qkv_regroup_columns(self.tp_factor, local)
        qkv_dev = gather_cols(attn.to_qkv.weight)
        cols = [ttnn.concat([ttnn.slice(qkv_dev, [0, a], [dim, b]) for a, b in ranges], dim=-1) for ranges in regroup]
        ttnn.deallocate(qkv_dev)
        chunk_sizes = [dim, dim, dim]
        bias_parts = None
        if attn.to_qkv.bias is not None:
            b = host_cols(attn.to_qkv.bias)
            bias_parts = [torch.cat([b[..., a:e] for a, e in ranges], dim=-1) for ranges in regroup]
        # The gate is num_heads columns: under a tile per device, so it is reassembled on host.
        self.gated = attn._gate_is_live()
        if self.gated:
            g_w = host_cols(attn.to_gate_logits.weight)
            pad = -g_w.shape[1] % ttnn.TILE_SIZE
            cols.append(replicate(torch.nn.functional.pad(g_w, (0, pad))))
            chunk_sizes.append(g_w.shape[1] + pad)
            if bias_parts is not None:
                g_b = host_cols(attn.to_gate_logits.bias)
                bias_parts.append(torch.nn.functional.pad(g_b, (0, pad)))
        self.gate_width = self.num_heads

        self.to_qkv = ColParallelLinear(
            dim,
            sum(chunk_sizes),
            bias=bias_parts is not None,
            mesh_device=self.mesh_device,
            mesh_axis=None,
            chunks=len(chunk_sizes),
            chunk_sizes=chunk_sizes,
            compute_kernel_config=attn.to_qkv.compute_config,
        )
        self.to_qkv.weight.data = ttnn.concat(cols, dim=-1)
        for t in cols:
            ttnn.deallocate(t)
        if bias_parts is not None:
            self.to_qkv.bias.data = replicate(torch.cat(bias_parts, dim=-1))

        self.norm_q_w = replicate(host_cols(attn.norm_q.weight))
        self.norm_k_w = replicate(host_cols(attn.norm_k.weight))

        self.to_out = ColParallelLinear(
            dim,
            attn.to_out.out_features,
            bias=attn.to_out.bias is not None,
            mesh_device=self.mesh_device,
            mesh_axis=None,
            compute_kernel_config=attn.to_out.compute_config,
        )
        self.to_out.weight.data = gather_cols(attn.to_out.weight)
        if attn.to_out.bias is not None:
            self.to_out.bias.data = replicate(host_cols(attn.to_out.bias))

        self.ff1 = ColParallelLinear(
            ff.dim,
            ff.inner_dim,
            bias=ff.ff1.bias is not None,
            activation_fn=ff.activation_fn,
            mesh_device=self.mesh_device,
            mesh_axis=None,
            compute_kernel_config=ff.ff1.compute_config,
        )
        self.ff1.weight.data = gather_cols(ff.ff1.weight)
        if ff.ff1.bias is not None:
            self.ff1.bias.data = replicate(host_cols(ff.ff1.bias))
        self.ff2 = ColParallelLinear(
            ff.inner_dim,
            ff.dim_out,
            bias=ff.ff2.bias is not None,
            mesh_device=self.mesh_device,
            mesh_axis=None,
            compute_kernel_config=ff.ff2.compute_config,
        )
        # ff2 is row-parallel: its rows follow ff1's column shards, and only TP index 0 holds the bias.
        self.ff2.weight.data = self._gather(ff.ff2.weight.data, dim=0)
        if ff.ff2.bias is not None:
            self.ff2.bias.data = replicate(_tp_shards(ff.ff2.bias.data, mesh_shape, self.tp_axis)[0])

        self.dummy_joint = bf16_tensor(torch.zeros((1, self.num_heads, 0, self.head_dim)), device=self.mesh_device)

    def _gather(self, t: ttnn.Tensor, dim: int) -> ttnn.Tensor:
        """All-gather along TP on a 2D or 4D tensor, keeping its rank."""
        rank = len(t.shape)
        out = self.ccl_manager.all_gather(t, dim=dim, mesh_axis=self.tp_axis, use_hyperparams=False)
        if len(out.shape) != rank:
            out = ttnn.reshape(out, list(out.shape)[len(out.shape) - rank :])
        return out

    def _enter(self, x_frac: ttnn.Tensor, shift: ttnn.Tensor, scale_p1: ttnn.Tensor):
        x_full = self._gather(x_frac, dim=3)
        normed = ttnn.experimental.dit_rms_norm_unary_fused(
            x_full, weight=None, bias=None, epsilon=self.eps, compute_kernel_config=self.norm_compute_config
        )
        return x_full, ttnn.addcmul(shift, normed, scale_p1)

    def _leave(self, x_full: ttnn.Tensor, out: ttnn.Tensor, gate: ttnn.Tensor) -> ttnn.Tensor:
        x_full = ttnn.addcmul(x_full, out, gate)
        return ttnn.mesh_partition(x_full, dim=3, cluster_axis=self.tp_axis)

    def self_attn(
        self,
        x_frac: ttnn.Tensor,
        shift: ttnn.Tensor,
        scale_p1: ttnn.Tensor,
        gate: ttnn.Tensor,
        *,
        N: int,
        rope_cos: ttnn.Tensor,
        rope_sin: ttnn.Tensor,
        trans_mat: ttnn.Tensor,
        attn_mask: ttnn.Tensor | None = None,
        skip_qk: bool = False,
    ) -> ttnn.Tensor:
        """Audio self-attention with gated residual; rope_cos/sin carry all heads (gathered on TP)."""
        attn = self.attn
        x_full, normed = self._enter(x_frac, shift, scale_p1)
        outs = self.to_qkv(normed, compute_kernel_config=attn.mm_compute_kernel_config)
        q, k, v = outs[0], outs[1], outs[2]

        def heads(t):
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                t, num_heads=self.num_heads, num_kv_heads=0, transpose_k_heads=False
            )
            return out

        v = heads(v)
        if skip_qk:
            spatial = v
        else:
            # LTX's QK norm spans the whole inner dim, across heads, so it runs before the head split.
            q = ttnn.experimental.dit_rms_norm_unary_fused(
                q, weight=self.norm_q_w, epsilon=attn.eps, compute_kernel_config=self.norm_compute_config
            )
            k = ttnn.experimental.dit_rms_norm_unary_fused(
                k, weight=self.norm_k_w, epsilon=attn.eps, compute_kernel_config=self.norm_compute_config
            )
            q = ttnn.experimental.rotary_embedding_llama(
                heads(q), rope_cos, rope_sin, trans_mat, compute_kernel_config=attn.rope_compute_kernel_config
            )
            k = ttnn.experimental.rotary_embedding_llama(
                heads(k), rope_cos, rope_sin, trans_mat, compute_kernel_config=attn.rope_compute_kernel_config
            )
            spatial = self._sdpa(q, k, v, N, attn_mask)

        if self.gated:
            logits = outs[3]
            if logits.shape[-1] != self.gate_width:
                s = logits.shape
                logits = ttnn.slice(logits, [0, 0, 0, 0], [s[0], s[1], s[2], self.gate_width])
            spatial = ttnn.multiply(spatial, attn._gate_from_logits(logits))

        spatial = ttnn.unsqueeze(ttnn.transformer.concatenate_heads(spatial), 0)
        out = self.to_out(spatial, compute_kernel_config=attn.mm_compute_kernel_config)
        return self._leave(x_full, out, gate)

    def _sdpa(self, q, k, v, N: int, attn_mask):
        attn = self.attn
        dummy = self.dummy_joint
        sdpa_dtype = getattr(attn, "_sdpa_input_dtype", None)
        if sdpa_dtype is not None:
            q, k, v, dummy = (maybe_cast_activation(t, sdpa_dtype) for t in (q, k, v, dummy))
        if self.sp_factor > 1 and attn_mask is None:
            out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                q,
                k,
                v,
                dummy,
                dummy,
                dummy,
                persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                    k.shape, 2, self.sp_axis, dtype=k.get_dtype()
                ),
                persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                    v.shape, 2, self.sp_axis, dtype=v.get_dtype()
                ),
                joint_strategy="rear",
                logical_n=N,
                program_config=attn._ring_pc_by_n.get(
                    -(-N // (ttnn.TILE_SIZE * self.sp_factor)) * ttnn.TILE_SIZE * self.sp_factor,
                    attn.ring_sdpa_program_config,
                ),
                compute_kernel_config=attn.sdpa_compute_kernel_config,
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.sp_axis),
                num_links=self.ccl_manager.num_links,
                cluster_axis=self.sp_axis,
                mesh_device=self.mesh_device,
                topology=self.ccl_manager.topology,
                subdevice_id=self.ccl_manager.ccl_sub_device_id,
                ccl_core_grid_offset=(attn.sdpa_worker_grid[0], 0),
                use_column_major_ccl=True,
            )
            return out
        if self.sp_factor > 1:
            k = self.ccl_manager.all_gather_persistent_buffer(k, dim=2, mesh_axis=self.sp_axis)
            v = self.ccl_manager.all_gather_persistent_buffer(v, dim=2, mesh_axis=self.sp_axis)
        return ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=False,
            program_config=attn.sdpa_program_config,
            compute_kernel_config=attn.sdpa_compute_kernel_config,
        )

    def ffn(self, x_frac: ttnn.Tensor, shift: ttnn.Tensor, scale_p1: ttnn.Tensor, gate: ttnn.Tensor) -> ttnn.Tensor:
        x_full, normed = self._enter(x_frac, shift, scale_p1)
        h = self.ff1(normed, compute_kernel_config=self.ff_compute_config)
        out = self.ff2(h, compute_kernel_config=self.ff_compute_config)
        return self._leave(x_full, out, gate)
