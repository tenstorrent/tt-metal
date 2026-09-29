# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 attention, single-chunk prefill prototype (§4).

On device: q stem (wq_a -> q_norm -> wq_b -> RoPE, no per-head q norm), window-KV stem
(wkv -> kv_norm -> RoPE), one ``sparse_sdpa`` call over ``[window KV | compressed KV]`` with the per-head
sink, inverse RoPE on the output and the grouped low-rank output projection. The compressed-KV path and
the window-KV FP8 quantize-dequantize are host fallbacks (``host_fallback.HostCompressedAttention``).

Layout follows the in-tree V4 attention: SP shards tokens (mesh axis 0), TP shards hidden and heads
(axis 1); the input and output are ``[1, 1, S/sp, hidden/tp]``.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.host_fallback import HostCompressedAttention, attention_index_rows
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin


class TtV41Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        seq_len: int,
        host: HostCompressedAttention,
        sp_axis: int = 0,
        tp_axis: int = 1,
        topology=ttnn.Topology.Linear,
        weights_dtype=ttnn.bfloat8_b,
        compute_kernel_config=None,
    ):
        """``weights`` holds bf16/fp32 torch tensors in the checkpoint's [out, in] orientation: wq_a, q_norm,
        wq_b, wkv, kv_norm, wo_a, wo_b, attn_sink. ``seq_len`` is the single chunk length (prefill from 0)."""
        assert sp_axis == 0 and tp_axis == 1, "the prototype supports sp_axis=0, tp_axis=1 only"
        self.mesh_device = mesh_device
        self.config = config
        self.layer = layer
        self.host = host
        self.ratio = config.compress_ratio(layer)
        self.heads, self.head_dim = config.NUM_ATTENTION_HEADS, config.HEAD_DIM
        self.rope_dim, self.window = config.QK_ROPE_HEAD_DIM, config.SLIDING_WINDOW
        self.o_groups, self.eps = config.O_GROUPS, config.RMS_NORM_EPS
        self.scale = self.head_dim**-0.5
        self.sp, self.tp = mesh_device.shape[sp_axis], mesh_device.shape[tp_axis]
        self.sp_axis, self.tp_axis, self.topology = sp_axis, tp_axis, topology
        self.seq_len = seq_len
        self.weights_dtype = weights_dtype
        self.compute_kernel_config = compute_kernel_config
        assert seq_len % (32 * self.sp) == 0, f"chunk {seq_len} must split into whole tiles over sp={self.sp}"
        self.ccl = V41Collectives(mesh_device, topology)
        self.num_links = self.ccl.num_links

        tp_mapper = lambda dim: ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=(None, dim))
        replicate = ttnn.ReplicateTensorToMesh(mesh_device)

        def linear_weight(w, tp_dim):
            t = w.detach().to(torch.bfloat16).transpose(-2, -1).contiguous()[None, None]
            return ttnn.from_torch(
                t, device=mesh_device, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=tp_mapper(tp_dim)
            )

        def row(w):
            return ttnn.from_torch(
                w.detach().to(torch.bfloat16).reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=replicate,
            )

        self.wq_a = linear_weight(weights["wq_a"], 2)  # row-parallel over hidden
        self.wq_b = linear_weight(weights["wq_b"], 3)  # column-parallel over heads
        self.wkv = linear_weight(weights["wkv"], 2)
        self.q_norm = row(weights["q_norm"])
        self.kv_norm = row(weights["kv_norm"])
        in_per_group = self.heads * self.head_dim // self.o_groups
        o_a = weights["wo_a"].detach().to(torch.bfloat16).view(self.o_groups, -1, in_per_group)
        self.wo_a = ttnn.from_torch(
            o_a.transpose(1, 2).unsqueeze(0).contiguous(),
            device=mesh_device,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=tp_mapper(1),
        )
        self.wo_b = linear_weight(weights["wo_b"], 2)
        # sparse_sdpa multiplies the sink by the softmax scale; the reference adds it unscaled.
        sink = (weights["attn_sink"].detach().float() / self.scale).reshape(1, 1, 1, self.heads)
        self.sink = ttnn.from_torch(
            sink.to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=tp_mapper(3),
        )
        self.sink_full = ttnn.from_torch(
            sink.to(torch.bfloat16),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=replicate,
        )
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        cos, sin = cos_sin(config, self.ratio > 0, torch.arange(seq_len))
        sp_mapper = ttnn.ShardTensor2dMesh(mesh_device, tuple(mesh_device.shape), dims=(2, None))
        self.cos, self.sin = (
            ttnn.from_torch(t, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=sp_mapper)
            for t in (cos, sin)
        )

    def _rope(self, t, inverse=False):
        """Rotate the trailing rope_head_dim channels of ``t`` [B, H, S, head_dim]."""
        b, h, s, d = t.shape
        nope_dim = d - self.rope_dim
        nope = ttnn.slice(t, [0, 0, 0, 0], [b, h, s, nope_dim])
        rope = ttnn.slice(t, [0, 0, 0, nope_dim], [b, h, s, d])
        sin = ttnn.neg(self.sin) if inverse else self.sin
        rope = ttnn.experimental.rotary_embedding_llama(rope, self.cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)

    # --- host transfers (prototype fallbacks) --------------------------------------------------------
    def _to_host_replicated_tp(self, t):
        """[1, 1, S/sp, W] replicated across TP -> host [S, W] (take TP column 0)."""
        full = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, 3))
        )
        return full[0, 0, :, : full.shape[-1] // self.tp]

    def _to_host_sharded_tp(self, t):
        """[1, 1, S/sp, W/tp] -> host [S, W]."""
        full = ttnn.to_torch(
            t, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, 3))
        )
        return full[0, 0]

    # --- forward -------------------------------------------------------------------------------------
    def forward(self, x):
        """x [1, 1, S/sp, hidden/tp] bf16 (after attn_norm) -> [1, 1, S/sp, hidden/tp] bf16."""
        seq_local = x.shape[2]
        heads_local = self.heads // self.tp

        qr = ttnn.rms_norm(
            self.ccl.tp_all_reduce(ttnn.linear(x, self.wq_a, compute_kernel_config=self.compute_kernel_config)),
            weight=self.q_norm,
            epsilon=self.eps,
        )
        q = ttnn.linear(qr, self.wq_b, compute_kernel_config=self.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=heads_local, num_kv_heads=0, transpose_k_heads=False
        )
        q = self._rope(q)

        kv = ttnn.rms_norm(
            self.ccl.tp_all_reduce(ttnn.linear(x, self.wkv, compute_kernel_config=self.compute_kernel_config)),
            weight=self.kv_norm,
            epsilon=self.eps,
        )
        kv = self._rope(kv)

        # Host fallback: window-KV QDQ, compressed KV and top-k, index rows.
        window_kv = self.host.window_kv(self._to_host_replicated_tp(kv))
        compressed_idxs, rows = None, [window_kv]
        if self.ratio:
            comp_kv, compressed_idxs = self.host.compressed(
                self.layer, self._to_host_sharded_tp(x), self._to_host_replicated_tp(qr)
            )
            rows.append(comp_kv.to(torch.bfloat16))
        kv_all = torch.cat(rows, dim=0)[None, None]
        # -1 wraps to MASKED_INDEX (0xFFFFFFFF) when the int32 host tensor is stored as uint32.
        index_rows = attention_index_rows(self.seq_len, self.window, compressed_idxs)[None, None].to(torch.int32)

        tt_kv = ttnn.from_torch(
            kv_all,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        tt_idx = ttnn.from_torch(
            index_rows,
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, None)),
        )
        # sparse_sdpa needs >= 32 heads per chip (multiple of 32). A thin TP head shard (64 / tp=4 = 16)
        # is transposed to a sequence shard with one TP all-to-all and back afterwards, as ttMLA does.
        head_to_seq = self.tp > 1 and (heads_local % 32 != 0)
        q_attn = q
        if head_to_seq:
            q_attn = ttnn.experimental.all_to_all_async_generic(
                q,
                in_dim=1,
                out_dim=2,
                num_links=self.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=self.tp_axis,
            )
            tt_idx = ttnn.mesh_partition(tt_idx, dim=2, cluster_axis=self.tp_axis)
        attn = ttnn.transformer.sparse_sdpa(
            ttnn.to_layout(q_attn, ttnn.ROW_MAJOR_LAYOUT),
            tt_kv,
            tt_idx,
            self.head_dim,
            kv_format=ttnn.transformer.SparseKVFormat.BF16,
            scale=self.scale,
            k_chunk_size=next(c for c in (128, 64, 32) if tt_idx.shape[-1] % c == 0),
            attention_sink=self.sink_full if head_to_seq else self.sink,
        )
        if head_to_seq:
            attn = ttnn.experimental.all_to_all_async_generic(
                ttnn.to_layout(attn, ttnn.TILE_LAYOUT),
                in_dim=2,
                out_dim=1,
                num_links=self.num_links,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cluster_axis=self.tp_axis,
            )
        attn = self._rope(ttnn.to_layout(attn, ttnn.TILE_LAYOUT), inverse=True)
        return self._o_proj(attn, seq_local)

    def _o_proj(self, attn, seq_local):
        """[1, H/tp, S/sp, head_dim] -> [1, 1, S/sp, hidden/tp] (same as the in-tree V4 attention)."""
        in_per_group = self.heads * self.head_dim // self.o_groups
        groups_local = self.o_groups // self.tp
        x = ttnn.reshape(attn, [groups_local, attn.shape[1] // groups_local, seq_local, self.head_dim])
        x = ttnn.experimental.nlp_concat_heads(x)
        x = ttnn.reshape(x, [1, groups_local, seq_local, in_per_group])
        grouped = ttnn.linear(x, self.wo_a, compute_kernel_config=self.compute_kernel_config)
        rank = grouped.shape[-1]
        grouped = ttnn.concat(
            [ttnn.slice(grouped, [0, g, 0, 0], [1, g + 1, seq_local, rank]) for g in range(groups_local)], dim=-1
        )
        out = ttnn.linear(grouped, self.wo_b, compute_kernel_config=self.compute_kernel_config)
        return self.ccl.tp_reduce_scatter(out)
