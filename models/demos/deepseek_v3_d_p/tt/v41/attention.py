# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 attention for one prefill chunk (graph nodes B4-B16; beads F3, F7).

q = wq_b(q_norm(wq_a(x))) with RoPE on the trailing 64 channels (no per-head norm); window KV
kv_norm(wkv(x)), RoPE, FP8 QDQ (BF16 KV format; the SCALED_FP8 format encodes the unrounded KV, epic KV
FORMAT; likewise the compressed KV's FP4 QDQ). Compressed layers attend over their KV source's compressed rows too:
a KV source runs its compressor, index keys and compressed-KV write; an index source runs its indexer
(the candidate source publishes candidate blocks); consumers reuse the top-k their index source published
this chunk. One ``sparse_sdpa`` over the layer's KV tensor (``cache.V41PrefillState``: window region + the
source's compressed rows) with the per-head sink, inverse RoPE, grouped low-rank output projection. Position-dependent
inputs (RoPE, window rows) are device slices of ``state.tables`` (``cache.V41ChunkTables``): the forward uploads
nothing, so it can be traced.

Every quantized GEMM input gets the reference's FP8 activation QDQ (dev-spec D-I). Layout: input/output
``[1, 1, S/sp, hidden/tp]``; the attention itself runs on a sequence shard of all heads (head->sequence
all-to-all over TP), which is also the query layout of the indexer's selection.
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.cache import V41PrefillState
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.compressor import TtV41Compressor
from models.demos.deepseek_v3_d_p.tt.v41.indexer import TtV41Indexer, TtV41IndexKeys
from models.demos.deepseek_v3_d_p.tt.v41.layout import TP_AXIS
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_e4m3_qdq, fp8_qdq
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import MlaKvCacheFormat

TOPK_ALIGN = 32  # topk_large_indices needs a multiple of 16; sparse_sdpa k chunks a multiple of 32


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


class TtV41Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        config,
        layer: int,
        weights: dict,
        weights_dtype=ttnn.bfloat8_b,
        compute_kernel_config=None,
    ):
        """``weights`` (torch, checkpoint orientation [out, in], FP8 already dequantized): wq_a, q_norm, wq_b,
        wkv, kv_norm, wo_a, wo_b, attn_sink; plus ``compressor`` {wkv, norm[, wgate]} for KV sources and
        ``indexer`` {wq_b, weights_proj[, wk, k_norm]} for index sources (``tt/v41/weights.py`` keys)."""
        self.mesh_device, self.config, self.layer = mesh_device, config, layer
        self.ratio = config.compress_ratio(layer)
        self.heads, self.head_dim = config.NUM_ATTENTION_HEADS, config.HEAD_DIM
        self.rope_dim, self.window = config.QK_ROPE_HEAD_DIM, config.SLIDING_WINDOW
        self.o_groups, self.eps = config.O_GROUPS, config.RMS_NORM_EPS
        self.scale = self.head_dim**-0.5
        self.sp, self.tp = mesh_device.shape
        self.compute_kernel_config = compute_kernel_config
        self.ccl = V41Collectives(mesh_device)
        shape = tuple(mesh_device.shape)
        tp_mapper = lambda dim: ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(None, dim))
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
        self.q_norm, self.kv_norm = row(weights["q_norm"]), row(weights["kv_norm"])
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
        # sparse_sdpa multiplies the sink by the softmax scale; the reference adds it unscaled
        sink = (weights["attn_sink"].detach().float() / self.scale).reshape(1, 1, 1, self.heads).to(torch.bfloat16)
        self.sink = ttnn.from_torch(
            sink, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=replicate
        )
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )
        # column iota of the index-row compaction (``_compact_leading``): the widest row is window + top-k
        self.compact_rows = _round_up(self.window - 1, 32)  # queries that can lack window rows
        width = _round_up(self.window + (config.INDEX_TOPK if self.ratio else 0), TOPK_ALIGN)
        self.column_iota = ttnn.from_torch(
            torch.arange(width, dtype=torch.int32).expand(self.compact_rows, width).reshape(1, 1, -1, width),
            device=mesh_device,
            dtype=ttnn.int32,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=replicate,
        )

        self.is_kv_source = layer in config.KV_SOURCE_LAYERS
        self.is_index_source = layer in config.INDEX_SOURCE_LAYERS
        self.compressor = (
            TtV41Compressor(mesh_device, config, layer, weights["compressor"]) if self.is_kv_source else None
        )
        self.index_keys = TtV41IndexKeys(mesh_device, config, layer, weights["indexer"]) if self.is_kv_source else None
        self.indexer = TtV41Indexer(mesh_device, config, layer, weights["indexer"]) if self.is_index_source else None

    def _rope(self, t, cos, sin, inverse=False):
        b, h, s, d = t.shape
        nope = ttnn.slice(t, [0, 0, 0, 0], [b, h, s, d - self.rope_dim])
        rope = ttnn.slice(t, [0, 0, 0, d - self.rope_dim], [b, h, s, d])
        rope = ttnn.experimental.rotary_embedding_llama(
            rope, cos, ttnn.neg(sin) if inverse else sin, self.trans_mat, is_decode_mode=False
        )
        return ttnn.concat([nope, rope], dim=-1)

    def _rope_tail(self, t, cos, sin):
        """RoPE on the trailing ``rope_dim`` channels of tiled ``t`` [b, h, s, d] -> row-major [b, h, s, d] (the
        layout sparse_sdpa reads): the untilize copies the other channels, the rotated tail is written over its
        columns; no slice / concat of the full head."""
        b, h, s, d = t.shape
        tail = ttnn.slice(t, [0, 0, 0, d - self.rope_dim], [b, h, s, d])
        tail = ttnn.experimental.rotary_embedding_llama(tail, cos, sin, self.trans_mat, is_decode_mode=False)
        out = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.experimental.slice_write(
            ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT), out, [0, 0, 0, d - self.rope_dim], [b, h, s, d], [1, 1, 1, 1]
        )
        return out

    def _inverse_rope_tail(self, t, cos, sin):
        """Inverse RoPE on the trailing ``rope_dim`` channels of row-major ``t`` [b, h, s, d] (in place) -> tiled."""
        b, h, s, d = t.shape
        tail = ttnn.to_layout(ttnn.slice(t, [0, 0, 0, d - self.rope_dim], [b, h, s, d]), ttnn.TILE_LAYOUT)
        tail = ttnn.experimental.rotary_embedding_llama(tail, cos, ttnn.neg(sin), self.trans_mat, is_decode_mode=False)
        ttnn.experimental.slice_write(
            ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT), t, [0, 0, 0, d - self.rope_dim], [b, h, s, d], [1, 1, 1, 1]
        )
        return ttnn.to_layout(t, ttnn.TILE_LAYOUT)

    def _index_rows(self, window, window_base: int, compressed, start: int):
        """``window``: the chunk's per-chip window rows (``V41ChunkTables.window_rows``) -> per-chip
        [1, 1, S/(sp*tp), K] uint32 rows into the KV tensor, valid first, sentinel tail.

        Both parts are valid-first except the window of a query with fewer than ``window`` earlier tokens: its
        missing rows are a leading run of -1 (``_window_table``); the top-k has a sentinel tail. So [window | top-k]
        is valid-first for every query at or after position window - 1; the earlier ones are compacted."""
        if compressed is None:
            rows = window
        else:
            comp = ttnn.typecast(ttnn.to_layout(compressed, ttnn.TILE_LAYOUT), ttnn.int32)
            comp = ttnn.where(ttnn.eq(comp, -1), -1, ttnn.add(comp, window_base))
            rows = ttnn.concat([window, comp], dim=-1)
        width = rows.shape[-1]
        k = _round_up(width, TOPK_ALIGN)
        if k != width:
            rows = ttnn.pad(rows, [(0, 0), (0, 0), (0, 0), (0, k - width)], -1)
        if start < self.window - 1:
            rows = self._compact_leading(rows, window)
        return ttnn.to_layout(ttnn.typecast(rows, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)

    def _compact_leading(self, rows, window):
        """Rotate each of the chip's first ``compact_rows`` query rows left by its count of missing window rows (the
        leading -1 run): valid rows first, the -1 run after the top-k's sentinel tail. Later queries (all window rows
        present) rotate by 0; a chip's rows are contiguous query positions, so no later chip needs more rows."""
        q_rows, k = rows.shape[2], rows.shape[3]
        n = min(self.compact_rows, q_rows)
        head_window = ttnn.slice(window, [0, 0, 0, 0], [1, 1, n, window.shape[3]])
        missing = ttnn.sum(ttnn.typecast(ttnn.eq(head_window, -1), ttnn.float32), dim=-1, keepdim=True)
        shift = ttnn.add(ttnn.slice(self.column_iota, [0, 0, 0, 0], [1, 1, n, k]), ttnn.typecast(missing, ttnn.int32))
        # sentinel columns past the end keep every shifted index in range
        head = ttnn.pad(
            ttnn.slice(rows, [0, 0, 0, 0], [1, 1, n, k]), [(0, 0), (0, 0), (0, 0), (0, window.shape[3])], -1
        )
        head = ttnn.gather(head, -1, ttnn.typecast(shift, ttnn.uint32))
        if n == q_rows:
            return head
        return ttnn.concat([head, ttnn.slice(rows, [0, 0, n, 0], [1, 1, q_rows, k])], dim=2)

    # --- forward -------------------------------------------------------------------------------------
    def forward(self, x, state: V41PrefillState, length: int):
        """x [1, 1, S/sp, hidden/tp] bf16 (after attn_norm) for the chunk at ``state.start`` with ``length``
        valid tokens -> [1, 1, S/sp, hidden/tp] bf16. Writes this layer's window carry and, for a KV source,
        its compressed rows and carry; an index source publishes its selection in ``state.selection``."""
        start, seq_local = state.start, x.shape[2]
        heads_local = self.heads // self.tp
        tables = state.tables
        cos, sin = tables.rope(self.ratio > 0, 1, start)
        xq = fp8_qdq(x)

        qr = ttnn.rms_norm(
            self.ccl.tp_all_reduce(ttnn.linear(xq, self.wq_a, compute_kernel_config=self.compute_kernel_config)),
            weight=self.q_norm,
            epsilon=self.eps,
        )
        q = ttnn.linear(fp8_qdq(qr), self.wq_b, compute_kernel_config=self.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=heads_local, num_kv_heads=0, transpose_k_heads=False
        )

        kv = ttnn.rms_norm(
            self.ccl.tp_all_reduce(ttnn.linear(xq, self.wkv, compute_kernel_config=self.compute_kernel_config)),
            weight=self.kv_norm,
            epsilon=self.eps,
        )
        # stage 1 (BF16) stores the reference's QDQ values; SCALED_FP8 quantizes the unrounded KV itself
        kv_format = state.layer_kv_format(self.layer)
        reference_values = kv_format == MlaKvCacheFormat.BF16_RM
        kv = self._rope(kv, cos, sin)
        kv_tensor = state.write_window(self.layer, fp8_qdq(kv) if reference_values else kv)

        compressed = None
        if self.ratio:
            if self.is_kv_source:
                latent, carry = self.compressor(x)
                state.set_compressor_carry(self.layer, carry, length)
                c_rope = tables.rope(True, self.ratio, start)  # the compressed rows' group-first positions
                keys = self.index_keys(latent, c_rope)
                comp_kv = self._rope(latent, *c_rope)
                state.write_compressed(self.layer, fp4_e4m3_qdq(comp_kv) if reference_values else comp_kv, keys, length)
            if self.is_index_source:
                src = self.config.kv_source(self.layer)
                compressed, candidates = self.indexer(
                    x, qr, state.index_k[src], tables, start, length, state.selection.get("candidates")
                )
                state.selection["topk"] = compressed
                if candidates is not None:
                    state.selection["candidates"] = candidates
            else:
                compressed = state.selection["topk"]
        rows = self._index_rows(tables.window_rows(start), state.geometry.window_rows, compressed, start)

        # sparse_sdpa needs >= 32 heads per chip: attend on a sequence shard of all heads (head->sequence); RoPE and
        # its inverse run on that shard (the chip's rows of cos / sin), next to the row-major sparse_sdpa operands
        head_to_seq = self.tp > 1
        if head_to_seq:
            q = self.ccl.tp_all_to_all(q, in_dim=1, out_dim=2)
            cos, sin = (ttnn.mesh_partition(t, dim=2, cluster_axis=TP_AXIS) for t in (cos, sin))
        attn = ttnn.transformer.sparse_sdpa(
            self._rope_tail(q, cos, sin),
            kv_tensor,
            rows,
            self.head_dim,
            kv_format=kv_format.sparse_sdpa_format,
            scale=self.scale,
            k_chunk_size=next(c for c in (128, 64, 32) if rows.shape[-1] % c == 0),
            attention_sink=self.sink,
        )
        attn = self._inverse_rope_tail(attn, cos, sin)
        if head_to_seq:
            attn = self.ccl.tp_all_to_all(attn, in_dim=2, out_dim=1)
        state.update_window_carry(self.layer, length)
        return self._o_proj(attn, seq_local)

    def _o_proj(self, attn, seq_local):
        """[1, H/tp, S/sp, head_dim] -> [1, 1, S/sp, hidden/tp]."""
        in_per_group = self.heads * self.head_dim // self.o_groups
        groups_local = self.o_groups // self.tp
        x = ttnn.reshape(attn, [groups_local, attn.shape[1] // groups_local, seq_local, self.head_dim])
        x = ttnn.experimental.nlp_concat_heads(x)
        x = ttnn.reshape(x, [1, groups_local, seq_local, in_per_group])
        grouped = ttnn.experimental.nlp_concat_heads(
            ttnn.linear(x, self.wo_a, compute_kernel_config=self.compute_kernel_config)
        )  # [1, groups, S, rank] -> [1, 1, S, groups * rank]
        out = ttnn.linear(fp8_qdq(grouped), self.wo_b, compute_kernel_config=self.compute_kernel_config)
        return self.ccl.tp_reduce_scatter(out)
