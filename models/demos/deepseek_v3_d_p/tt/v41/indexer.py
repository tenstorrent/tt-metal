# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 index keys, indexer, candidate selection and top-k (bead F4; graph nodes B7, B9-B12).

Index keys (KV sources): ``k = k_norm(wk(latent))``, RoPE at each row's group-first position, FP4 QDQ
(block 32, ue8m0). Indexer (index sources): ``q = wq_b(qr)`` split into 32 heads, RoPE, FP4 QDQ;
``score[s, t] = sum_h relu(q_h . k_t) * w_h`` with ``w = weights_proj(x) * (d^-1/2 * H^-1/2)``; row t is visible
to the query at absolute position p iff ``t < (p + 1) // r``. The candidate source (layer 20) keeps, per query,
the 2048 blocks of 8 rows with the best max score, always including the block of its newest row; candidate
index sources mask their scores to those blocks. Top-k keeps ``min(512, visible)`` rows; unreachable picks
become the 0xFFFFFFFF sentinel, in one tail (descending score order).

Distribution: queries are split over SP and then TP (each chip scores ``S/(sp*tp)`` contiguous queries with all
32 heads), so nothing is reduced across chips and the selection comes out in the attention's head->sequence
query layout. Scores use ``indexer_score_dsa`` (ratio 1 relies on its token-causal mask; ratio 2 adds its own).
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_ue8m0_qdq, fp8_qdq
from models.demos.deepseek_v3_d_p.tt.v41.rope import cos_sin

SENTINEL = 0xFFFFFFFF
TOPK_MIN, TOPK_ALIGN = 16, 16


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


class _Rope:
    """Interleaved RoPE of the trailing rope_head_dim channels at given positions (compressed-layer table)."""

    def __init__(self, mesh_device, config):
        self.mesh_device, self.config = mesh_device, config
        self.rope_dim = config.QK_ROPE_HEAD_DIM
        self.trans_mat = ttnn.from_torch(
            get_rot_transformation_mat(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    def tables(self, positions: torch.Tensor, mapper):
        cos, sin = cos_sin(self.config, True, positions)
        return tuple(
            ttnn.from_torch(
                t, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            for t in (cos, sin)
        )

    def __call__(self, t, cos, sin):
        b, h, s, d = t.shape
        nope = ttnn.slice(t, [0, 0, 0, 0], [b, h, s, d - self.rope_dim])
        rope = ttnn.slice(t, [0, 0, 0, d - self.rope_dim], [b, h, s, d])
        rope = ttnn.experimental.rotary_embedding_llama(rope, cos, sin, self.trans_mat, is_decode_mode=False)
        return ttnn.concat([nope, rope], dim=-1)


class TtV41IndexKeys(LightweightModule):
    """B7: index keys from a KV source's RoPE-free latent."""

    def __init__(self, mesh_device, config, layer: int, weights: dict, qdq=fp4_ue8m0_qdq):
        """``weights``: ``wk`` [index_head_dim, head_dim] bf16, ``k_norm`` [index_head_dim]. ``qdq``: FP4 block-32
        ue8m0 quantize-dequantize on a tiled bf16 tensor."""
        self.mesh_device, self.config = mesh_device, config
        self.ratio = config.compress_ratio(layer)
        self.qdq = qdq
        self.rope = _Rope(mesh_device, config)
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        self.wk = ttnn.from_torch(
            weights["wk"].detach().to(torch.bfloat16).T.contiguous()[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        self.k_norm = ttnn.from_torch(
            weights["k_norm"].detach().to(torch.bfloat16).reshape(1, 1, 1, -1),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )

    def forward(self, latent, start: int):
        """latent [1, 1, rows/sp, head_dim] bf16 (SP rank r holds the rows of its contiguous tokens), chunk start
        -> index keys [1, 1, rows/sp, index_head_dim] bf16 after RoPE and QDQ."""
        rows_local = latent.shape[2]
        sp = self.mesh_device.shape[0]
        first = start // self.ratio
        positions = (torch.arange(first, first + rows_local * sp) * self.ratio).view(-1)
        cos, sin = self.rope.tables(
            positions, ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(2, None))
        )
        k = ttnn.linear(latent, self.wk, compute_kernel_config=self.compute_kernel_config)
        k = ttnn.rms_norm(k, weight=self.k_norm, epsilon=self.config.RMS_NORM_EPS)
        return self.qdq(self.rope(k, cos, sin))


class TtV41Indexer(LightweightModule):
    """B9-B12 for one index source; returns the per-query selection in head->sequence query layout."""

    def __init__(self, mesh_device, config, layer: int, weights: dict, qdq=fp4_ue8m0_qdq):
        """``weights``: ``wq_b`` [heads*d, q_lora] bf16, ``weights_proj`` [heads, hidden] bf16. ``qdq`` as in keys."""
        self.mesh_device, self.config, self.layer = mesh_device, config, layer
        self.ratio = config.compress_ratio(layer)
        self.heads, self.head_dim = config.INDEX_N_HEADS, config.INDEX_HEAD_DIM
        self.is_candidate_source = layer == config.CANDIDATE_SOURCE_LAYER
        self.uses_candidates = 0 <= config.CANDIDATE_SOURCE_LAYER < layer
        self.qdq = qdq
        self.rope = _Rope(mesh_device, config)
        self.ccl = V41Collectives(mesh_device)
        self.sp, self.tp = mesh_device.shape
        shape = tuple(mesh_device.shape)
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        # FP8 checkpoint weights are exact in bf16; the reference quantizes qr to FP8 (act_quant) before this
        # GEMM, so the device does too (dev-spec D-I) and accumulates in fp32.
        self.wq_b = ttnn.from_torch(
            weights["wq_b"].detach().to(torch.bfloat16).T.contiguous()[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=rep,
        )
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        )
        scale = self.head_dim**-0.5 * self.heads**-0.5
        self.weights_proj = ttnn.from_torch(
            (weights["weights_proj"].detach().float() * scale).to(torch.bfloat16).T.contiguous()[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(None, 2)),  # row-parallel over hidden
        )

    def _per_query_chip(self, host: torch.Tensor, dtype=ttnn.bfloat16):
        """Host rows in token order [n * q_rows, W] -> each chip its contiguous q_rows ((sp, tp) chip order)."""
        n_rows, width = host.shape[-2], host.shape[-1]
        q_rows = n_rows // (self.sp * self.tp)
        return ttnn.from_torch(
            host.reshape(self.sp, self.tp, q_rows, width),
            device=self.mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, tuple(self.mesh_device.shape), dims=(0, 1)),
        )

    def _query_shard(self, t):
        """[1, *, S/sp, W] replicated across TP -> this chip's contiguous quarter of the SP shard."""
        return ttnn.mesh_partition(t, dim=2, cluster_axis=1) if self.tp > 1 else t

    def _visibility_mask(self, start: int, rows_local: int, width: int):
        """Additive mask [1, 1, S/(sp*tp), width]: -inf where row t >= (p + 1) // ratio."""
        n = self.sp * self.tp
        positions = start + torch.arange(n * rows_local)
        limit = ((positions + 1) // self.ratio).view(n * rows_local, 1)
        mask = torch.where(torch.arange(width).view(1, width) >= limit, float("-inf"), 0.0)
        return self._per_query_chip(mask)

    def scores(self, x, qr, index_k, start: int, length: int):
        """x [1,1,S/sp,hidden/tp] bf16, qr [1,1,S/sp,q_lora] bf16 (TP-replicated), index_k [1,1,T_max,d] tiled
        -> scores [1, 1, S/(sp*tp), T] bf16 (T = visible rows rounded up to tiles), -inf where not visible."""
        rows = qr.shape[2]
        q_rows = rows // self.tp
        visible = (start + length) // self.ratio
        width = _round_up(max(visible, TOPK_MIN), 32)
        positions = start + torch.arange(self.sp * self.tp * q_rows)
        cos, sin = (self._per_query_chip(t[0, 0]) for t in cos_sin(self.config, True, positions))
        q = ttnn.linear(fp8_qdq(self._query_shard(qr)), self.wq_b, compute_kernel_config=self.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.heads, num_kv_heads=0, transpose_k_heads=False
        )
        q = self.qdq(self.rope(q, cos, sin))
        w = self._query_shard(self.ccl.tp_all_reduce(ttnn.linear(x, self.weights_proj)))
        k = ttnn.to_layout(ttnn.slice(index_k, [0, 0, 0, 0], [1, 1, width, self.head_dim]), ttnn.TILE_LAYOUT)
        # The kernel's causal rule applies chunk_start_idx as-is on every chip (no per-chip offset under
        # seq_shard_axes=[]), so it cannot express this query split: pad K by one zero tile and start the
        # kernel's window at `width`, which masks nothing real; the V4.1 rule is applied explicitly below.
        k = ttnn.pad(k, [(0, 0), (0, 0), (0, 32), (0, 0)], 0.0)
        score = ttnn.experimental.indexer_score_dsa(
            q, k, w, kv_len=width + 32, chunk_start_idx=width, seq_shard_axes=[]
        )
        score = ttnn.slice(ttnn.to_layout(score, ttnn.TILE_LAYOUT), [0, 0, 0, 0], [1, 1, q_rows, width])
        score = ttnn.add(score, self._visibility_mask(start, q_rows, width))
        return ttnn.to_layout(score, ttnn.ROW_MAJOR_LAYOUT), visible

    def candidates(self, score, start: int, q_rows: int, visible: int):
        """Candidate source: additive mask [1,1,S/(sp*tp),T] (0 inside the kept blocks, -inf elsewhere)."""
        block = self.config.CANDIDATE_BLOCK_SIZE
        width = score.shape[-1]
        nblocks = -(-width // block)
        s = ttnn.to_layout(score, ttnn.TILE_LAYOUT)
        if nblocks * block != width:
            s = ttnn.pad(s, [(0, 0), (0, 0), (0, 0), (0, nblocks * block - width)], float("-inf"))
        # block score = max over its rows: elementwise max of the `block` strided column slices (a tile
        # reduction would see the zero padding of a narrow last dim)
        s = ttnn.to_layout(s, ttnn.ROW_MAJOR_LAYOUT)
        blocks = None
        for i in range(block):
            col = ttnn.slice(s, [0, 0, 0, i], [1, 1, q_rows, nblocks * block], [1, 1, 1, block])
            blocks = col if blocks is None else ttnn.maximum(blocks, col)
        # pin the block holding each query's newest visible row
        n = self.sp * self.tp
        positions = start + torch.arange(n * q_rows)
        newest = (((positions + 1) // self.ratio) - 1) // block
        pin = torch.where(torch.arange(nblocks).view(1, -1) == newest.view(-1, 1), float("inf"), 0.0)
        pin_tt = self._per_query_chip(pin)
        blocks = ttnn.add(ttnn.to_layout(blocks, ttnn.TILE_LAYOUT), pin_tt)
        k = min(self.config.CANDIDATE_TOPK_BLOCKS, nblocks)
        k_pad = _round_up(max(k, TOPK_MIN), TOPK_ALIGN)
        padded = _round_up(max(nblocks, k_pad), 32) + 32  # spare columns: -inf fill and a sentinel target
        blocks = ttnn.pad(blocks, [(0, 0), (0, 0), (0, 0), (0, padded - nblocks)], float("-inf"))
        idx = ttnn.experimental.topk_large_indices(ttnn.to_layout(blocks, ttnn.ROW_MAJOR_LAYOUT), k=k_pad)
        if k_pad != k:
            idx = ttnn.slice(idx, [0, 0, 0, 0], [1, 1, q_rows, k])
        # sentinel picks (fewer reachable blocks than k) scatter into the last spare column
        idx = ttnn.where(
            ttnn.eq(ttnn.typecast(idx, ttnn.float32), float(SENTINEL)),
            float(padded - 1),
            ttnn.typecast(idx, ttnn.float32),
        )
        keep = ttnn.scatter(
            ttnn.full(
                [1, 1, q_rows, padded], 0.0, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
            ),
            -1,
            ttnn.typecast(idx, ttnn.int32),
            ttnn.full(
                [1, 1, q_rows, k], 1.0, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
            ),
        )
        keep = ttnn.slice(keep, [0, 0, 0, 0], [1, 1, q_rows, nblocks])
        keep = ttnn.repeat_interleave(ttnn.to_layout(keep, ttnn.TILE_LAYOUT), block, dim=-1)
        keep = ttnn.slice(keep, [0, 0, 0, 0], [1, 1, q_rows, width])
        # additive mask: 0 where kept, -inf elsewhere
        return ttnn.where(ttnn.gtz(keep), 0.0, float("-inf"))

    def forward(self, x, qr, index_k, start: int, length: int, candidate_mask=None):
        """-> (top-k rows [1, 1, S/(sp*tp), k] uint32 row-major (sentinel tail), candidate mask or None)."""
        score, visible = self.scores(x, qr, index_k, start, length)
        q_rows = score.shape[2]
        published = None
        if self.is_candidate_source:
            published = self.candidates(score, start, q_rows, visible)
        elif self.uses_candidates:
            assert candidate_mask is not None, "a candidate index source needs the published candidates"
            score = ttnn.to_layout(
                ttnn.add(ttnn.to_layout(score, ttnn.TILE_LAYOUT), candidate_mask), ttnn.ROW_MAJOR_LAYOUT
            )
        k = min(self.config.INDEX_TOPK, visible)
        k_pad = _round_up(max(k, TOPK_MIN), TOPK_ALIGN)
        idx = ttnn.experimental.topk_large_indices(score, k=k_pad)
        if k_pad != k:
            idx = ttnn.slice(idx, [0, 0, 0, 0], [1, 1, q_rows, k])
        return idx, published
