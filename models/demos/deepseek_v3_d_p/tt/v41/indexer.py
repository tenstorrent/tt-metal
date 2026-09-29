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
No host tables: position-dependent inputs are slices of the geometry's ``cache.V41ChunkTables`` (trace-safe).
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_ue8m0_qdq, fp8_qdq

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

    def forward(self, latent, rope):
        """latent [1, 1, rows/sp, head_dim] bf16 (SP rank r holds the rows of its contiguous tokens), ``rope``: the
        (cos, sin) of those rows' group-first positions (``V41ChunkTables.rope(True, ratio, start)``)
        -> index keys [1, 1, rows/sp, index_head_dim] bf16 after RoPE and QDQ."""
        k = ttnn.linear(latent, self.wk, compute_kernel_config=self.compute_kernel_config)
        k = ttnn.rms_norm(k, weight=self.k_norm, epsilon=self.config.RMS_NORM_EPS)
        return self.qdq(self.rope(k, *rope))


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

    def _query_shard(self, t):
        """[1, *, S/sp, W] replicated across TP -> this chip's contiguous quarter of the SP shard."""
        return ttnn.mesh_partition(t, dim=2, cluster_axis=1) if self.tp > 1 else t

    @staticmethod
    def _add_tail(t, first: int, tail):
        """Tiled ``t`` [1, 1, R, first + w] with the additive ``tail`` [1, 1, R, w] added to columns [first, first + w)
        (the columns before ``first`` are unmasked)."""
        rows, width = t.shape[2], t.shape[3]
        if first == 0:
            return ttnn.add(t, tail)
        head = ttnn.slice(t, [0, 0, 0, 0], [1, 1, rows, first])
        rest = ttnn.add(ttnn.slice(t, [0, 0, 0, first], [1, 1, rows, width]), tail)
        return ttnn.concat([head, rest], dim=-1)

    def scores(self, x, qr, index_k, tables, start: int, length: int):
        """x [1,1,S/sp,hidden/tp] bf16, qr [1,1,S/sp,q_lora] bf16 (TP-replicated), index_k [1,1,T_max,d] tiled,
        ``tables`` the geometry's ``V41ChunkTables`` -> scores [1, 1, S/(sp*tp), T] bf16 row-major (T = visible rows
        rounded up to tiles), -inf where not visible."""
        rows = qr.shape[2]
        q_rows = rows // self.tp
        assert start % self.ratio == 0 and (start // self.ratio) % 32 == 0, f"chunk start {start} not tile-aligned"
        visible = (start + length) // self.ratio
        width = _round_up(max(visible, TOPK_MIN), 32)
        cos, sin = (self._query_shard(t) for t in tables.rope(True, 1, start))
        q = ttnn.linear(fp8_qdq(self._query_shard(qr)), self.wq_b, compute_kernel_config=self.compute_kernel_config)
        q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            q, num_heads=self.heads, num_kv_heads=0, transpose_k_heads=False
        )
        q = self.qdq(self.rope(q, cos, sin))
        w = self._query_shard(self.ccl.tp_all_reduce(ttnn.linear(x, self.weights_proj)))
        k = ttnn.to_layout(ttnn.slice(index_k, [0, 0, 0, 0], [1, 1, width, self.head_dim]), ttnn.TILE_LAYOUT)
        if self.ratio == 1:
            # V4.1's rule t < p + 1 is the kernel's causal mask: under seq_shard_axes=[] chip c (row-major (sp, tp)
            # order) starts its diagonal at chunk_start_idx + c * q_rows, which is this query split
            return self._score(q, k, w, chunk_start_idx=start), visible
        # ratio 2: the kernel scores every row (pad K by one zero tile and start its causal window at `width`,
        # which masks nothing real); rows before the chunk's own are visible to all its queries, and the chunk's
        # own rows follow the start-independent pattern of the tables' visibility tail
        k = ttnn.pad(k, [(0, 0), (0, 0), (0, 32), (0, 0)], 0.0)
        score = self._score(q, k, w, chunk_start_idx=width)
        score = ttnn.slice(ttnn.to_layout(score, ttnn.TILE_LAYOUT), [0, 0, 0, 0], [1, 1, q_rows, width])
        first = start // self.ratio
        score = self._add_tail(score, first, tables.visibility_tail(self.ratio, width - first))
        return ttnn.to_layout(score, ttnn.ROW_MAJOR_LAYOUT), visible

    def _score(self, q, k, w, chunk_start_idx: int):
        # all heads resident, 256-row key chunks (bead F10: 7.7-8x the default q32/k32/h1 config at 128K-1M)
        config = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(256, k.shape[2]), head_group_size=0)
        return ttnn.experimental.indexer_score_dsa(
            q, k, w, chunk_start_idx=chunk_start_idx, seq_shard_axes=[], program_config=config
        )

    def candidates(self, score, tables, start: int, q_rows: int, visible: int):
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
        # pin the block holding each query's newest visible row: block (start + i) // block of query i, i.e. the
        # tables' start-relative pin shifted by the chunk's first block
        assert start % (block * 32) == 0, f"chunk start {start} does not begin a tile of blocks"
        first = start // block
        blocks = self._add_tail(ttnn.to_layout(blocks, ttnn.TILE_LAYOUT), first, tables.pin_tail(nblocks - first))
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
        # scatter base and source filled on device (zeros_like / ones_like of a tiled tensor run ttnn.fill; a
        # ttnn.full on the device would write from the host)
        keep = ttnn.scatter(
            ttnn.to_layout(ttnn.zeros_like(blocks), ttnn.ROW_MAJOR_LAYOUT),
            -1,
            ttnn.typecast(idx, ttnn.int32),
            ttnn.to_layout(ttnn.ones_like(ttnn.slice(blocks, [0, 0, 0, 0], [1, 1, q_rows, k])), ttnn.ROW_MAJOR_LAYOUT),
        )
        keep = ttnn.slice(keep, [0, 0, 0, 0], [1, 1, q_rows, nblocks])
        keep = ttnn.repeat_interleave(ttnn.to_layout(keep, ttnn.TILE_LAYOUT), block, dim=-1)
        keep = ttnn.slice(keep, [0, 0, 0, 0], [1, 1, q_rows, width])
        # additive mask: 0 where kept, -inf elsewhere
        return ttnn.where(ttnn.gtz(keep), 0.0, float("-inf"))

    def forward(self, x, qr, index_k, tables, start: int, length: int, candidate_mask=None):
        """-> (top-k rows [1, 1, S/(sp*tp), k] uint32 row-major (sentinel tail), candidate mask or None)."""
        score, visible = self.scores(x, qr, index_k, tables, start, length)
        q_rows = score.shape[2]
        published = None
        if self.is_candidate_source:
            published = self.candidates(score, tables, start, q_rows, visible)
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
