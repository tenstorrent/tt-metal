# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 index keys, indexer, candidate selection and top-k (beads F4, F10; graph nodes B7, B9-B12).

Index keys (KV sources): ``k = k_norm(wk(latent))``, RoPE at each row's group-first position, FP4 QDQ
(block 32, ue8m0). Indexer (index sources): ``q = wq_b(qr)`` split into 32 heads, RoPE, FP4 QDQ;
``score[s, t] = sum_h relu(q_h . k_t) * w_h`` with ``w = weights_proj(x) * (d^-1/2 * H^-1/2)``; row t is visible
to the query at absolute position p iff ``t < (p + 1) // r``. The candidate source (layer 20) keeps, per query,
the 2048 blocks of 8 rows with the best max score, always including the block of its newest row, and publishes
their ids (``CandidateBlocks``); candidate index sources select only among those blocks' rows. Top-k keeps
``min(512, visible)`` rows; unreachable picks become the 0xFFFFFFFF sentinel, in one tail (descending score order).

Distribution: queries are split over SP and then TP (each chip scores ``S/(sp*tp)`` contiguous queries with all
32 heads), so nothing is reduced across chips and the selection comes out in the attention's head->sequence
query layout. Scores use ``indexer_score_dsa`` (ratio 1 relies on its token-causal mask; ratio 2 adds its own).
No host tables: position-dependent inputs are slices of the geometry's ``cache.V41ChunkTables`` (trace-safe).

Exact subset selection (bead F10, design R): with tie-free scores the k best elements of a row lie in its k best
blocks by block max, for any block size (k other blocks with larger maxima would hold k larger elements). So the
candidate source takes the top-2048 blocks of 8 among the blocks of its top-2048 superblocks of 32 rows (level 1
is skipped while a row has at most 2048 superblocks), and a candidate index source takes its top-512 rows among
the 16,384 rows of its candidate blocks. Two V4.1 kernels (``tt/v41/kernels``, launched through ``ttnn.generic_op``)
work on the row-major score in DRAM: ``segment_max`` (block or superblock maxima in one read of the row, the newest
row's block pinned to +inf) and ``gather_runs`` (a query's ranked blocks or superblocks, gathered from its row);
``tt/v41/indexer_kernels.py``.
"""

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import gather_runs, segment_max
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_ue8m0_qdq, fp8_qdq

SENTINEL = 0xFFFFFFFF
TOPK_MIN, TOPK_ALIGN = 16, 16
SUPERBLOCK = 32  # the candidate selection's first level: 32 rows (one 64-byte bf16 run), 4 candidate blocks
# ttnn.gather's row-major factory for index rows wider than 60 tiles gives every core a slice of each index row and
# re-reads the whole input row on each core; up to this width it splits rows over cores (measured on [640, 2048]
# uint32: 2.8 ms by a 2048-wide index vs 0.14 ms by a 1024-wide one)
GATHER_INDEX_WIDTH = 60 * 32
# The candidate source's own top-k runs among its candidate blocks' rows (exact: its top-512 rows lie in its top-512
# blocks, which are among the 2048 kept ones) above this width, where that is cheaper than a top-k over the row
# (LB 2x4, 640 rows at 56,320 columns: direct top-512 1.4 ms; gather_runs 0.85 ms + top-512 of 16,384 0.38 ms + the
# id mapping; the gather's sequential row read and the direct top-k both grow with the row)
SUBSET_TOPK_MIN_WIDTH = 1 << 16

WQ_B_GRID = (11, 10)  # the attention projections' 2D-multicast grid (tt/v41/attention.py MATMUL_GRID)
WQ_B_L1_BUDGET = 1 << 20  # bytes of output / fp32 partials / double-buffered in0 and in1 blocks per core


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


def _wq_b_program_config(m: int, k: int, n: int):
    """2D-multicast config of the [m, k] x [k, n] bf16 x bf16 fp32-accumulating wq_b GEMM on ``WQ_B_GRID``, or None
    (ttnn's default, 150 us at chunk 5120 on LB) when no in0 block width fits ``WQ_B_L1_BUDGET``."""
    mt, kt, nt = m // 32, k // 32, n // 32
    gx, gy = WQ_B_GRID
    per_m, per_n = -(-mt // gy), -(-nt // gx)
    tile = 2048
    for block_w in (8, 4, 2, 1):
        l1 = per_m * per_n * (tile + 2 * tile) + 2 * per_m * block_w * tile + 2 * block_w * per_n * tile
        if kt % block_w == 0 and l1 <= WQ_B_L1_BUDGET:
            break
    else:
        return None
    sub_w = max(w for w in range(1, 5) if per_n % w == 0)  # fp32 accumulation: at most 4 tiles per subblock
    sub_h = max(h for h in range(1, 5) if per_m % h == 0 and h * sub_w <= 4)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=WQ_B_GRID,
        in0_block_w=block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=per_m,
        per_core_N=per_n,
        transpose_mcast=False,
        fuse_batch=False,
        fused_activation=None,
    )


@dataclass(frozen=True)
class CandidateBlocks:
    """The candidate source's published selection (B11), consumed by the candidate index sources of the chunk.

    ``ids``: per query the kept blocks of ``CANDIDATE_BLOCK_SIZE`` rows, [1, 1, S/(sp*tp), CANDIDATE_TOPK_BLOCKS]
    uint32 row-major in descending block score (the pinned newest block first) with a 0xFFFFFFFF tail when fewer
    blocks are reachable; ``None`` when a row has at most CANDIDATE_TOPK_BLOCKS blocks, i.e. every visible block
    is a candidate and consumers select over their whole score."""

    ids: ttnn.Tensor | None


def _is_sentinel(t):
    """uint32 1 where ``t`` is the 0xFFFFFFFF sentinel, else 0 (valid ids are below 2^31)."""
    return ttnn.bitwise_right_shift(t, 31)


def _keep_sentinel(values, pos):
    """uint32 ``values`` with the sentinel wherever ``pos`` holds it (bitwise: ttnn.where does not select uint32)."""
    return ttnn.bitwise_or(values, ttnn.multiply(_is_sentinel(pos), SENTINEL))


def _gather_cols(t, index):
    """``ttnn.gather(t, -1, index)`` of row-major uint32 tensors, in index slices the row-splitting factory takes."""
    width = index.shape[3]
    if width <= GATHER_INDEX_WIDTH:
        return ttnn.gather(t, -1, index)
    rows = index.shape[2]
    parts = [
        ttnn.gather(t, -1, ttnn.slice(index, [0, 0, 0, a], [1, 1, rows, min(a + GATHER_INDEX_WIDTH, width)]))
        for a in range(0, width, GATHER_INDEX_WIDTH)
    ]
    return ttnn.concat(parts, dim=-1)


def _topk(score, k: int):
    """Row-major bf16 ``score`` [1, 1, R, W] -> [1, 1, R, k] uint32 row-major indices (``topk_large_indices``)."""
    k_pad = _round_up(max(k, TOPK_MIN), TOPK_ALIGN)
    idx = ttnn.experimental.topk_large_indices(score, k=k_pad)
    return idx if k_pad == k else ttnn.slice(idx, [0, 0, 0, 0], [1, 1, score.shape[2], k])


def _min(t, bound: int):
    """uint32 elementwise ``min(t, bound)``."""
    return _u32(ttnn.minimum(t, bound))


def _u32(t):
    """``t`` as uint32 (comparisons and scalar ops may return another dtype)."""
    return t if t.dtype == ttnn.uint32 else ttnn.typecast(t, ttnn.uint32)


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
        # both operands carry at most 4 significant bits (FP8 e4m3 with power-of-two block scales), so LoFi products
        # are exact: bit-identical to HiFi4 (measured on [640, 1024] x [1024, 8192]) at a quarter of the math passes
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(), math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True
        )
        scale = self.head_dim**-0.5 * self.heads**-0.5
        self.weights_proj = ttnn.from_torch(
            (weights["weights_proj"].detach().float() * scale).to(torch.bfloat16).T.contiguous()[None, None],
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, shape, dims=(None, 2)),  # row-parallel over hidden
        )
        if self.is_candidate_source or self.uses_candidates:
            assert (
                self.ratio == config.compress_ratio(config.CANDIDATE_SOURCE_LAYER) == 1
            ), "candidates are ratio-1 rows"
            assert SUPERBLOCK % config.CANDIDATE_BLOCK_SIZE == 0

    def _query_shard(self, t):
        """[1, *, S/sp, W] replicated across TP -> this chip's contiguous quarter of the SP shard."""
        return ttnn.mesh_partition(t, dim=2, cluster_axis=1) if self.tp > 1 else t

    def scores(self, x, qr, index_k, tables, start: int, length: int):
        """x [1,1,S/sp,hidden/tp] bf16, qr [1,1,S/sp,q_lora] bf16 (TP-replicated), index_k [1,1,T_max,d] tiled,
        ``tables`` the geometry's ``V41ChunkTables`` -> (scores [1, 1, S/(sp*tp), W] bf16 row-major DRAM-interleaved,
        -inf where not visible, W the visible rows rounded up to tiles (ratio 2: plus one -inf tile); visible rows)."""
        rows = qr.shape[2]
        q_rows = rows // self.tp
        assert start % self.ratio == 0 and (start // self.ratio) % 32 == 0, f"chunk start {start} not tile-aligned"
        visible = (start + length) // self.ratio
        width = _round_up(max(visible, TOPK_MIN), 32)
        cos, sin = (self._query_shard(t) for t in tables.rope(True, 1, start))
        q = ttnn.linear(
            fp8_qdq(self._query_shard(qr)),
            self.wq_b,
            program_config=_wq_b_program_config(q_rows, self.wq_b.shape[-2], self.wq_b.shape[-1]),
            compute_kernel_config=self.compute_kernel_config,
        )
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
        # own rows (and the pad tile) follow the start-independent pattern of the tables' visibility tail, which
        # is added to that region only and written back in place
        k = ttnn.pad(k, [(0, 0), (0, 0), (0, 32), (0, 0)], 0.0)
        score = self._score(q, k, w, chunk_start_idx=width)  # [1, 1, q_rows, width + 32]
        first, end = start // self.ratio, score.shape[3]
        tail = ttnn.to_layout(ttnn.slice(score, [0, 0, 0, first], [1, 1, q_rows, end]), ttnn.TILE_LAYOUT)
        tail = ttnn.add(tail, tables.visibility_tail(self.ratio, end - first))
        region = ([0, 0, 0, first], [1, 1, q_rows, end], [1, 1, 1, 1])
        ttnn.experimental.slice_write(ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT), score, *region)
        return score, visible

    def _score(self, q, k, w, chunk_start_idx: int):
        # all heads resident, 256-row key chunks (bead F10: 7.7-8x the default q32/k32/h1 config at 128K-1M)
        config = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(256, k.shape[2]), head_group_size=0)
        return ttnn.experimental.indexer_score_dsa(
            q, k, w, chunk_start_idx=chunk_start_idx, seq_shard_axes=[], program_config=config
        )

    # ---- selection (B11 candidates, B12 top-k) on a row-major score [1, 1, R, W] (W a multiple of 32) ----

    def candidates(self, score, tables, start: int) -> CandidateBlocks:
        """B11 on the candidate source's score: per query the CANDIDATE_TOPK_BLOCKS best blocks by max score, its
        newest row's block pinned in (``v41.select_candidate_blocks``)."""
        block, k = self.config.CANDIDATE_BLOCK_SIZE, self.config.CANDIDATE_TOPK_BLOCKS
        width = score.shape[3]
        if width // block <= k:
            return CandidateBlocks(None)
        assert start % SUPERBLOCK == 0, f"chunk start {start} does not begin a superblock"
        pin, every = tables.query_first(), 0xFFFFFFFF
        if width // SUPERBLOCK <= k:
            # one level: block maxima of the whole row (block ids in order), the newest row's block pinned to +inf
            return CandidateBlocks(_topk(segment_max(score, block, pin, start, every), k))
        # two levels: the top-k blocks lie in the top-k superblocks by max (the newest row's superblock pinned first);
        # their rows are gathered in rank order and ranked by block max, the newest row pinned again in rank 0 (at
        # column (start + q) % 32; chunk starts and per-chip query offsets are multiples of 32)
        per = SUPERBLOCK // block
        sb = _topk(segment_max(score, SUPERBLOCK, pin, start, every), k)  # sentinels gather -inf superblocks
        rows = gather_runs(score, sb, SUPERBLOCK)
        pos = ttnn.to_layout(_topk(segment_max(rows, block, pin, start, SUPERBLOCK - 1), k), ttnn.TILE_LAYOUT)
        rank = _min(ttnn.bitwise_right_shift(pos, per.bit_length() - 1), k - 1)  # pos = rank * per + sub-block
        c = ttnn.to_layout(_gather_cols(sb, ttnn.to_layout(rank, ttnn.ROW_MAJOR_LAYOUT)), ttnn.TILE_LAYOUT)
        ids = ttnn.add(ttnn.multiply(c, per), ttnn.bitwise_and(pos, per - 1))
        return CandidateBlocks(ttnn.to_layout(_keep_sentinel(ids, pos), ttnn.ROW_MAJOR_LAYOUT))

    def _topk_in_blocks(self, score, ids, k: int):
        """Top-k rows of ``score`` among each query's candidate blocks ``ids`` [1, 1, R, K] uint32 row-major (sentinel
        tail) -> [1, 1, R, k] uint32 row-major row ids (sentinel tail)."""
        block = self.config.CANDIDATE_BLOCK_SIZE
        rows = gather_runs(score, ids, block)  # rank r's rows at [r * block, (r + 1) * block); sentinels -inf
        if rows.shape[3] < k:  # fewer candidate rows than picks (small candidate counts): the rest are sentinels
            rows = ttnn.pad(rows, [(0, 0), (0, 0), (0, 0), (0, k - rows.shape[3])], float("-inf"))
        pos = ttnn.to_layout(_topk(rows, k), ttnn.TILE_LAYOUT)
        rank = _min(ttnn.bitwise_right_shift(pos, block.bit_length() - 1), ids.shape[3] - 1)
        chosen = ttnn.to_layout(_gather_cols(ids, ttnn.to_layout(rank, ttnn.ROW_MAJOR_LAYOUT)), ttnn.TILE_LAYOUT)
        row = ttnn.add(ttnn.multiply(chosen, block), ttnn.bitwise_and(pos, block - 1))
        return ttnn.to_layout(_keep_sentinel(row, pos), ttnn.ROW_MAJOR_LAYOUT)

    def select(self, score, tables, start: int, visible: int, candidates: CandidateBlocks | None = None):
        """B11-B12 on this index source's score (``scores``) -> (top-k rows [1, 1, R, k] uint32 row-major with a
        sentinel tail, the published ``CandidateBlocks`` of the candidate source or None)."""
        k = min(self.config.INDEX_TOPK, visible)
        published = None
        if self.is_candidate_source:
            published = self.candidates(score, tables, start)
            in_blocks = self.config.CANDIDATE_TOPK_BLOCKS > k  # its top-k rows lie in its top-k blocks
            if published.ids is not None and in_blocks and score.shape[3] > SUBSET_TOPK_MIN_WIDTH:
                return self._topk_in_blocks(score, published.ids, k), published
        elif self.uses_candidates:
            assert isinstance(candidates, CandidateBlocks), "a candidate index source needs the published candidates"
            if candidates.ids is not None:
                return self._topk_in_blocks(score, candidates.ids, k), None
        return _topk(score, k), published

    def forward(self, x, qr, index_k, tables, start: int, length: int, candidates: CandidateBlocks | None = None):
        """-> (top-k rows [1, 1, S/(sp*tp), k] uint32 row-major (sentinel tail), published CandidateBlocks or None)."""
        score, visible = self.scores(x, qr, index_k, tables, start, length)
        return self.select(score, tables, start, visible, candidates)
