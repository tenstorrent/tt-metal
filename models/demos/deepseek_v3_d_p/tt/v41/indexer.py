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

Candidate scoring (bead F10): a candidate index source's score is observable only in its candidate blocks (the
reference masks the rest to -inf), so on long rows it scores only those rows (``candidate_scores``: per query its
16,384 candidate index-K rows gathered from the cache) instead of the full row; on short rows the full-row score and
a gather of its candidate blocks are cheaper (``SPARSE_SCORE_MIN_WIDTH``).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.indexer_kernels import (
    PIN_PAGE,
    RANGE_CHUNK,
    candidate_scores,
    gather_runs,
    segment_max,
)
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_ue8m0_qdq, fp8_qdq

SENTINEL = 0xFFFFFFFF
TOPK_MIN, TOPK_ALIGN = 16, 16
SUPERBLOCK = 32  # the candidate selection's first level: 32 rows (one 64-byte bf16 run), 4 candidate blocks
# ttnn.gather's row-major factory for index rows wider than 60 tiles gives every core a slice of each index row and
# re-reads the whole input row on each core; up to this width it splits rows over cores (measured on [640, 2048]
# uint32: 2.8 ms by a 2048-wide index vs 0.14 ms by a 1024-wide one)
GATHER_INDEX_WIDTH = 60 * 32
# The candidate source's own top-k runs among the rows of its best candidate blocks above this width, where that is
# cheaper than a top-k over the row. Exact: its top-512 rows lie in its top-512 blocks by max, which are among the
# first 513 published ids (descending block max after the pinned block), so only those blocks are gathered
# (LB 2x4, 640 rows at 56,320 columns: direct top-512 1.43 ms; gather_runs of all 2048 blocks 0.62 ms + top-512 of
# 16,384 0.31 ms + the id mapping; the gather's sequential row read and the direct top-k both grow with the row)
SUBSET_TOPK_MIN_WIDTH = 1 << 15
# A candidate index source scores only its candidate rows (``candidate_scores``: a per-query gather of its 16,384
# index-K rows, a fixed ~2.7 GB per chip at 640 queries, plus a relayout of the visible index-K) from this score width
# up; below it the full-row score (index-K read once per query tile-row) and ``gather_runs`` of its candidate blocks are
# cheaper. Measured, LB 2x4 traced L24 indexer at chunk start 51,200 / 262,144 / 512,000: candidate scoring
# 8.22 / 9.72 / 11.12 ms, full row 3.04 / 8.47 / 15.35 ms (crossover ~310K columns)
SPARSE_SCORE_MIN_WIDTH = 5 << 16

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


@dataclass(frozen=True)
class DecodePins:
    """What ``select`` needs of ``V41ChunkTables`` for the decode tile: a per-row pin page ``[1, 1, 1, ROWS]`` uint32
    (``indexer_kernels._pin_rows``), entry ``i`` the newest compressed row of row ``i``'s user in the scored source
    (its visible count less one; the sentinel when the row is idle, which pins nothing). ``start`` is 0 with it: a
    row's pin is its absolute row."""

    newest: ttnn.Tensor

    def query_first(self):
        return self.newest

    def pin(self, start: int):
        """(pin page, pin_base) as ``V41ChunkTables.pin``: a row's pin is its absolute row, so ``start`` is 0."""
        assert start == 0, start
        return self.newest, 0


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
        self._split_consts = {}  # the split selection's id constants (select_split)
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

    def _width(self, start: int, length: int, tables=None) -> tuple[int, int]:
        """(visible rows, score width: the visible rows rounded up to tiles; in generic mode the bucket width of
        ``tables``, the same for every chunk of the bucket)."""
        assert start % self.ratio == 0 and (start // self.ratio) % 32 == 0, f"chunk start {start} not tile-aligned"
        visible = (start + length) // self.ratio
        if tables is not None and tables.generic:
            return visible, tables.bucket_width(self.ratio, start, length)
        return visible, _round_up(max(visible, TOPK_MIN), 32)

    def _query(self, x, qr, tables, start: int):
        """This chip's queries: (q [1, heads, S/(sp*tp), d] bf16 tiled after RoPE and QDQ, scaled head weights
        w [1, 1, S/(sp*tp), heads] bf16 tiled)."""
        q_rows = qr.shape[2] // self.tp
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
        return q, self._query_shard(self.ccl.tp_all_reduce(ttnn.linear(x, self.weights_proj)))

    def scores(self, x, qr, index_k, tables, start: int, length: int, index_k_tiled=None):
        """x [1,1,S/sp,hidden/tp] bf16, qr [1,1,S/sp,q_lora] bf16 (TP-replicated), index_k [1,1,T_max,d] row-major,
        ``tables`` the geometry's ``V41ChunkTables`` -> (scores [1, 1, S/(sp*tp), W] bf16 row-major DRAM-interleaved,
        -inf where not visible, W the visible rows rounded up to tiles (ratio 2: plus one -inf tile); visible rows).
        ``index_k_tiled``: the state's tiled twin of ``index_k`` with a zero tile past its rows
        (``V41PrefillState.index_k_tiled``), read instead of ``index_k`` by the generic path under the chunk's shared
        masks (``V41ChunkTables.chunk_mask``)."""
        visible, width = self._width(start, length, tables)
        q, w = self._query(x, qr, tables, start)
        q_rows = q.shape[2]
        mask = tables.chunk_mask(self.ratio, width + 32) if tables.generic else None
        if mask is not None and index_k_tiled is not None:
            # the tiled prefix with its pad tile in one slice (TT_V41_PREFILL_MASK_ONCE)
            k = ttnn.slice(index_k_tiled, [0, 0, 0, 0], [1, 1, width + 32, self.head_dim])
        else:
            k = ttnn.to_layout(ttnn.slice(index_k, [0, 0, 0, 0], [1, 1, width, self.head_dim]), ttnn.TILE_LAYOUT)
            if tables.generic:
                k = ttnn.pad(k, [(0, 0), (0, 0), (0, 32), (0, 0)], 0.0)
        if tables.generic:
            # position-generic: the kernel's causal window masks nothing real (it starts at ``width``, past every
            # row; K is padded by one zero tile for it), and the rows a query may not see (at or past its visible
            # count, a device input) are masked over the whole score: one add, the same graph at every start
            score = self._score(q, k, w, chunk_start_idx=width)  # [1, 1, q_rows, width + 32]
            if mask is not None:
                # the chunk's shared row-major mask (TT_V41_PREFILL_MASK_ONCE): the add runs row-major on the
                # kernel's output, no layout round trip of the score, no mask build a layer
                masked = ttnn.add(score, mask)
                ttnn.deallocate(score)
                return masked, visible
            masked = ttnn.add(ttnn.to_layout(score, ttnn.TILE_LAYOUT), tables.mask(self.ratio, score.shape[3]))
            ttnn.deallocate(score)
            return ttnn.to_layout(masked, ttnn.ROW_MAJOR_LAYOUT), visible
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
        (pin, pin_base), every = tables.pin(start), 0xFFFFFFFF
        if width // SUPERBLOCK <= k:
            # one level: block maxima of the whole row (block ids in order), the newest row's block pinned to +inf
            return CandidateBlocks(_topk(segment_max(score, block, pin, pin_base, every), k))
        # two levels: the top-k blocks lie in the top-k superblocks by max (the newest row's superblock pinned first);
        # their rows are gathered in rank order and ranked by block max, the newest row pinned again in rank 0 (at
        # column (start + q) % 32; chunk starts and per-chip query offsets are multiples of 32)
        per = SUPERBLOCK // block
        sb = _topk(segment_max(score, SUPERBLOCK, pin, pin_base, every), k)  # sentinels gather -inf superblocks
        rows = gather_runs(score, sb, SUPERBLOCK)
        pos = ttnn.to_layout(_topk(segment_max(rows, block, pin, pin_base, SUPERBLOCK - 1), k), ttnn.TILE_LAYOUT)
        rank = _min(ttnn.bitwise_right_shift(pos, per.bit_length() - 1), k - 1)  # pos = rank * per + sub-block
        c = ttnn.to_layout(_gather_cols(sb, ttnn.to_layout(rank, ttnn.ROW_MAJOR_LAYOUT)), ttnn.TILE_LAYOUT)
        ids = ttnn.add(ttnn.multiply(c, per), ttnn.bitwise_and(pos, per - 1))
        return CandidateBlocks(ttnn.to_layout(_keep_sentinel(ids, pos), ttnn.ROW_MAJOR_LAYOUT))

    def candidate_rows(self, x, qr, index_k, tables, start: int, ids):
        """A candidate index source's score over its queries' candidate rows only (``candidate_scores``): [1, 1, R,
        K * block] bf16 row-major, rank r's rows at [r * block, (r + 1) * block), -inf for sentinels and rows past the
        query; the layout ``gather_runs`` makes of the full score."""
        assert self.uses_candidates and self.ratio == 1
        q, w = self._query(x, qr, tables, start)
        q = ttnn.to_layout(ttnn.experimental.nlp_concat_heads(q), ttnn.ROW_MAJOR_LAYOUT)
        w = ttnn.to_layout(w, ttnn.ROW_MAJOR_LAYOUT)
        _, width = self._width(start, qr.shape[2] * self.sp, tables)
        pin, pin_base = tables.pin(start)
        return candidate_scores(q, w, index_k, ids, pin, pin_base, width)

    def decode_query(self, q, w):
        """The decode tile's query and head weights (``q`` [1, heads, ROWS, 128] and ``w`` [1, 1, ROWS, heads] as
        the step builds them) in ``candidate_scores``'s row-major forms."""
        return ttnn.to_layout(ttnn.experimental.nlp_concat_heads(q), ttnn.ROW_MAJOR_LAYOUT), ttnn.to_layout(
            w, ttnn.ROW_MAJOR_LAYOUT
        )

    def select_user_in_blocks(self, q_rm, w_rm, blocks, newest, ids, user: int, rows: int, block_base: int, k: int):
        """A candidate index source's top-k for one user of the decode tile, scoring its candidate rows only
        (``candidate_scores`` over the pool's block pages) -> [1, 1, 1, k] uint32 row-major row ids (sentinel tail).

        ``q_rm``, ``w_rm``: ``decode_query``; ``blocks``: the pool's block pages; ``newest`` [1, 1, 1, ROWS] the
        tile's per-row pins (or [1, 1, 1, 1], the user's own); ``ids`` [1, 1, ROWS, K] the blocks the candidate
        source published against this user's keys (row ``user`` the user's own; or [1, 1, 1, K], the user's);
        ``rows`` the user's rows in the pool (the candidate scorer's visible bound; its pin masks the rows past the
        user's newest); ``block_base`` the page of the user's block 0. The scorer works a query row per core, so the
        user's K blocks are split over ROWS rows of K / ROWS blocks of the same query (ROWS cores, a ROWS-th of the
        traffic of scoring every tile row against this user), and the ROWS x (K / ROWS x 8) scores are one row again
        for the top-k."""
        assert self.uses_candidates and self.ratio == 1
        n = q_rm.shape[2]
        row = [0, 0, user, 0]
        q_u = ttnn.slice(q_rm, row, [1, 1, user + 1, q_rm.shape[3]])
        w_u = ttnn.slice(w_rm, row, [1, 1, user + 1, w_rm.shape[3]])
        # ids published for the whole tile (row user the user's own) or for the user alone (the single-query path); likewise the tile's pin page or the user's one-entry page
        ids_u = ids if ids.shape[2] == 1 else ttnn.slice(ids, row, [1, 1, user + 1, ids.shape[3]])  # [1, 1, 1, K]
        pin_u = newest if newest.shape[3] == 1 else ttnn.slice(newest, [0, 0, 0, user], [1, 1, 1, user + 1])
        q_n, w_n = ttnn.concat([q_u] * n, dim=2), ttnn.concat([w_u] * n, dim=2)
        pin_n = ttnn.concat([pin_u] * n, dim=3)
        return self._select_rows_in_blocks(q_n, w_n, blocks, pin_n, ids_u, rows, block_base, k)

    def select_live_in_blocks(self, q_n, w_n, blocks, pin_n, ids_u, base_page, rows: int, k: int):
        """``select_user_in_blocks`` for the one-live-row capture: the live user's query and
        head weights already repeated over the ROWS rows (``q_n`` [1, 1, ROWS, 32 * 128], ``w_n`` [1, 1, ROWS, 32]
        bf16 row-major, a one-hot matmul of the tile), its pin repeated (``pin_n`` [1, 1, 1, ROWS] uint32 row-major),
        its published blocks ``ids_u`` [1, 1, 1, K] and the page of its block 0 as word 1 of ``base_page`` (a
        per-step host row), so the capture carries no slot."""
        assert self.uses_candidates and self.ratio == 1
        return self._select_rows_in_blocks(q_n, w_n, blocks, pin_n, ids_u, rows, 0, k, base_page=base_page)

    def _select_rows_in_blocks(
        self, q_n, w_n, blocks, pin_n, ids_u, rows: int, block_base: int, k: int, base_page=None
    ):
        n = q_n.shape[2]
        per = ids_u.shape[3] // n
        assert per * n == ids_u.shape[3] and (per * self.config.CANDIDATE_BLOCK_SIZE) % 32 == 0, (tuple(ids_u.shape), n)
        ids_n = ttnn.reshape(ids_u, [1, 1, n, per])  # row r: blocks [r * per, (r + 1) * per) of the user's
        scores = candidate_scores(
            q_n, w_n, blocks, ids_n, pin_n, 0, rows, block_base=block_base, base_page=base_page
        )  # [1, 1, n, per*8]
        rows_u = ttnn.reshape(scores, [1, 1, 1, n * per * self.config.CANDIDATE_BLOCK_SIZE])
        # the top-k kernels take a tile of rows: the user's row first, the rest -inf (sentinel picks)
        rows_n = ttnn.pad(rows_u, [(0, 0), (0, 0), (0, n - 1), (0, 0)], float("-inf"))
        ids_rep = ttnn.concat([ids_u] * n, dim=2)
        return ttnn.slice(self._topk_in_rows(rows_n, ids_rep, k), [0, 0, 0, 0], [1, 1, 1, k])

    def _topk_in_blocks(self, score, ids, k: int):
        """Top-k rows of ``score`` among each query's candidate blocks ``ids`` [1, 1, R, K] uint32 row-major (sentinel
        tail) -> [1, 1, R, k] uint32 row-major row ids (sentinel tail)."""
        # rank r's rows at [r * block, (r + 1) * block); sentinels -inf
        return self._topk_in_rows(gather_runs(score, ids, self.config.CANDIDATE_BLOCK_SIZE), ids, k)

    def _topk_in_rows(self, rows, ids, k: int):
        """Top-k of the candidate ``rows`` [1, 1, R, K * block] (rank r's block at [r * block, (r + 1) * block)) ->
        [1, 1, R, k] uint32 row-major row ids (sentinel tail)."""
        block = self.config.CANDIDATE_BLOCK_SIZE
        if rows.shape[3] < k:  # fewer candidate rows than picks (small candidate counts): the rest are sentinels
            rows = ttnn.pad(rows, [(0, 0), (0, 0), (0, 0), (0, k - rows.shape[3])], float("-inf"))
        pos = ttnn.to_layout(_topk(rows, k), ttnn.TILE_LAYOUT)
        rank = _min(ttnn.bitwise_right_shift(pos, block.bit_length() - 1), ids.shape[3] - 1)
        chosen = ttnn.to_layout(_gather_cols(ids, ttnn.to_layout(rank, ttnn.ROW_MAJOR_LAYOUT)), ttnn.TILE_LAYOUT)
        row = ttnn.add(ttnn.multiply(chosen, block), ttnn.bitwise_and(pos, block - 1))
        return ttnn.to_layout(_keep_sentinel(row, pos), ttnn.ROW_MAJOR_LAYOUT)

    # ---- the split layout's selection (range_scores: one user's score as rows of chunks) ----

    def _split_const(self, name: str, shape: tuple, fill):
        """A memoized replicated uint32 tiled constant (built on the first eager step, before any capture)."""
        key = (name,) + tuple(shape)
        if key not in self._split_consts:
            self._split_consts[key] = ttnn.from_torch(
                fill(),
                device=self.mesh_device,
                dtype=ttnn.uint32,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self._split_consts[key]

    def _merge_split(self, rows, k: int, per_chunk: int):
        """Top-k over a split layout ``rows`` [1, 1, R, Wr] bf16 row-major (element (r, j) is global
        ``((j // per_chunk) * R + r) * per_chunk + j % per_chunk``, R a power of two) -> [1, 1, 1, k] uint32
        row-major global ids in descending score (sentinel tail for -inf). Per row a top-k on its own core, then one
        top-k over the R x k_r picks' values."""
        return self._merge_split_rows(rows, 1, k, per_chunk)

    def _merge_split_rows(self, rows, n_live: int, k: int, per_chunk: int):
        """``_merge_split`` over ``n_live`` split layouts stacked along dim 2 (``rows`` [1, 1, L * R, Wr]: layout
        ``l`` at rows ``l * R .. (l + 1) * R - 1``) in one chain of ops -> [1, 1, L, k] uint32 row-major, row ``l``
        the global top-k of layout ``l`` (the verify rows of speculative decode merge their selections together: the chain's op count does not grow with the rows)."""
        total, width = rows.shape[2], rows.shape[3]
        n_rows = total // n_live
        assert n_rows * n_live == total, (total, n_live)
        if n_rows == 1:  # one row a layout: its top-k is the global one (slot 0 of row 0 is key 0)
            return _topk(rows, k)
        kr = min(k, width)
        assert kr >= TOPK_MIN and width >= _round_up(kr, TOPK_ALIGN), (tuple(rows.shape), k)
        local = _topk(rows, kr)  # [1, 1, L * R, kr] row-major
        local_t = ttnn.to_layout(local, ttnn.TILE_LAYOUT)
        safe = _min(local_t, width - 1)
        safe_rm = ttnn.to_layout(safe, ttnn.ROW_MAJOR_LAYOUT)
        values = ttnn.to_layout(_gather_cols(rows, safe_rm), ttnn.TILE_LAYOUT)  # [1, 1, L * R, kr] bf16
        picked = ttnn.typecast(_is_sentinel(local_t), ttnn.bfloat16)
        values = ttnn.where(picked, float("-inf"), values)  # a sentinel pick's value is -inf whatever the column
        # the global id of (r, j): a shift and a mask per power of two
        shift, row_shift = per_chunk.bit_length() - 1, n_rows.bit_length() - 1
        row_ids = self._split_const(
            f"row{n_rows}",
            (total, kr),
            lambda: (torch.arange(total, dtype=torch.int32) % n_rows)
            .view(1, 1, total, 1)
            .expand(1, 1, total, kr)
            .contiguous(),
        )
        slot = ttnn.bitwise_right_shift(safe, shift)
        within = ttnn.bitwise_and(safe, per_chunk - 1)
        chunk = ttnn.add(ttnn.bitwise_left_shift(slot, row_shift), row_ids)
        glob = _keep_sentinel(ttnn.add(ttnn.bitwise_left_shift(chunk, shift), within), local_t)
        flat_values = ttnn.reshape(ttnn.to_layout(values, ttnn.ROW_MAJOR_LAYOUT), [1, 1, n_live, n_rows * kr])
        flat_ids = ttnn.reshape(ttnn.to_layout(glob, ttnn.ROW_MAJOR_LAYOUT), [1, 1, n_live, n_rows * kr])
        pos = _topk(flat_values, k)  # [1, 1, L, k]
        pos_t = ttnn.to_layout(pos, ttnn.TILE_LAYOUT)
        pos_safe = ttnn.to_layout(_min(pos_t, n_rows * kr - 1), ttnn.ROW_MAJOR_LAYOUT)
        chosen = ttnn.to_layout(_gather_cols(flat_ids, pos_safe), ttnn.TILE_LAYOUT)
        return ttnn.to_layout(_keep_sentinel(chosen, pos_t), ttnn.ROW_MAJOR_LAYOUT)

    def _split_pins(self, newest, n_rows: int, per_chunk: int):
        """The per-row pin page [1, 1, 1, R] uint32 row-major of a split layout for the user's ``newest`` [1, 1, 1, 1]:
        the row holding the newest key gets its local column, every other row the sentinel (an idle user's sentinel
        lands past every row: no pin)."""
        if n_rows == 1:
            return newest  # the user's own one-entry page
        n_pins = max(n_rows, 2 * PIN_PAGE)  # a page of exactly PIN_PAGE entries would read as a query_first page
        return self._split_pins_page(ttnn.concat([newest] * n_pins, dim=3), n_rows, per_chunk)

    def _split_pins_page(self, newest, n_rows: int, per_chunk: int):
        """``_split_pins`` from a page ``newest`` [1, 1, 1, P] uint32 row-major that already holds every split row's
        user newest key (P a multiple of R, not PIN_PAGE: stacked layouts, each row ``l * R + r`` its layout's
        newest) -> the per-row pin page [1, 1, 1, P]."""
        n_pins = newest.shape[3]
        assert n_pins % n_rows == 0 and n_pins != PIN_PAGE, (n_pins, n_rows)
        shift, row_shift = per_chunk.bit_length() - 1, n_rows.bit_length() - 1
        pins = ttnn.to_layout(newest, ttnn.TILE_LAYOUT)  # [1, 1, 1, P]
        chunk = ttnn.bitwise_right_shift(pins, shift)
        row = ttnn.bitwise_and(chunk, n_rows - 1)
        local = ttnn.add(
            ttnn.bitwise_left_shift(ttnn.bitwise_right_shift(chunk, row_shift), shift),
            ttnn.bitwise_and(pins, per_chunk - 1),
        )
        rows = self._split_const(
            f"rowpin{n_rows}",
            (1, n_pins),
            lambda: (torch.arange(n_pins, dtype=torch.int32) % n_rows).view(1, 1, 1, n_pins),
        )
        other = _u32(ttnn.ne(row, rows))  # 1 where the row is not the newest key's
        return ttnn.to_layout(ttnn.bitwise_or(local, ttnn.multiply(other, SENTINEL)), ttnn.ROW_MAJOR_LAYOUT)

    def select_split_rows(self, stacked, newest, n_live: int, width: int, k: int, candidates=None):
        """``select_split`` over ``n_live`` users' (or verify rows') split layouts stacked along dim 2 (``stacked``
        [1, 1, L * R, Wr] bf16 row-major; ``newest`` [1, 1, 1, L * R] uint32 row-major, entry ``l * R + r`` layout
        ``l``'s newest key) in one chain of ops -> (top-k rows [1, 1, L, k], the published blocks [1, 1, L, kb] of
        the candidate source as one ``CandidateBlocks`` or None); row ``l`` of each is layout ``l``'s."""
        n_rows = stacked.shape[2] // n_live
        assert newest.shape[3] != PIN_PAGE, "a page of PIN_PAGE entries would read as a query_first page"
        assert n_rows * n_live == stacked.shape[2] and newest.shape[3] == stacked.shape[2], (
            tuple(stacked.shape),
            tuple(newest.shape),
            n_live,
        )
        chunk_cols = RANGE_CHUNK * 32
        block = self.config.CANDIDATE_BLOCK_SIZE
        published = None
        if self.is_candidate_source:
            kb = self.config.CANDIDATE_TOPK_BLOCKS
            if width // block > kb:
                pins = newest if n_rows == 1 else self._split_pins_page(newest, n_rows, chunk_cols)
                maxima = segment_max(stacked, block, pins, 0, 0xFFFFFFFF)  # [1, 1, L * R, Wr / block]
                assert maxima.shape[3] == stacked.shape[3] // block, (tuple(maxima.shape), tuple(stacked.shape))
                published = CandidateBlocks(self._merge_split_rows(maxima, n_live, kb, chunk_cols // block))
            else:
                published = CandidateBlocks(None)
        elif self.uses_candidates:
            assert isinstance(candidates, CandidateBlocks) and candidates.ids is None, "published blocks: in blocks"
        return self._merge_split_rows(stacked, n_live, k, chunk_cols), published

    def select_split(self, rows, newest, width: int, k: int, candidates: CandidateBlocks | None = None):
        """B11-B12 on one user's score in the split layout (``range_scores`` of ``width`` columns) -> (top-k rows
        [1, 1, 1, k] uint32 row-major with a sentinel tail, the published ``CandidateBlocks`` [1, 1, 1,
        CANDIDATE_TOPK_BLOCKS] of the candidate source or None), the semantics of ``select``: the candidate source
        publishes its best blocks by maximum with the newest key's block pinned (``segment_max`` per row with the
        pin in the row holding it, the rows merged), and its top-k is the dense top-k (which lies in those blocks);
        a consumer with published blocks does not come here (``select_user_in_blocks``); everything else is the
        dense top-k of the row. ``newest`` [1, 1, 1, 1] uint32 row-major: the user's newest key (the sentinel for
        none)."""
        n_rows, row_width = rows.shape[2], rows.shape[3]
        chunk_cols = RANGE_CHUNK * 32
        block = self.config.CANDIDATE_BLOCK_SIZE
        published = None
        if self.is_candidate_source:
            kb = self.config.CANDIDATE_TOPK_BLOCKS
            if width // block > kb:
                pins = self._split_pins(newest, n_rows, chunk_cols)
                maxima = segment_max(rows, block, pins, 0, 0xFFFFFFFF)  # [1, 1, R, Wr / block]
                assert maxima.shape[3] == row_width // block, (tuple(maxima.shape), row_width)
                published = CandidateBlocks(self._merge_split(maxima, kb, chunk_cols // block))
            else:
                published = CandidateBlocks(None)
        elif self.uses_candidates:
            assert isinstance(candidates, CandidateBlocks) and candidates.ids is None, "published blocks: in blocks"
        return self._merge_split(rows, k, chunk_cols), published

    def select(self, score, tables, start: int, visible: int, candidates: CandidateBlocks | None = None):
        """B11-B12 on this index source's score (``scores``) -> (top-k rows [1, 1, R, k] uint32 row-major with a
        sentinel tail, the published ``CandidateBlocks`` of the candidate source or None)."""
        k = min(self.config.INDEX_TOPK, visible)
        published = None
        if self.is_candidate_source:
            published = self.candidates(score, tables, start)
            in_blocks = self.config.CANDIDATE_TOPK_BLOCKS > k  # its top-k rows lie in its top-k blocks
            if published.ids is not None and in_blocks and score.shape[3] > SUBSET_TOPK_MIN_WIDTH:
                best = _round_up(k + 1, 4)  # the pinned block + its top-k blocks, whole tiles of candidate rows
                ids = ttnn.slice(published.ids, [0, 0, 0, 0], [1, 1, published.ids.shape[2], best])
                return self._topk_in_blocks(score, ids, k), published
        elif self.uses_candidates:
            assert isinstance(candidates, CandidateBlocks), "a candidate index source needs the published candidates"
            if candidates.ids is not None:
                return self._topk_in_blocks(score, candidates.ids, k), None
        return _topk(score, k), published

    def forward(
        self,
        x,
        qr,
        index_k,
        tables,
        start: int,
        length: int,
        candidates: CandidateBlocks | None = None,
        index_k_tiled=None,
    ):
        """-> (top-k rows [1, 1, S/(sp*tp), k] uint32 row-major (sentinel tail), published CandidateBlocks or None)."""
        visible, width = self._width(start, length, tables)
        if self.uses_candidates and candidates.ids is not None and width >= SPARSE_SCORE_MIN_WIDTH:
            rows = self.candidate_rows(x, qr, index_k, tables, start, candidates.ids)
            return self._topk_in_rows(rows, candidates.ids, min(self.config.INDEX_TOPK, visible)), None
        score, visible = self.scores(x, qr, index_k, tables, start, length, index_k_tiled)
        return self.select(score, tables, start, visible, candidates)
