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
the 16,384 rows of its candidate blocks, gathered from its dense score. Superblocks are 64-byte bf16 runs, gathered
by ``ttnn.embedding`` from a zero-copy 64-byte-row view of the row-major score (``_gather_superblocks``).
"""

from dataclasses import dataclass

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.deepseek_v3_d_p.tt.mla.rope import get_rot_transformation_mat
from models.demos.deepseek_v3_d_p.tt.v41.cache import SUPERBLOCK, dram_banks
from models.demos.deepseek_v3_d_p.tt.v41.ccl import V41Collectives
from models.demos.deepseek_v3_d_p.tt.v41.qdq import fp4_ue8m0_qdq, fp8_qdq

SENTINEL = 0xFFFFFFFF
TOPK_MIN, TOPK_ALIGN = 16, 16
# ttnn.gather's row-major factory for index rows wider than 60 tiles gives every core a slice of each index row and
# re-reads the whole input row on each core; up to this width it splits rows over cores (measured on [640, 2048]
# uint32: 2.8 ms by a 2048-wide index vs 0.14 ms by a 1024-wide one)
GATHER_INDEX_WIDTH = 60 * 32


def _round_up(x: int, m: int) -> int:
    return -(-x // m) * m


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


def _split_major(pos, n: int, parts: int):
    """uint32 tiled positions ``pos = j * n + c`` (j < parts, c < n) -> (j, c); a sentinel gives j = parts - 1 and a
    large c (callers clamp c and restore the sentinel)."""
    j = _u32(ttnn.ge(pos, n))
    for m in range(2, parts):
        j = ttnn.add(j, _u32(ttnn.ge(pos, m * n)))
    return j, _u32(ttnn.subtract(pos, ttnn.multiply(j, n)))


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
        self.banks = dram_banks(mesh_device)
        self.subblock_pattern = None
        if self.is_candidate_source or self.uses_candidates:
            assert (
                self.ratio == config.compress_ratio(config.CANDIDATE_SOURCE_LAYER) == 1
            ), "candidates are ratio-1 rows"
            block, blocks = config.CANDIDATE_BLOCK_SIZE, config.CANDIDATE_TOPK_BLOCKS
            assert SUPERBLOCK % block == 0
            # column u of the gathered candidate superblocks lies in sub-block (u % 32) // block of its superblock
            pattern = (torch.arange(blocks * SUPERBLOCK) % SUPERBLOCK) // block
            self.subblock_pattern = ttnn.from_torch(
                pattern.to(torch.bfloat16).reshape(1, 1, 1, -1),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=rep,
            )

    def _query_shard(self, t):
        """[1, *, S/sp, W] replicated across TP -> this chip's contiguous quarter of the SP shard."""
        return ttnn.mesh_partition(t, dim=2, cluster_axis=1) if self.tp > 1 else t

    @staticmethod
    def _add_cols(t, first: int, cols):
        """Tiled ``t`` [1, 1, R, W] with the additive ``cols`` [1, 1, R, w] added to its columns [first, first + w)."""
        rows, width, w = t.shape[2], t.shape[3], cols.shape[3]
        pieces = [ttnn.slice(t, [0, 0, 0, 0], [1, 1, rows, first])] if first else []
        pieces.append(ttnn.add(ttnn.slice(t, [0, 0, 0, first], [1, 1, rows, first + w]), cols))
        if first + w < width:
            pieces.append(ttnn.slice(t, [0, 0, 0, first + w], [1, 1, rows, width]))
        return pieces[0] if len(pieces) == 1 else ttnn.concat(pieces, dim=-1)

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
        score = self._add_cols(score, first, tables.visibility_tail(self.ratio, width - first))
        return ttnn.to_layout(score, ttnn.ROW_MAJOR_LAYOUT), visible

    def _score(self, q, k, w, chunk_start_idx: int):
        # all heads resident, 256-row key chunks (bead F10: 7.7-8x the default q32/k32/h1 config at 128K-1M)
        config = ttnn.IndexerScoreProgramConfig(q_chunk_size=32, k_chunk_size=min(256, k.shape[2]), head_group_size=0)
        return ttnn.experimental.indexer_score_dsa(
            q, k, w, chunk_start_idx=chunk_start_idx, seq_shard_axes=[], program_config=config
        )

    # ---- selection (B11 candidates, B12 top-k) on a row-major score [1, 1, R, W] (W a multiple of 32) ----

    def _gather_superblocks(self, score, superblocks, tables):
        """Row i of the output holds ``score[i, 32 c : 32 c + 32]`` for its superblocks c = ``superblocks[i, r]``
        (r < k): ``score`` [1, 1, R, W] bf16 row-major DRAM-interleaved, ``superblocks`` [1, 1, R, k] uint32 tiled
        (< W / 32) -> [1, 1, R, 32 k] bf16 row-major.

        Zero-copy view: a row-major interleaved tensor stores page (row) p in DRAM bank p % n at offset (p // n) * its
        page size, so the same buffer read with 64-byte pages holds superblock c of row i at page
        ``((i // n) * W/32 + c) * n + i % n``; ``ttnn.embedding`` gathers those pages. Output page ``((i // n) * k + r)
        * n + i % n`` gets superblock r of row i, which is where the [R, 32 k] view of the output keeps it."""
        rows, width, k, n = score.shape[2], score.shape[3], superblocks.shape[3], self.banks
        memory = score.memory_config()
        assert score.layout == ttnn.ROW_MAJOR_LAYOUT and score.dtype == ttnn.bfloat16 and width % SUPERBLOCK == 0
        assert memory.buffer_type == ttnn.BufferType.DRAM and not memory.is_sharded(), "needs a DRAM-interleaved score"
        nsb = width // SUPERBLOCK
        view = ttnn.experimental.view(score, [1, 1, rows * nsb, SUPERBLOCK])
        page = ttnn.add(ttnn.multiply(superblocks, n), tables.row_view_base(nsb))
        page = ttnn.transpose(ttnn.reshape(page, [rows // n, n, k]), -2, -1)  # output page order
        page = ttnn.to_layout(ttnn.reshape(page, [1, 1, 1, rows * k]), ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.embedding(page, view, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return ttnn.experimental.view(out, [1, 1, rows, SUPERBLOCK * k])

    def _block_max(self, s):
        """Tiled ``s`` [1, 1, R, w] -> tiled [1, 1, R, w / block]: max of each block of ``CANDIDATE_BLOCK_SIZE``
        columns, sub-block major (column j * w/32 + c holds block j of superblock c)."""
        rows, width = s.shape[2], s.shape[3]
        block, nsb = self.config.CANDIDATE_BLOCK_SIZE, width // SUPERBLOCK
        t = ttnn.reshape(ttnn.transpose(s, -2, -1), [1, nsb, SUPERBLOCK, rows])  # superblock c = batch c
        parts = []
        for j in range(SUPERBLOCK // block):
            part = ttnn.max(ttnn.slice(t, [0, 0, j * block, 0], [1, nsb, (j + 1) * block, rows]), dim=-2)
            parts.append(ttnn.transpose(ttnn.reshape(part, [1, 1, nsb, rows]), -2, -1))
        return ttnn.concat(parts, dim=-1)

    def _superblock_max(self, s):
        """Tiled ``s`` [1, 1, R, W] -> tiled [1, 1, R, W / 32]: max of each superblock of 32 columns."""
        rows, nsb = s.shape[2], s.shape[3] // SUPERBLOCK
        t = ttnn.reshape(ttnn.transpose(s, -2, -1), [1, nsb, SUPERBLOCK, rows])
        return ttnn.transpose(ttnn.reshape(ttnn.max(t, dim=-2), [1, 1, nsb, rows]), -2, -1)

    def candidates(self, score, tables, start: int) -> CandidateBlocks:
        """B11 on the candidate source's score: per query the CANDIDATE_TOPK_BLOCKS best blocks by max score, its
        newest row's block pinned in (``v41.select_candidate_blocks``)."""
        block, k = self.config.CANDIDATE_BLOCK_SIZE, self.config.CANDIDATE_TOPK_BLOCKS
        per = SUPERBLOCK // block
        width = score.shape[3]
        if width // block <= k:
            return CandidateBlocks(None)
        assert start % SUPERBLOCK == 0, f"chunk start {start} does not begin a superblock"
        nsb = width // SUPERBLOCK
        tiled = ttnn.to_layout(score, ttnn.TILE_LAYOUT)
        if nsb <= k:
            # one level: block maxima of the whole row, the newest row pinned to +inf (so is its block's max)
            tiled = self._add_cols(tiled, start, tables.newest_pin(width - start))
            pos = _topk(ttnn.to_layout(self._block_max(tiled), ttnn.ROW_MAJOR_LAYOUT), k)
            pos = ttnn.to_layout(pos, ttnn.TILE_LAYOUT)
            j, c = _split_major(pos, nsb, per)
            ids = ttnn.add(ttnn.multiply(c, per), j)
        else:
            # two levels: the top-k blocks lie in the top-k superblocks by max (the newest row's superblock pinned
            # first); their rows are gathered, pinned again at the newest row, and ranked by block max
            first = start // SUPERBLOCK
            sbmax = self._add_cols(self._superblock_max(tiled), first, tables.newest_pin(nsb - first, superblock=True))
            sb = _topk(ttnn.to_layout(sbmax, ttnn.ROW_MAJOR_LAYOUT), k)
            # a sentinel pick (the row sees fewer than k superblocks) reads the last superblock, invisible to it
            sb_safe = _min(ttnn.to_layout(sb, ttnn.TILE_LAYOUT), nsb - 1)
            rows = ttnn.to_layout(self._gather_superblocks(score, sb_safe, tables), ttnn.TILE_LAYOUT)
            rows = self._add_cols(rows, 0, tables.rank0_pin())
            pos = _topk(ttnn.to_layout(self._block_max(rows), ttnn.ROW_MAJOR_LAYOUT), k)
            pos = ttnn.to_layout(pos, ttnn.TILE_LAYOUT)
            j, r = _split_major(pos, k, per)
            r = ttnn.to_layout(_min(r, k - 1), ttnn.ROW_MAJOR_LAYOUT)
            c = ttnn.to_layout(_gather_cols(sb, r), ttnn.TILE_LAYOUT)
            ids = ttnn.add(ttnn.multiply(c, per), j)
        return CandidateBlocks(ttnn.to_layout(_keep_sentinel(ids, pos), ttnn.ROW_MAJOR_LAYOUT))

    def _topk_in_blocks(self, score, ids, tables, k: int):
        """Top-k rows of ``score`` among each query's candidate blocks ``ids`` [1, 1, R, K] uint32 row-major (sentinel
        tail) -> [1, 1, R, k] uint32 row-major row ids (sentinel tail)."""
        block, per = self.config.CANDIDATE_BLOCK_SIZE, SUPERBLOCK // self.config.CANDIDATE_BLOCK_SIZE
        nsb = score.shape[3] // SUPERBLOCK
        c = ttnn.to_layout(ids, ttnn.TILE_LAYOUT)
        sentinel = _is_sentinel(c)
        superblocks = _min(ttnn.bitwise_right_shift(c, per.bit_length() - 1), nsb - 1)
        # sub-block of each candidate inside its superblock; a sentinel candidate gets a key no column matches
        key = ttnn.add(ttnn.bitwise_and(c, per - 1), ttnn.multiply(sentinel, per))
        rows = ttnn.to_layout(self._gather_superblocks(score, superblocks, tables), ttnn.TILE_LAYOUT)
        keep = ttnn.eq(
            ttnn.repeat_interleave(ttnn.typecast(key, ttnn.bfloat16), SUPERBLOCK, dim=-1), self.subblock_pattern
        )
        rows = ttnn.where(keep, rows, float("-inf"))
        pos = ttnn.to_layout(_topk(ttnn.to_layout(rows, ttnn.ROW_MAJOR_LAYOUT), k), ttnn.TILE_LAYOUT)
        rank = _min(ttnn.bitwise_right_shift(pos, SUPERBLOCK.bit_length() - 1), ids.shape[3] - 1)
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
        elif self.uses_candidates:
            assert isinstance(candidates, CandidateBlocks), "a candidate index source needs the published candidates"
            if candidates.ids is not None:
                return self._topk_in_blocks(score, candidates.ids, tables, k), None
        return _topk(score, k), published

    def forward(self, x, qr, index_k, tables, start: int, length: int, candidates: CandidateBlocks | None = None):
        """-> (top-k rows [1, 1, S/(sp*tp), k] uint32 row-major (sentinel tail), published CandidateBlocks or None)."""
        score, visible = self.scores(x, qr, index_k, tables, start, length)
        return self.select(score, tables, start, visible, candidates)
