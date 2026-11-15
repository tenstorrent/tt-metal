# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DSpark parallel verify: the lightning indexer select family on a block of ``S`` rows of one session.

Row ``j`` of the block is the token at position ``p + j``: it scores the index keys that exist at that
position and keeps its own top-``index_topk``, exactly as a single-row select at ``p + j`` would. These
tests check every select mode of the V4.1 decode for ``S = 4`` against ``S`` single-row calls (one-level,
key split + ``merge_select_kv``, two-level candidate source + ``build_candidate_pool`` + pool consumer) and
profile ``S = 1`` against ``S = 4`` at production shapes.

Run the profile with::

    python -m tracy -r -p -v --no-web-server -t 8123 -o <out> -m pytest <this file> -k profile -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn

from tests.ttnn.unit_tests.operations.experimental.deepseek.test_fused_lightning_select_kv import (
    COMPRESS_RATE,
    PARENT_MESH,
    TP_SIZE,
    _per_device,
    _tp_submesh,
    _paged_inputs,
    _replicated_rm_hs,
    _rm,
    _tile,
)

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="indexer_score is Blackhole-only")

# V4.1-Flash indexer.
INDEX_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 512
HEAD_DIM = 512
PAGE_BLOCK = 64
CANDIDATE_TOPK_BLOCKS = 2048
CANDIDATE_BLOCK_SIZE = 8
ROWS = 4
ITERATIONS = 10
# The profiles' block sizes; DSPARK_SELECT_PROFILE_ROWS=1 profiles only S = 1 (e.g. against the pre-change kernels).
PROFILE_ROWS = [int(r) for r in os.environ.get("DSPARK_SELECT_PROFILE_ROWS", f"1,{ROWS}").split(",")]


def _signpost(name: str) -> None:
    try:
        from tracy import signpost

        signpost(name)
    except ImportError:
        pass


def _positions(rows: int, last_keys: int) -> torch.Tensor:
    """Inclusive positions ``p .. p + rows - 1`` of a verify block whose last row has ``last_keys`` keys."""
    last = last_keys * COMPRESS_RATE - 1
    return torch.arange(last - rows + 1, last + 1, dtype=torch.int32)


class _Inputs:
    """A paged index key cache and comp_kv cache of ``num_keys`` (rounded up to whole page blocks), plus
    ``rows`` query rows and head weights, on ``device``."""

    def __init__(self, device, num_keys: int, rows: int, seed: int = 0, mesh_mapper=None):
        torch.manual_seed(seed)
        length = -(-num_keys // PAGE_BLOCK) * PAGE_BLOCK
        self.length = length
        self.kv_cache, self.page_table = _paged_inputs(1, length, PAGE_BLOCK, 1, HEAD_DIM, seed=seed)
        self.key_cache = torch.randn(self.page_table.numel(), 1, PAGE_BLOCK, INDEX_HEAD_DIM)
        self.query = torch.randn(1, rows * INDEX_HEADS, 1, INDEX_HEAD_DIM)
        self.weights = torch.randn(1, 1, rows, INDEX_HEADS) * INDEX_HEAD_DIM**-0.5
        self.cur_pos = _positions(rows, num_keys)
        self.key_tt = _tile(self.key_cache, device, mesh_mapper=mesh_mapper)
        self.kv_tt = _rm(self.kv_cache, device, ttnn.bfloat16, mesh_mapper=mesh_mapper)
        self.page_table_tt = _rm(self.page_table, device, ttnn.int32, mesh_mapper=mesh_mapper)
        self.mesh_mapper = mesh_mapper
        self.device = device

    def query_tt(self, rows=None):
        rows = rows if rows is not None else range(self.query.shape[1] // INDEX_HEADS)
        q = torch.cat([self.query[:, j * INDEX_HEADS : (j + 1) * INDEX_HEADS] for j in rows], dim=1)
        return _replicated_rm_hs(q, self.device, mesh_mapper=self.mesh_mapper)

    def weights_tt(self, rows=None):
        rows = rows if rows is not None else range(self.weights.shape[2])
        w = torch.cat([self.weights[:, :, j : j + 1] for j in rows], dim=2)
        return _replicated_rm_hs(w, self.device, mesh_mapper=self.mesh_mapper)

    def cur_pos_tt(self, rows=None):
        rows = rows if rows is not None else range(self.cur_pos.numel())
        return _rm(self.cur_pos[list(rows)].contiguous(), self.device, ttnn.int32, mesh_mapper=self.mesh_mapper)


def _sel_buffer(device, rows: int, mesh_mapper=None) -> ttnn.Tensor:
    """The contract's selected-rows buffer: ``[1, 1, rows * index_topk, Dh]`` ROW_MAJOR DRAM (tilized after)."""
    return _rm(torch.zeros(1, 1, rows * INDEX_TOPK, HEAD_DIM, dtype=torch.bfloat16), device, ttnn.bfloat16, mesh_mapper)


# ---------------------------------------------------------------------------------------------------------------
# Correctness: S = 4 rows against S single-row calls of the same op (row j at position p + j)
# ---------------------------------------------------------------------------------------------------------------


def _single_select(inp: _Inputs, j: int, **kwargs):
    """Row j alone: the one-row select at its own position (today's s = 1 path)."""
    return ttnn.experimental.deepseek.fused_lightning_select_kv(
        inp.query_tt([j]),
        kwargs.pop("key_tt", inp.key_tt),
        inp.weights_tt([j]),
        inp.kv_tt,
        kwargs.pop("page_table_tt", inp.page_table_tt),
        inp.cur_pos_tt([j]),
        INDEX_TOPK,
        **kwargs,
    )


def _rows_equal(got: torch.Tensor, want: torch.Tensor, n: int, what: str) -> None:
    """``got`` / ``want`` ``[..., rows, Dh]``: the first ``n`` rows must be bit-identical (PCC 1.0)."""
    got = got.reshape(-1, HEAD_DIM)[:n].float()
    want = want.reshape(-1, HEAD_DIM)[:n].float()
    same = (got == want).all(dim=-1)
    assert bool(same.all()), f"{what}: {int((~same).sum())}/{n} rows differ, first at {int((~same).nonzero()[0])}"


@pytest.mark.parametrize("num_keys", [600, 8 * 1024 - 3, 64 * 1024 - 1], ids=["600", "8k", "64k"])
def test_one_level_rows(device, num_keys) -> None:
    """One-level unsplit: the 4-row select writes row j's top-512 at rows [j * 512, ...) of the contract's buffer,
    identical to row j's single-row call. ``600`` straddles nothing but leaves the first rows near the dense
    boundary."""
    inp = _Inputs(device, num_keys, ROWS)
    out = _sel_buffer(device, ROWS)
    _, scores = ttnn.experimental.deepseek.fused_lightning_select_kv(
        inp.query_tt(),
        inp.key_tt,
        inp.weights_tt(),
        inp.kv_tt,
        inp.page_table_tt,
        inp.cur_pos_tt(),
        INDEX_TOPK,
        output_tensor=out,
        output_row_offset=0,
        output_row_stride=INDEX_TOPK,
    )
    got = ttnn.to_torch(out).reshape(ROWS, INDEX_TOPK, HEAD_DIM)
    scores = ttnn.to_torch(scores).reshape(ROWS, -1)
    for j in range(ROWS):
        want, want_scores = _single_select(inp, j, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        n = min(INDEX_TOPK, int(inp.cur_pos[j] + 1) // COMPRESS_RATE)
        keys = int(inp.cur_pos[j] + 1) // COMPRESS_RATE
        assert torch.equal(scores[j, :keys], ttnn.to_torch(want_scores).reshape(-1)[:keys]), f"row {j}: scores"
        _rows_equal(got[j], ttnn.to_torch(want), n, f"row {j}")
    logger.info(f"one-level {num_keys} keys: 4 rows match their single-row selects")


def _candidate_source(inp: _Inputs, rows, topk_blocks, **kwargs):
    rows = list(rows)
    return ttnn.experimental.deepseek.fused_lightning_select_kv(
        inp.query_tt(rows),
        inp.key_tt,
        inp.weights_tt(rows),
        inp.kv_tt,
        inp.page_table_tt,
        inp.cur_pos_tt(rows),
        INDEX_TOPK,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
        **kwargs,
    )


@pytest.mark.parametrize(
    "num_keys, topk_blocks", [(8 * 1024 - 3, 256), (32 * 1024 - 1, CANDIDATE_TOPK_BLOCKS)], ids=["8k_256", "32k_2048"]
)
def test_candidate_source_rows(device, num_keys, topk_blocks) -> None:
    """Candidate source (unsplit): each row keeps its own blocks (its own newest block pinned) and count, and its
    own top-512 rows, all identical to its single-row call."""
    inp = _Inputs(device, num_keys, ROWS, seed=1)
    out = _sel_buffer(device, ROWS)
    _, _, blocks, counts = _candidate_source(
        inp, range(ROWS), topk_blocks, output_tensor=out, output_row_offset=0, output_row_stride=INDEX_TOPK
    )
    assert list(blocks.shape) == [ROWS, topk_blocks] and list(counts.shape) == [ROWS]
    blocks, counts = ttnn.to_torch(blocks), ttnn.to_torch(counts).reshape(-1)
    got = ttnn.to_torch(out).reshape(ROWS, INDEX_TOPK, HEAD_DIM)
    for j in range(ROWS):
        rows_j, _, blocks_j, count_j = _candidate_source(inp, [j], topk_blocks, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        assert torch.equal(blocks[j], ttnn.to_torch(blocks_j).reshape(-1)), f"row {j}: candidate blocks differ"
        assert int(counts[j]) == int(ttnn.to_torch(count_j).reshape(-1)[0]), f"row {j}: pool count differs"
        _rows_equal(got[j], ttnn.to_torch(rows_j), INDEX_TOPK, f"row {j}")
    logger.info(f"candidate source {num_keys} keys: pool counts {counts.tolist()}")


def _union_reference(blocks: torch.Tensor, counts: torch.Tensor, cbs: int):
    """Host union pool: ascending blocks of every row, each with its row mask, and each row's pool prefix."""
    chosen = [blocks[j, : -(-int(counts[j]) // cbs)].tolist() for j in range(blocks.shape[0])]
    union = sorted(set(b for c in chosen for b in c))
    masks = [sum(1 << j for j in range(len(chosen)) if b in set(chosen[j])) for b in union]
    prefix = []
    for j, c in enumerate(chosen):
        if not c:
            prefix.append(0)
            continue
        pinned_keys = int(counts[j]) - (len(c) - 1) * cbs
        prefix.append(union.index(c[-1]) * cbs + pinned_keys)
    return union, masks, prefix


@pytest.mark.parametrize(
    "num_keys, topk_blocks", [(8 * 1024 - 3, 256), (32 * 1024 - 1, CANDIDATE_TOPK_BLOCKS)], ids=["8k_256", "32k_2048"]
)
def test_union_pool_consumer_rows(device, num_keys, topk_blocks) -> None:
    """Two-level for 4 rows: source -> ``build_candidate_pool`` (union of the rows' blocks, with row masks and
    per-row prefixes) -> consumer on the union pool. Every row's consumer selection is identical to its own
    single-row chain (source -> one-row pool -> consumer). The union pool is checked against a host union."""
    cbs = CANDIDATE_BLOCK_SIZE
    inp = _Inputs(device, num_keys, ROWS, seed=2)
    torch.manual_seed(5)
    dst_query = torch.randn(1, ROWS * INDEX_HEADS, 1, INDEX_HEAD_DIM)
    dst_weights = torch.randn(1, 1, ROWS, INDEX_HEADS) * INDEX_HEAD_DIM**-0.5

    _, _, blocks, counts = _candidate_source(inp, range(ROWS), topk_blocks)
    pool_keys, pool_rows, pool_masks, pool_counts = ttnn.experimental.deepseek.build_candidate_pool(
        blocks, counts, inp.key_tt, inp.page_table_tt, cbs
    )
    pool_size = ROWS * topk_blocks * cbs
    assert list(pool_keys.shape) == [pool_size // PAGE_BLOCK, 1, PAGE_BLOCK, INDEX_HEAD_DIM]
    union, masks, prefix = _union_reference(ttnn.to_torch(blocks), ttnn.to_torch(counts).reshape(-1), cbs)
    got_counts = ttnn.to_torch(pool_counts).reshape(-1).tolist()
    assert got_counts == prefix, f"pool counts {got_counts}, expected {prefix}"
    got_masks = ttnn.to_torch(pool_masks).reshape(-1)[: len(union) * cbs].reshape(len(union), cbs)
    assert bool((got_masks == torch.tensor(masks, dtype=got_masks.dtype)[:, None]).all()), "pool masks differ"
    logger.info(f"union pool: {len(union)} blocks for 4 rows of {[-(-c // cbs) for c in prefix]}")

    identity = _rm(torch.arange(pool_size // PAGE_BLOCK, dtype=torch.int32).reshape(1, -1), device, ttnn.int32)
    out = _sel_buffer(device, ROWS)
    ttnn.experimental.deepseek.fused_lightning_select_kv(
        _replicated_rm_hs(dst_query, device),
        pool_keys,
        _replicated_rm_hs(dst_weights, device),
        inp.kv_tt,
        identity,
        inp.cur_pos_tt(),
        INDEX_TOPK,
        num_keys_tensor=pool_counts,
        row_table=pool_rows,
        row_mask_table=pool_masks,
        output_tensor=out,
        output_row_offset=0,
        output_row_stride=INDEX_TOPK,
    )
    got = ttnn.to_torch(out).reshape(ROWS, INDEX_TOPK, HEAD_DIM)

    single_identity = _rm(
        torch.arange(topk_blocks * cbs // PAGE_BLOCK, dtype=torch.int32).reshape(1, -1), device, ttnn.int32
    )
    for j in range(ROWS):
        _, _, blocks_j, count_j = _candidate_source(inp, [j], topk_blocks)
        keys_j, rows_j = ttnn.experimental.deepseek.build_candidate_pool(
            blocks_j, count_j, inp.key_tt, inp.page_table_tt, cbs
        )
        want, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            _replicated_rm_hs(dst_query[:, j * INDEX_HEADS : (j + 1) * INDEX_HEADS], device),
            keys_j,
            _replicated_rm_hs(dst_weights[:, :, j : j + 1], device),
            inp.kv_tt,
            single_identity,
            inp.cur_pos_tt([j]),
            INDEX_TOPK,
            num_keys_tensor=count_j,
            row_table=rows_j,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        _rows_equal(got[j], ttnn.to_torch(want), INDEX_TOPK, f"consumer row {j}")


# ---------------------------------------------------------------------------------------------------------------
# Key split over the 4 TP ranks: num_parts select, all_gather, merge_select_kv (S rows)
# ---------------------------------------------------------------------------------------------------------------

_MESH = [
    pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY}], indirect=True),
    pytest.mark.parametrize("mesh_device", [PARENT_MESH], indirect=True, ids=["8x4"]),
]


def _mesh(f):
    for mark in reversed(_MESH):
        f = mark(f)
    return f


def _part_index(submesh):
    return _rm(
        torch.arange(TP_SIZE, dtype=torch.int32),
        submesh,
        ttnn.int32,
        mesh_mapper=ttnn.ShardTensorToMesh(submesh, dim=0),
    )


def _split(submesh, part_index, rows, q, key_tt, w, kv_tt, page_table_tt, pos, out, merge_kwargs=None, **kwargs):
    """The model's split select of ``rows`` rows: each rank scores its part, all_gather, merge into ``out``."""
    candidates, scores = ttnn.experimental.deepseek.fused_lightning_select_kv(
        q, key_tt, w, kv_tt, page_table_tt, pos, INDEX_TOPK, num_parts=TP_SIZE, part_index_tensor=part_index, **kwargs
    )
    ttnn.deallocate(scores)
    gathered = ttnn.all_gather(candidates, dim=2, cluster_axis=1)
    ttnn.deallocate(candidates)
    ttnn.experimental.deepseek.merge_select_kv(
        gathered,
        kv_tt,
        INDEX_TOPK,
        output_tensor=out,
        output_row_offset=0,
        num_queries=rows,
        output_row_stride=INDEX_TOPK,
        **(merge_kwargs or {}),
    )
    ttnn.deallocate(gathered)


def _split_on_one_chip(device, rows, q, key_tt, w, kv_tt, page_table_tt, pos, out, merge_kwargs=None, **kwargs):
    """The split select's parts run one after another on one chip (part_index p for part p), their candidates
    concatenated in part order as all_gather would, then merged: everything of the split but the CCL."""
    parts = []
    for p in range(TP_SIZE):
        part_index = _rm(torch.tensor([p], dtype=torch.int32), device, ttnn.int32)
        candidates, scores = ttnn.experimental.deepseek.fused_lightning_select_kv(
            q,
            key_tt,
            w,
            kv_tt,
            page_table_tt,
            pos,
            INDEX_TOPK,
            num_parts=TP_SIZE,
            part_index_tensor=part_index,
            **kwargs,
        )
        ttnn.deallocate(scores)
        assert list(candidates.shape)[:3] == [1, 1, rows]
        parts.append(candidates)
    gathered = ttnn.concat(parts, dim=2)
    ttnn.experimental.deepseek.merge_select_kv(
        gathered,
        kv_tt,
        INDEX_TOPK,
        output_tensor=out,
        output_row_offset=0,
        num_queries=rows,
        output_row_stride=INDEX_TOPK,
        **(merge_kwargs or {}),
    )


@pytest.mark.parametrize("num_keys", [1000, 64 * 1024 - 1], ids=["1000", "64k"])
@pytest.mark.parametrize("mode", ["one_level", "source", "consumer"])
def test_split_rows_one_chip(device, num_keys, mode) -> None:
    """Key split of a 4-row select, with the 4 parts run on one chip: the merged rows (and in candidate source mode
    the per-row blocks and counts) equal the unsplit 4-row select's. ``consumer`` splits a union-pool consumer."""
    cbs, topk_blocks = CANDIDATE_BLOCK_SIZE, (CANDIDATE_TOPK_BLOCKS if num_keys > 2048 else 32)
    inp = _Inputs(device, num_keys, ROWS, seed=11)
    q, w, pos = inp.query_tt(), inp.weights_tt(), inp.cur_pos_tt()
    key_tt, page_table_tt, kwargs, merge_kwargs = inp.key_tt, inp.page_table_tt, {}, {}
    mode_args = dict(candidate_topk_blocks=topk_blocks, candidate_block_size=cbs)
    if mode == "source":
        kwargs = dict(mode_args)
    elif mode == "consumer":
        _, _, blocks, counts = ttnn.experimental.deepseek.fused_lightning_select_kv(
            q, key_tt, w, inp.kv_tt, page_table_tt, pos, INDEX_TOPK, **mode_args
        )
        pool_keys, pool_rows, pool_masks, pool_counts = ttnn.experimental.deepseek.build_candidate_pool(
            blocks, counts, key_tt, page_table_tt, cbs
        )
        key_tt = pool_keys
        page_table_tt = _rm(torch.arange(pool_keys.shape[0], dtype=torch.int32).reshape(1, -1), device, ttnn.int32)
        kwargs = dict(num_keys_tensor=pool_counts, row_table=pool_rows, row_mask_table=pool_masks)
    want = _sel_buffer(device, ROWS)
    unsplit_extra = {}
    if mode == "source":
        want_blocks = _rm(torch.zeros(ROWS, topk_blocks, dtype=torch.int32), device, ttnn.int32)
        want_counts = _rm(torch.zeros(ROWS, dtype=torch.int32), device, ttnn.int32)
        unsplit_extra = dict(candidate_blocks_output=want_blocks, pool_count_output=want_counts)
        blocks = _rm(torch.full((ROWS, topk_blocks), -1, dtype=torch.int32), device, ttnn.int32)
        counts = _rm(torch.full((ROWS,), -1, dtype=torch.int32), device, ttnn.int32)
        merge_kwargs = dict(candidate_blocks_output=blocks, pool_count_output=counts, **mode_args)
    ttnn.experimental.deepseek.fused_lightning_select_kv(
        q,
        key_tt,
        w,
        inp.kv_tt,
        page_table_tt,
        pos,
        INDEX_TOPK,
        output_tensor=want,
        output_row_offset=0,
        output_row_stride=INDEX_TOPK,
        **kwargs,
        **unsplit_extra,
    )
    got = _sel_buffer(device, ROWS)
    _split_on_one_chip(device, ROWS, q, key_tt, w, inp.kv_tt, page_table_tt, pos, got, merge_kwargs, **kwargs)
    g = ttnn.to_torch(got).reshape(ROWS, INDEX_TOPK, HEAD_DIM)
    x = ttnn.to_torch(want).reshape(ROWS, INDEX_TOPK, HEAD_DIM)
    for j in range(ROWS):
        n = min(INDEX_TOPK, int(inp.cur_pos[j] + 1) // COMPRESS_RATE)
        _rows_equal(g[j], x[j], n, f"{mode} row {j}")
    if mode == "source":
        assert torch.equal(ttnn.to_torch(blocks), ttnn.to_torch(want_blocks)), "merged candidate blocks differ"
        assert torch.equal(ttnn.to_torch(counts), ttnn.to_torch(want_counts)), "merged pool counts differ"


@_mesh
@pytest.mark.parametrize("num_keys", [1000, 64 * 1024 - 1], ids=["1000", "64k"])
def test_split_rows(mesh_device, num_keys) -> None:
    """Split one-level select of 4 rows gathers, on every rank, exactly the unsplit 4-row select's rows (which
    ``test_one_level_rows`` ties to the single-row selects)."""
    submesh = _tp_submesh(mesh_device)
    inp = _Inputs(submesh, num_keys, ROWS, seed=3, mesh_mapper=ttnn.ReplicateTensorToMesh(submesh))
    q, w, pos = inp.query_tt(), inp.weights_tt(), inp.cur_pos_tt()
    want = _sel_buffer(submesh, ROWS, inp.mesh_mapper)
    ttnn.experimental.deepseek.fused_lightning_select_kv(
        q,
        inp.key_tt,
        w,
        inp.kv_tt,
        inp.page_table_tt,
        pos,
        INDEX_TOPK,
        output_tensor=want,
        output_row_offset=0,
        output_row_stride=INDEX_TOPK,
    )
    got = _sel_buffer(submesh, ROWS, inp.mesh_mapper)
    _split(submesh, _part_index(submesh), ROWS, q, inp.key_tt, w, inp.kv_tt, inp.page_table_tt, pos, got)
    for d, (g, x) in enumerate(zip(_per_device(got, submesh), _per_device(want, submesh))):
        g, x = g.reshape(ROWS, INDEX_TOPK, HEAD_DIM), x.reshape(ROWS, INDEX_TOPK, HEAD_DIM)
        for j in range(ROWS):
            n = min(INDEX_TOPK, int(inp.cur_pos[j] + 1) // COMPRESS_RATE)
            _rows_equal(g[j], x[j], n, f"rank {d} row {j}")


@_mesh
@pytest.mark.parametrize(
    "num_keys, topk_blocks", [(8 * 1024 - 3, 256), (64 * 1024 - 1, CANDIDATE_TOPK_BLOCKS)], ids=["8k_256", "64k_2048"]
)
def test_split_candidate_source_rows(mesh_device, num_keys, topk_blocks) -> None:
    """Split candidate source of 4 rows: on every rank, the merged rows, per-row blocks and counts equal the
    unsplit 4-row source's (which ``test_candidate_source_rows`` ties to the single-row sources)."""
    submesh = _tp_submesh(mesh_device)
    rep = ttnn.ReplicateTensorToMesh(submesh)
    inp = _Inputs(submesh, num_keys, ROWS, seed=4, mesh_mapper=rep)
    q, w, pos = inp.query_tt(), inp.weights_tt(), inp.cur_pos_tt()
    mode = dict(candidate_topk_blocks=topk_blocks, candidate_block_size=CANDIDATE_BLOCK_SIZE)
    want = _sel_buffer(submesh, ROWS, rep)
    _, _, want_blocks, want_counts = ttnn.experimental.deepseek.fused_lightning_select_kv(
        q,
        inp.key_tt,
        w,
        inp.kv_tt,
        inp.page_table_tt,
        pos,
        INDEX_TOPK,
        output_tensor=want,
        output_row_offset=0,
        output_row_stride=INDEX_TOPK,
        **mode,
    )
    got = _sel_buffer(submesh, ROWS, rep)
    blocks = _rm(torch.full((ROWS, topk_blocks), -1, dtype=torch.int32), submesh, ttnn.int32, rep)
    counts = _rm(torch.full((ROWS,), -1, dtype=torch.int32), submesh, ttnn.int32, rep)
    _split(
        submesh,
        _part_index(submesh),
        ROWS,
        q,
        inp.key_tt,
        w,
        inp.kv_tt,
        inp.page_table_tt,
        pos,
        got,
        merge_kwargs=dict(candidate_blocks_output=blocks, pool_count_output=counts, **mode),
        **mode,
    )
    for d in range(TP_SIZE):
        g = _per_device(got, submesh)[d].reshape(ROWS, INDEX_TOPK, HEAD_DIM)
        x = _per_device(want, submesh)[d].reshape(ROWS, INDEX_TOPK, HEAD_DIM)
        for j in range(ROWS):
            _rows_equal(g[j], x[j], INDEX_TOPK, f"rank {d} row {j}")
        assert torch.equal(_per_device(blocks, submesh)[d], _per_device(want_blocks, submesh)[d]), f"rank {d}: blocks"
        assert torch.equal(_per_device(counts, submesh)[d], _per_device(want_counts, submesh)[d]), f"rank {d}: counts"


@_mesh
def test_split_pool_consumer_rows(mesh_device) -> None:
    """A union-pool consumer of 4 rows split over the ranks gathers exactly the unsplit consumer's rows."""
    submesh = _tp_submesh(mesh_device)
    rep = ttnn.ReplicateTensorToMesh(submesh)
    cbs, topk_blocks = CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS
    inp = _Inputs(submesh, 64 * 1024 - 1, ROWS, seed=6, mesh_mapper=rep)
    _, _, blocks, counts = ttnn.experimental.deepseek.fused_lightning_select_kv(
        inp.query_tt(),
        inp.key_tt,
        inp.weights_tt(),
        inp.kv_tt,
        inp.page_table_tt,
        inp.cur_pos_tt(),
        INDEX_TOPK,
        candidate_topk_blocks=topk_blocks,
        candidate_block_size=cbs,
    )
    pool_keys, pool_rows, pool_masks, pool_counts = ttnn.experimental.deepseek.build_candidate_pool(
        blocks, counts, inp.key_tt, inp.page_table_tt, cbs
    )
    pool_size = ROWS * topk_blocks * cbs
    identity = _rm(torch.arange(pool_size // PAGE_BLOCK, dtype=torch.int32).reshape(1, -1), submesh, ttnn.int32, rep)
    consumer = dict(num_keys_tensor=pool_counts, row_table=pool_rows, row_mask_table=pool_masks)
    q, w, pos = inp.query_tt(), inp.weights_tt(), inp.cur_pos_tt()
    want = _sel_buffer(submesh, ROWS, rep)
    ttnn.experimental.deepseek.fused_lightning_select_kv(
        q,
        pool_keys,
        w,
        inp.kv_tt,
        identity,
        pos,
        INDEX_TOPK,
        output_tensor=want,
        output_row_offset=0,
        output_row_stride=INDEX_TOPK,
        **consumer,
    )
    got = _sel_buffer(submesh, ROWS, rep)
    _split(submesh, _part_index(submesh), ROWS, q, pool_keys, w, inp.kv_tt, identity, pos, got, **consumer)
    for d in range(TP_SIZE):
        g = _per_device(got, submesh)[d].reshape(ROWS, INDEX_TOPK, HEAD_DIM)
        x = _per_device(want, submesh)[d].reshape(ROWS, INDEX_TOPK, HEAD_DIM)
        for j in range(ROWS):
            _rows_equal(g[j], x[j], INDEX_TOPK, f"rank {d} row {j}")


# ---------------------------------------------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_keys", [8 * 1024, 64 * 1024], ids=["8k", "64k"])
def test_profile_one_level(device, num_keys) -> None:
    """One-level unsplit select, ``S = 1`` vs ``S = 4``, into the contract's ``[S * 512, Dh]`` buffer."""
    for rows in PROFILE_ROWS:
        inp = _Inputs(device, num_keys, rows)
        q, w, pos = inp.query_tt(), inp.weights_tt(), inp.cur_pos_tt()
        out = _sel_buffer(device, rows)

        def run():
            sel, scores = ttnn.experimental.deepseek.fused_lightning_select_kv(
                q,
                inp.key_tt,
                w,
                inp.kv_tt,
                inp.page_table_tt,
                pos,
                INDEX_TOPK,
                output_tensor=out,
                output_row_offset=0,
                output_row_stride=INDEX_TOPK,
            )
            ttnn.deallocate(scores)

        run()
        ttnn.synchronize_device(device)
        _signpost(f"one_level-{num_keys}-s{rows}")
        for _ in range(ITERATIONS):
            run()
        ttnn.synchronize_device(device)


def _timed(name: str, device, run) -> None:
    run()
    ttnn.synchronize_device(device)
    _signpost(name)
    for _ in range(ITERATIONS):
        run()
    ttnn.synchronize_device(device)


def _correlated(inp: _Inputs, noise: float) -> None:
    """Rows of one session: every row's query / weights are row 0's plus ``noise`` (0: the same query)."""
    q0 = inp.query[:, :INDEX_HEADS]
    w0 = inp.weights[:, :, :1]
    rows = inp.weights.shape[2]
    inp.query = torch.cat([q0 + noise * torch.randn_like(q0) * (j > 0) for j in range(rows)], dim=1)
    inp.weights = torch.cat([w0 + noise * torch.randn_like(w0) * (j > 0) for j in range(rows)], dim=2)


@pytest.mark.parametrize("num_keys", [24 * 1024, 64 * 1024], ids=["24k", "64k"])
@pytest.mark.parametrize("noise", [0.3, 1e9], ids=["correlated", "independent"])
def test_profile_two_level(device, num_keys, noise) -> None:
    """Two-level, unsplit: candidate source, build_candidate_pool, pool consumer, ``S = 1`` vs ``S = 4``. The pool
    a 4-row consumer scores is the union of the rows' blocks: ``correlated`` rows (query = row 0's + 0.3 noise)
    share most blocks, ``independent`` rows (random queries) almost none."""
    cbs, topk_blocks = CANDIDATE_BLOCK_SIZE, CANDIDATE_TOPK_BLOCKS
    for rows in PROFILE_ROWS:
        inp = _Inputs(device, num_keys, rows, seed=7)
        if noise < 1e6:
            _correlated(inp, noise)
        q, w, pos = inp.query_tt(), inp.weights_tt(), inp.cur_pos_tt()
        out = _sel_buffer(device, rows)
        blocks = _rm(torch.zeros(rows, topk_blocks, dtype=torch.int32), device, ttnn.int32)
        counts = _rm(torch.zeros(rows, dtype=torch.int32), device, ttnn.int32)
        sel = dict(output_tensor=out, output_row_offset=0, output_row_stride=INDEX_TOPK)

        def source():
            _, scores, _, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
                q,
                inp.key_tt,
                w,
                inp.kv_tt,
                inp.page_table_tt,
                pos,
                INDEX_TOPK,
                candidate_topk_blocks=topk_blocks,
                candidate_block_size=cbs,
                candidate_blocks_output=blocks,
                pool_count_output=counts,
                **sel,
            )
            ttnn.deallocate(scores)

        _timed(f"source-{num_keys}-{noise}-s{rows}", device, source)
        pool = ttnn.experimental.deepseek.build_candidate_pool(blocks, counts, inp.key_tt, inp.page_table_tt, cbs)

        def build():
            ttnn.experimental.deepseek.build_candidate_pool(
                blocks,
                counts,
                inp.key_tt,
                inp.page_table_tt,
                cbs,
                pool_keys_output=pool[0],
                pool_rows_output=pool[1],
                **(dict(pool_masks_output=pool[2], pool_counts_output=pool[3]) if rows > 1 else {}),
            )

        _timed(f"build-{num_keys}-{noise}-s{rows}", device, build)
        if rows > 1:
            union = int(ttnn.to_torch(pool[3]).reshape(-1).max())
            logger.info(f"{num_keys} keys, noise {noise}: union pool of {union} keys for {rows} rows")
        identity = _rm(torch.arange(pool[0].shape[0], dtype=torch.int32).reshape(1, -1), device, ttnn.int32)
        consumer = dict(num_keys_tensor=pool[3] if rows > 1 else counts, row_table=pool[1])
        if rows > 1:
            consumer["row_mask_table"] = pool[2]

        def consume():
            _, scores = ttnn.experimental.deepseek.fused_lightning_select_kv(
                q, pool[0], w, inp.kv_tt, identity, pos, INDEX_TOPK, **consumer, **sel
            )
            ttnn.deallocate(scores)

        _timed(f"consumer-{num_keys}-{noise}-s{rows}", device, consume)


@_mesh
@pytest.mark.parametrize("num_keys", [31 * 1024, 45 * 1024, 64 * 1024], ids=["31k", "45k", "64k"])
def test_profile_split(mesh_device, num_keys) -> None:
    """Key split over the 4 ranks (select num_parts=4, all_gather, merge), one-level and candidate source,
    ``S = 1`` vs ``S = 4``; read the per-op medians (FusedLightningSelectKv, AllGather, MergeSelectKv)."""
    submesh = _tp_submesh(mesh_device)
    rep = ttnn.ReplicateTensorToMesh(submesh)
    part_index = _part_index(submesh)
    for rows in PROFILE_ROWS:
        inp = _Inputs(submesh, num_keys, rows, seed=8, mesh_mapper=rep)
        q, w, pos = inp.query_tt(), inp.weights_tt(), inp.cur_pos_tt()
        out = _sel_buffer(submesh, rows, rep)
        _timed(
            f"split-{num_keys}-s{rows}",
            submesh,
            lambda: _split(submesh, part_index, rows, q, inp.key_tt, w, inp.kv_tt, inp.page_table_tt, pos, out),
        )
        mode = dict(candidate_topk_blocks=CANDIDATE_TOPK_BLOCKS, candidate_block_size=CANDIDATE_BLOCK_SIZE)
        blocks = _rm(torch.zeros(rows, CANDIDATE_TOPK_BLOCKS, dtype=torch.int32), submesh, ttnn.int32, rep)
        counts = _rm(torch.zeros(rows, dtype=torch.int32), submesh, ttnn.int32, rep)
        _timed(
            f"split_source-{num_keys}-s{rows}",
            submesh,
            lambda: _split(
                submesh,
                part_index,
                rows,
                q,
                inp.key_tt,
                w,
                inp.kv_tt,
                inp.page_table_tt,
                pos,
                out,
                merge_kwargs=dict(candidate_blocks_output=blocks, pool_count_output=counts, **mode),
                **mode,
            ),
        )


@pytest.mark.parametrize("num_keys", [31 * 1024, 45 * 1024, 64 * 1024], ids=["31k", "45k", "64k"])
def test_profile_split_one_chip(device, num_keys) -> None:
    """Per-rank device time of the key split on one chip: one part's select (``num_parts`` 4, the last part,
    which holds the newest keys) and the merge of 4 parts' candidates, one-level and candidate source,
    ``S = 1`` vs ``S = 4``. The all_gather between them (~8-12 us, measured on the Galaxy) is not included."""
    mode = dict(candidate_topk_blocks=CANDIDATE_TOPK_BLOCKS, candidate_block_size=CANDIDATE_BLOCK_SIZE)
    for rows in PROFILE_ROWS:
        inp = _Inputs(device, num_keys, rows, seed=8)
        q, w, pos = inp.query_tt(), inp.weights_tt(), inp.cur_pos_tt()
        out = _sel_buffer(device, rows)
        for name, kw in (("split", {}), ("split_source", mode)):
            parts = []
            for p in range(TP_SIZE):
                part_index = _rm(torch.tensor([p], dtype=torch.int32), device, ttnn.int32)
                cand, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
                    q,
                    inp.key_tt,
                    w,
                    inp.kv_tt,
                    inp.page_table_tt,
                    pos,
                    INDEX_TOPK,
                    num_parts=TP_SIZE,
                    part_index_tensor=part_index,
                    **kw,
                )
                parts.append(cand)
            gathered = ttnn.concat(parts, dim=2)
            merge_kw = {}
            if kw:
                merge_kw = dict(
                    candidate_blocks_output=_rm(
                        torch.zeros(rows, CANDIDATE_TOPK_BLOCKS, dtype=torch.int32), device, ttnn.int32
                    ),
                    pool_count_output=_rm(torch.zeros(rows, dtype=torch.int32), device, ttnn.int32),
                    **kw,
                )

            def select():
                c, s = ttnn.experimental.deepseek.fused_lightning_select_kv(
                    q,
                    inp.key_tt,
                    w,
                    inp.kv_tt,
                    inp.page_table_tt,
                    pos,
                    INDEX_TOPK,
                    num_parts=TP_SIZE,
                    part_index_tensor=part_index,
                    **kw,
                )
                ttnn.deallocate(c)
                ttnn.deallocate(s)

            def merge():
                ttnn.experimental.deepseek.merge_select_kv(
                    gathered,
                    inp.kv_tt,
                    INDEX_TOPK,
                    output_tensor=out,
                    output_row_offset=0,
                    num_queries=rows,
                    output_row_stride=INDEX_TOPK,
                    **merge_kw,
                )

            _timed(f"{name}_select-{num_keys}-s{rows}", device, select)
            _timed(f"{name}_merge-{num_keys}-s{rows}", device, merge)
