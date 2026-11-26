# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""``fused_lightning_select_kv``'s row split (``query_row_tensor``): a B=4 batch step's rows, one session each, with
each device selecting one row. On one device the four devices' calls run one after another, their candidates pages
are concatenated (the all-gather) and ``merge_select_kv`` gathers every row's rows; all of it must match one-row
selects of each session (``page_table_row_tensor``). The candidate source's blocks and pool count, written locally,
must match the one-row source's, and so must ``build_candidate_pool`` (``query_row_tensor`` picking the session row)
and a pool consumer's row split.

Run::

    pytest tests/ttnn/unit_tests/operations/experimental/deepseek/test_fused_lightning_select_kv_row_split.py
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.experimental.deepseek.test_fused_lightning_select_kv import (
    COMPRESS_RATE,
    V4_HEAD_DIM,
    V4_INDEX_HEAD_DIM,
    V4_INDEX_TOPK,
    _replicated_rm_hs,
    _rm,
    _session_tables,
    _tile,
)

pytestmark = pytest.mark.skipif(not ttnn.device.is_blackhole(), reason="fused_lightning_select_kv is Blackhole-only")

HEADS = 32  # V4.1's index heads
ROWS = 4
TP = 4  # ranks: B=4 is one rank a row, B=2 a pair of ranks a row
BLOCK = 64


def _int(values, device) -> ttnn.Tensor:
    return _rm(torch.tensor(values, dtype=torch.int32), device, ttnn.int32)


def _inputs(device, num_keys: list[int], seed: int, rows: int = ROWS):
    """``rows`` sessions of one shared block pool, each with its own key count; the rows' queries and weights."""
    ROWS = rows  # noqa: N806 (the batch rows of this call)
    torch.manual_seed(seed)
    blocks_per_session = max(-(-n // BLOCK) for n in num_keys)
    tables = _session_tables(ROWS + 1, blocks_per_session, seed=seed)
    kv_cache = torch.randn(tables.numel(), 1, BLOCK, V4_HEAD_DIM).to(torch.bfloat16)
    key_cache = torch.randn(tables.numel(), 1, BLOCK, V4_INDEX_HEAD_DIM).to(torch.bfloat16)
    query = torch.randn(1, ROWS * HEADS, 1, V4_INDEX_HEAD_DIM)
    weights = torch.randn(1, 1, ROWS, HEADS) * V4_INDEX_HEAD_DIM**-0.5
    # Row j is session j + 1 (session 0 stays unused, as the scratch session of a short batch).
    sessions = list(range(1, ROWS + 1))
    return dict(
        tables=_rm(tables, device, ttnn.int32),
        kv=_rm(kv_cache, device, ttnn.bfloat16),
        keys=_tile(key_cache, device),
        query4=_replicated_rm_hs(query, device),
        weights4=_replicated_rm_hs(weights, device),
        query=[_replicated_rm_hs(query[:, j * HEADS : (j + 1) * HEADS], device) for j in range(ROWS)],
        weights=[_replicated_rm_hs(weights[:, :, j : j + 1], device) for j in range(ROWS)],
        pos4=_int([n * COMPRESS_RATE - 1 for n in num_keys], device),
        pos=[_int([n * COMPRESS_RATE - 1], device) for n in num_keys],
        page_rows4=_int(sessions, device),
        page_row=[_int([s], device) for s in sessions],
        query_row=[_int([j], device) for j in range(ROWS)],
        part=[_int([p], device) for p in range(TP)],
    )


def _merge(pages: list[ttnn.Tensor], kv: ttnn.Tensor, k: int) -> torch.Tensor:
    """The devices' candidates pages all-gathered (concatenated), then merged: every row's rows, row j at j * k."""
    gathered = ttnn.concat(pages, dim=2)
    merged = ttnn.experimental.deepseek.merge_select_kv(gathered, kv, k, num_queries=ROWS)
    return ttnn.to_torch(merged).reshape(ROWS, k, -1)


@pytest.mark.parametrize(
    "num_keys",
    [[1, 700, 2045, 4096], [8192, 8191, 6000, 513], [32768 - 3, 100, 32768, 20000]],
    ids=["short", "8k", "32k"],
)
def test_row_split_matches_one_row_selects(device, num_keys) -> None:
    k = V4_INDEX_TOPK
    t = _inputs(device, num_keys, seed=21)
    expected = []
    for j in range(ROWS):
        rows, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            t["query"][j],
            t["keys"],
            t["weights"][j],
            t["kv"],
            t["tables"],
            t["pos"][j],
            k,
            page_table_row_tensor=t["page_row"][j],
        )
        expected.append(ttnn.to_torch(rows).reshape(k, -1))
    pages = []
    for j in range(ROWS):
        page, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            t["query4"],
            t["keys"],
            t["weights4"],
            t["kv"],
            t["tables"],
            t["pos4"],
            k,
            page_table_row_tensor=t["page_rows4"],
            query_row_tensor=t["query_row"][j],
        )
        assert list(page.shape) == [1, 1, 1, 2 * k + 16], list(page.shape)
        header = ttnn.to_torch(page).reshape(-1)[2 * k : 2 * k + 3].tolist()
        assert header[0] == min(k, num_keys[j]) and header[2] == num_keys[j], (j, header)
        pages.append(page)
    got = _merge(pages, t["kv"], k)
    tiled = _merge_tiled(pages, t["kv"], k)
    for j in range(ROWS):
        n = min(k, num_keys[j])
        assert torch.equal(got[j, :n], expected[j][:n]), f"row {j}: the row split gathered different rows"
        assert torch.equal(tiled[j, :n], expected[j][:n]), f"row {j}: the tiled merge output differs"
        # Attention skips rows past the selection by length, but a NaN there would still poison it.
        assert not tiled[j, n : -(-n // 32) * 32].any(), f"row {j}: the tiled output's block tail is not zero"


def test_row_split_own_counts_and_page_row(device) -> None:
    """Per-row inputs given as this device's own ``[1]`` entries select the same rows as ``[Sq]`` ones."""
    k = V4_INDEX_TOPK
    num_keys = [3000, 4000, 1500, 600]
    t = _inputs(device, num_keys, seed=22)
    for j in (0, 3):
        outs = []
        for pos, row in ((t["pos4"], t["page_rows4"]), (t["pos"][j], t["page_row"][j])):
            page, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
                t["query4"],
                t["keys"],
                t["weights4"],
                t["kv"],
                t["tables"],
                pos,
                k,
                page_table_row_tensor=row,
                query_row_tensor=t["query_row"][j],
            )
            outs.append(ttnn.to_torch(page))
        assert torch.equal(outs[0], outs[1]), f"row {j}: own-entry inputs changed the candidates page"


def test_row_split_candidate_pool_chain(device) -> None:
    """Candidate source, ``build_candidate_pool`` and a pool consumer, each row split, against one-row calls."""
    k, cbs, topk_blocks = V4_INDEX_TOPK, 8, 256
    num_keys = [8189, 3000, 600, 8192]
    t = _inputs(device, num_keys, seed=23)
    pool_blocks = topk_blocks * cbs // BLOCK
    identity = _int([list(range(pool_blocks))], device)
    source = dict(candidate_topk_blocks=topk_blocks, candidate_block_size=cbs)
    torch.manual_seed(24)
    dst_query = torch.randn(1, ROWS * HEADS, 1, V4_INDEX_HEAD_DIM)
    dst_query4 = _replicated_rm_hs(dst_query, device)
    dst_query1 = [_replicated_rm_hs(dst_query[:, j * HEADS : (j + 1) * HEADS], device) for j in range(ROWS)]

    pages, consumer_pages = [], []
    for j in range(ROWS):
        # One-row reference: source, pool, consumer of session j.
        rows_1, _, blocks_1, count_1 = ttnn.experimental.deepseek.fused_lightning_select_kv(
            t["query"][j],
            t["keys"],
            t["weights"][j],
            t["kv"],
            t["tables"],
            t["pos"][j],
            k,
            page_table_row_tensor=t["page_row"][j],
            **source,
        )
        keys_1, prows_1 = ttnn.experimental.deepseek.build_candidate_pool(
            blocks_1, count_1, t["keys"], t["tables"], cbs, page_table_row_tensor=t["page_row"][j]
        )
        cons_1, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            dst_query1[j],
            keys_1,
            t["weights"][j],
            t["kv"],
            identity,
            t["pos"][j],
            k,
            num_keys_tensor=count_1,
            row_table=prows_1,
        )
        # Row split: device j's calls.
        page, _, blocks, count = ttnn.experimental.deepseek.fused_lightning_select_kv(
            t["query4"],
            t["keys"],
            t["weights4"],
            t["kv"],
            t["tables"],
            t["pos4"],
            k,
            page_table_row_tensor=t["page_rows4"],
            query_row_tensor=t["query_row"][j],
            **source,
        )
        assert torch.equal(ttnn.to_torch(blocks), ttnn.to_torch(blocks_1)), f"row {j}: candidate blocks differ"
        assert torch.equal(ttnn.to_torch(count), ttnn.to_torch(count_1)), f"row {j}: pool count differs"
        pool_keys, pool_rows = ttnn.experimental.deepseek.build_candidate_pool(
            blocks,
            count,
            t["keys"],
            t["tables"],
            cbs,
            page_table_row_tensor=t["page_rows4"],
            query_row_tensor=t["query_row"][j],
        )
        n_pool = int(ttnn.to_torch(count).reshape(-1)[0])
        assert torch.equal(
            ttnn.to_torch(pool_rows).reshape(-1)[:n_pool], ttnn.to_torch(prows_1).reshape(-1)[:n_pool]
        ), f"row {j}: pool rows differ"
        cpage, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            dst_query4,
            pool_keys,
            t["weights4"],
            t["kv"],
            identity,
            t["pos4"],
            k,
            num_keys_tensor=count,
            row_table=pool_rows,
            query_row_tensor=t["query_row"][j],
        )
        pages.append(page)
        consumer_pages.append(cpage)
        n = min(k, num_keys[j])
        assert torch.equal(
            _merge_one(page, t["kv"], k)[:n], ttnn.to_torch(rows_1).reshape(k, -1)[:n]
        ), f"row {j}: source rows differ"
        n = min(k, n_pool)
        assert torch.equal(
            _merge_one(cpage, t["kv"], k)[:n], ttnn.to_torch(cons_1).reshape(k, -1)[:n]
        ), f"row {j}: consumer rows differ"
    _merge(pages, t["kv"], k)
    _merge(consumer_pages, t["kv"], k)


def _merge_tiled(pages: list[ttnn.Tensor], kv: ttnn.Tensor, k: int) -> torch.Tensor:
    """As :func:`_merge`, into a TILE output (merge_select_kv tilizes each core's 32-row block)."""
    gathered = ttnn.concat(pages, dim=2)
    out = ttnn.from_torch(
        torch.full((1, 1, ROWS * k, V4_HEAD_DIM), float("nan"), dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=kv.device(),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ttnn.experimental.deepseek.merge_select_kv(
        gathered, kv, k, num_queries=ROWS, output_tensor=out, output_row_stride=k
    )
    return ttnn.to_torch(out).reshape(ROWS, k, -1)


def _merge_one(page: ttnn.Tensor, kv: ttnn.Tensor, k: int) -> torch.Tensor:
    return ttnn.to_torch(ttnn.experimental.deepseek.merge_select_kv(page, kv, k, num_queries=1)).reshape(k, -1)


# ---- B=2: a pair of ranks a row (rank r scores part r // 2 of row r % 2's keys) ----

PAIR_ROWS = 2
PAIR_PARTS = TP // PAIR_ROWS


def _pair_merge(pages, kv, k, **candidates) -> torch.Tensor:
    """The four ranks' pages all-gathered in rank order (part-major: page p * 2 + j) and merged, two parts a row."""
    gathered = ttnn.concat(pages, dim=2)
    merged = ttnn.experimental.deepseek.merge_select_kv(gathered, kv, k, num_queries=PAIR_ROWS, **candidates)
    return ttnn.to_torch(merged).reshape(PAIR_ROWS, k, -1)


@pytest.mark.parametrize("num_keys", [[1, 700], [8192, 6000], [32768 - 3, 20000]], ids=["short", "8k", "32k"])
def test_pair_split_matches_one_row_selects(device, num_keys) -> None:
    k = V4_INDEX_TOPK
    t = _inputs(device, num_keys, seed=31, rows=PAIR_ROWS)
    expected = []
    for j in range(PAIR_ROWS):
        rows, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            t["query"][j],
            t["keys"],
            t["weights"][j],
            t["kv"],
            t["tables"],
            t["pos"][j],
            k,
            page_table_row_tensor=t["page_row"][j],
        )
        expected.append(ttnn.to_torch(rows).reshape(k, -1))
    pages = []
    for r in range(TP):
        page, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            t["query4"],
            t["keys"],
            t["weights4"],
            t["kv"],
            t["tables"],
            t["pos4"],
            k,
            page_table_row_tensor=t["page_rows4"],
            query_row_tensor=t["query_row"][r % PAIR_ROWS],
            num_parts=PAIR_PARTS,
            part_index_tensor=t["part"][r // PAIR_ROWS],
        )
        assert list(page.shape) == [1, 1, 1, 2 * k + 16], list(page.shape)
        pages.append(page)
    got = _pair_merge(pages, t["kv"], k)
    for j in range(PAIR_ROWS):
        n = min(k, num_keys[j])
        assert torch.equal(got[j, :n], expected[j][:n]), f"row {j}: the pair split gathered different rows"


def test_pair_split_candidate_pool_chain(device) -> None:
    """B=2 two-level: the source's pair split and merge write both rows' blocks and counts, each rank's consumer
    builds its row's pool from them (``query_row_tensor`` picking the row) and key-splits it over its pair."""
    k, cbs, topk_blocks = V4_INDEX_TOPK, 8, 256
    num_keys = [8189, 3000]
    t = _inputs(device, num_keys, seed=33, rows=PAIR_ROWS)
    pool_blocks = topk_blocks * cbs // BLOCK
    identity = _int([list(range(pool_blocks))], device)
    source = dict(candidate_topk_blocks=topk_blocks, candidate_block_size=cbs)
    torch.manual_seed(34)
    dst_query = torch.randn(1, PAIR_ROWS * HEADS, 1, V4_INDEX_HEAD_DIM)
    dst_query2 = _replicated_rm_hs(dst_query, device)
    dst_query1 = [_replicated_rm_hs(dst_query[:, j * HEADS : (j + 1) * HEADS], device) for j in range(PAIR_ROWS)]

    ref_blocks, ref_counts, ref_rows, ref_cons = [], [], [], []
    for j in range(PAIR_ROWS):
        rows_1, _, blocks_1, count_1 = ttnn.experimental.deepseek.fused_lightning_select_kv(
            t["query"][j],
            t["keys"],
            t["weights"][j],
            t["kv"],
            t["tables"],
            t["pos"][j],
            k,
            page_table_row_tensor=t["page_row"][j],
            **source,
        )
        keys_1, prows_1 = ttnn.experimental.deepseek.build_candidate_pool(
            blocks_1, count_1, t["keys"], t["tables"], cbs, page_table_row_tensor=t["page_row"][j]
        )
        cons_1, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            dst_query1[j],
            keys_1,
            t["weights"][j],
            t["kv"],
            identity,
            t["pos"][j],
            k,
            num_keys_tensor=count_1,
            row_table=prows_1,
        )
        ref_rows.append(ttnn.to_torch(rows_1).reshape(k, -1))
        ref_blocks.append(ttnn.to_torch(blocks_1).reshape(-1))
        ref_counts.append(int(ttnn.to_torch(count_1).reshape(-1)[0]))
        ref_cons.append((ttnn.to_torch(cons_1).reshape(k, -1), ttnn.to_torch(prows_1).reshape(-1)))

    # Source: the four ranks' pages, merged with the rows' candidate blocks and counts.
    pages = []
    for r in range(TP):
        page, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            t["query4"],
            t["keys"],
            t["weights4"],
            t["kv"],
            t["tables"],
            t["pos4"],
            k,
            page_table_row_tensor=t["page_rows4"],
            query_row_tensor=t["query_row"][r % PAIR_ROWS],
            num_parts=PAIR_PARTS,
            part_index_tensor=t["part"][r // PAIR_ROWS],
            **source,
        )
        pages.append(page)
    blocks = _int([[0] * topk_blocks] * PAIR_ROWS, device)
    counts = _int([0] * PAIR_ROWS, device)
    got = _pair_merge(pages, t["kv"], k, candidate_blocks_output=blocks, pool_count_output=counts, **source)
    got_blocks, got_counts = ttnn.to_torch(blocks), ttnn.to_torch(counts).reshape(-1)
    for j in range(PAIR_ROWS):
        n = min(k, num_keys[j])
        assert torch.equal(got[j, :n], ref_rows[j][:n]), f"row {j}: source rows differ"
        assert int(got_counts[j]) == ref_counts[j], f"row {j}: pool count differs"
        assert torch.equal(got_blocks[j], ref_blocks[j]), f"row {j}: candidate blocks differ"

    # Consumers: rank r builds row r % 2's pool and scores part r // 2 of it.
    cpages = []
    for r in range(TP):
        j = r % PAIR_ROWS
        pool_keys, pool_rows = ttnn.experimental.deepseek.build_candidate_pool(
            blocks,
            counts,
            t["keys"],
            t["tables"],
            cbs,
            page_table_row_tensor=t["page_rows4"],
            query_row_tensor=t["query_row"][j],
        )
        n_pool = ref_counts[j]
        assert torch.equal(
            ttnn.to_torch(pool_rows).reshape(-1)[:n_pool], ref_cons[j][1][:n_pool]
        ), f"rank {r}: pool rows differ"
        cpage, _ = ttnn.experimental.deepseek.fused_lightning_select_kv(
            dst_query2,
            pool_keys,
            t["weights4"],
            t["kv"],
            identity,
            t["pos4"],
            k,
            num_keys_tensor=counts,
            row_table=pool_rows,
            query_row_tensor=t["query_row"][j],
            num_parts=PAIR_PARTS,
            part_index_tensor=t["part"][r // PAIR_ROWS],
        )
        cpages.append(cpage)
    got = _pair_merge(cpages, t["kv"], k)
    for j in range(PAIR_ROWS):
        n = min(k, ref_counts[j])
        assert torch.equal(got[j, :n], ref_cons[j][0][:n]), f"row {j}: consumer rows differ"
