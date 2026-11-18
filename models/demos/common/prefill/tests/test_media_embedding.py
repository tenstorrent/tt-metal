# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: plan_media_splice delivers each media row where pack_forward_tokens puts that position's token."""

import pytest
import torch

from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotated_chunk_positions
from models.demos.common.prefill.media_embedding import _positions, plan_media_splice, table_height, warm_media_refs
from models.demos.common.prefill.prefill_chunk import EMPTY_CHUNK, MediaRef, PrefillChunk

SP, LOCAL, HIDDEN = 4, 32, 8
C = SP * LOCAL


def ref(chunk_index, start, n, key):
    return MediaRef(chunk_index, start, start + n, bytes([key]) * 32)


def u16(t):
    return t.view(torch.int16).numpy().view("uint16")


def table_of(height, runs):
    """The logical [SP, height, HIDDEN] table the runs describe, row 0 zeros."""
    table = torch.zeros(SP, height, HIDDEN, dtype=torch.bfloat16)
    for sp_row, slot, rows in runs:
        table[sp_row, slot : slot + len(rows)] = torch.from_numpy(rows.view("int16")).view(torch.bfloat16)
    return table


def expected(chunks, refs, embs):
    """Rows and mask from the token layout: the cell holding (chunk k, position p) gets p's media row."""
    cells = pack_forward_tokens(
        [[(k, c.actual_start + i) for i in range(C)] for k, c in enumerate(chunks)],
        [c.actual_start for c in chunks],
        SP,
    )
    rows = torch.zeros(len(cells), HIDDEN, dtype=torch.bfloat16)
    mask = torch.zeros(len(cells), dtype=torch.bool)
    for j, (k, p) in enumerate(cells):
        for r in refs:
            if r.chunk_index == k and r.start_pos <= p < min(r.end_pos, chunks[k].actual_end):
                rows[j], mask[j] = embs[r.media_key][p - r.start_pos], True
    return rows.view(SP, -1, HIDDEN), mask.view(SP, -1)


@pytest.mark.parametrize("sp,chunk_local", [(1, 32), (2, 32), (4, 64), (8, 1)])
def test_positions_match_the_chunk_layout(sp, chunk_local):
    for start in range(0, 3 * sp * chunk_local, max(1, chunk_local // 2)):
        assert _positions(start, sp, chunk_local).tolist() == rotated_chunk_positions(start, sp, chunk_local)


@pytest.mark.parametrize(
    "chunks,refs",
    [
        pytest.param([PrefillChunk(0, 0, C)], [ref(0, 10, 40, 1)], id="inside_one_chunk"),
        pytest.param([PrefillChunk(0, 0, C)], [ref(0, 3, 20, 1), ref(0, 40, 30, 2), ref(0, 100, 12, 3)], id="several"),
        pytest.param([PrefillChunk(0, 0, C)], [ref(0, 100, 300, 1)], id="image_head"),
        pytest.param([PrefillChunk(0, C, 2 * C)], [ref(0, 100, 300, 1)], id="image_middle"),
        pytest.param([PrefillChunk(0, 2 * C, 2 * C + 50)], [ref(0, 2 * C + 20, 100, 1)], id="partial_chunk"),
        pytest.param([PrefillChunk(0, 96, 96 + C)], [ref(0, 100, 50, 1)], id="resume_mid_slab"),
        pytest.param(
            [PrefillChunk(0, 0, C), PrefillChunk(1, C, 2 * C), EMPTY_CHUNK],
            [ref(0, 100, 300, 1), ref(1, 2 * C - 10, 30, 3)],
            id="packed_two_slots",
        ),
    ],
)
@pytest.mark.parametrize("by_position", [False, True], ids=["summed_2d", "by_position_1d"])
def test_plan_delivers_the_image_rows(chunks, refs, by_position):
    """2D: summing every SP row's lookup over all positions gives the image rows. 1D: each SP row's lookup over its own
    positions does."""
    embs = {r.media_key: torch.randn(r.num_tokens, HIDDEN).to(torch.bfloat16) for r in refs}
    plan = plan_media_splice(refs, chunks, lambda r: u16(embs[r.media_key]), SP, LOCAL, by_position)
    table, lookup, is_img = table_of(plan.height, plan.runs), plan.lookup, plan.is_img
    want_rows, want_mask = expected(chunks, refs, embs)

    gathered = torch.stack([table[r][lookup[r].long()] for r in range(SP)])
    got = gathered.reshape(-1, HIDDEN) if by_position else gathered.sum(0)

    assert torch.equal(is_img, want_mask.flatten())
    assert torch.equal(got[is_img], want_rows.reshape(-1, HIDDEN)[is_img])
    assert not got[~is_img].any(), "a row outside the mask got a contribution"
    assert (lookup > 0).sum() == is_img.sum(), "every image position is assigned exactly once"


@pytest.mark.parametrize("by_position", [False, True], ids=["2d", "1d"])
def test_plan_gives_contiguous_blocks_per_sp_row(by_position):
    """Each SP row's slots are 1..k with no holes, so filling it is one copy, and the table holds the
    fullest row."""
    chunks, refs = [PrefillChunk(0, 0, C)], [ref(0, 0, SP * 4, 1)]
    embs = {refs[0].media_key: torch.randn(refs[0].num_tokens, HIDDEN).to(torch.bfloat16)}
    plan = plan_media_splice(refs, chunks, lambda r: u16(embs[r.media_key]), SP, LOCAL, by_position)
    lookup = plan.lookup
    for r in range(SP):
        slots = sorted(lookup[r][lookup[r] > 0].tolist())
        assert slots == list(range(1, len(slots) + 1)), (r, slots)
    assert max((lookup[r] > 0).sum().item() for r in range(SP)) + 1 <= plan.height


@pytest.mark.parametrize(
    "rows,want", [(1, 32), (32, 32), (33, 64), (73, 96), (289, 384), (577, 608), (1153, 1216), (2561, 3008)]
)
def test_table_height_snaps_to_the_ladder(rows, want):
    assert table_height(rows) == want


@pytest.mark.parametrize("by_position", [False, True], ids=["2d", "1d"])
def test_table_is_bucketed_and_row_zero_is_never_written(by_position):
    """Row 0 is the "not this row's" sentinel and is read, so no run may write it. Rows above the slots in
    use are not read — lookup never points at them — so the persistent host table may leave them stale."""
    chunks, refs = [PrefillChunk(0, 0, C)], [ref(0, 10, 40, 1)]
    embs = {refs[0].media_key: torch.randn(refs[0].num_tokens, HIDDEN).to(torch.bfloat16)}
    plan = plan_media_splice(refs, chunks, lambda r: u16(embs[r.media_key]), SP, LOCAL, by_position)
    assert plan.height == table_height(int(plan.lookup.max()) + 1)
    assert min(run.slot for run in plan.runs) == 1, "the sentinel row must not be written"


@pytest.mark.parametrize("by_position", [False, True], ids=["2d", "1d"])
@pytest.mark.parametrize("n_chunks", [1, 2, 3])
def test_warm_plan_reaches_the_tallest_table(n_chunks, by_position):
    """compile() warms the heights warm_media_refs names: each plan lands in its height, and the last is
    the tallest table images filling every chunk of a packed forward can need."""
    chunks = [PrefillChunk(k, 0, C) for k in range(n_chunks)]
    zeros = lambda r: u16(torch.zeros(r.num_tokens, HIDDEN, dtype=torch.bfloat16))
    plan = warm_media_refs(SP, C, n_chunks, by_position)
    for height, refs in plan:
        assert plan_media_splice(refs, chunks, zeros, SP, LOCAL, by_position).height == height
    assert plan[-1][0] == table_height(n_chunks * C // SP + 1)
