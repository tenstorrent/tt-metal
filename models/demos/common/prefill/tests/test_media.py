# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only: build_injection puts each media row where pack_forward_tokens puts that position's token."""

import pytest
import torch

from models.demos.common.prefill.chunk_layout import pack_forward_tokens, rotated_chunk_positions
from models.demos.common.prefill.media import _positions, build_injection, build_media_partials, table_height
from models.demos.common.prefill.prefill_chunk import EMPTY_CHUNK, MediaRef, PrefillChunk

SP, LOCAL, HIDDEN = 4, 32, 8
C = SP * LOCAL


def ref(chunk_index, start, n, key):
    return MediaRef(chunk_index, start, start + n, bytes([key]) * 32)


def u16(t):
    return t.view(torch.int16).numpy().view("uint16")


def table_of(height, runs):
    """The [SP, height, HIDDEN] table MediaPartials writes the runs into, row 0 zeros."""
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
            [ref(0, 100, 300, 1), ref(1, 100, 300, 2), ref(1, 2 * C - 10, 30, 3)],
            id="packed_two_slots",
        ),
    ],
)
def test_rows_follow_the_token_layout(chunks, refs):
    embs = {r.media_key: torch.randn(r.num_tokens, HIDDEN).to(torch.bfloat16) for r in refs}
    rows, mask = build_injection(refs, chunks, lambda r: u16(embs[r.media_key]), SP, LOCAL)
    want_rows, want_mask = expected(chunks, refs, embs)
    assert want_mask.any()
    assert torch.equal(mask, want_mask)
    assert torch.equal(rows[mask], want_rows[want_mask])


def test_text_only_forward():
    assert build_injection([], [PrefillChunk(0, 0, C)], None, SP, LOCAL) is None


@pytest.mark.parametrize("sp,chunk_local", [(1, 32), (2, 32), (4, 64), (8, 1)])
def test_positions_match_the_chunk_layout(sp, chunk_local):
    for start in range(0, 3 * sp * chunk_local, max(1, chunk_local // 2)):
        assert _positions(start, sp, chunk_local).tolist() == rotated_chunk_positions(start, sp, chunk_local)


@pytest.mark.parametrize(
    "chunks,refs",
    [
        pytest.param([PrefillChunk(0, 0, C)], [ref(0, 10, 40, 1)], id="inside_one_chunk"),
        pytest.param([PrefillChunk(0, 0, C)], [ref(0, 3, 20, 1), ref(0, 40, 30, 2)], id="several"),
        pytest.param([PrefillChunk(0, C, 2 * C)], [ref(0, 100, 300, 1)], id="image_middle"),
        pytest.param([PrefillChunk(0, 96, 96 + C)], [ref(0, 100, 50, 1)], id="resume_mid_slab"),
        pytest.param(
            [PrefillChunk(0, 0, C), PrefillChunk(1, C, 2 * C), EMPTY_CHUNK],
            [ref(0, 100, 300, 1), ref(1, 2 * C - 10, 30, 3)],
            id="packed_two_slots",
        ),
    ],
)
def test_partials_sum_to_the_image_rows(chunks, refs):
    """D: summing every SP row's partial reproduces build_injection's rows at the image positions."""
    embs = {r.media_key: torch.randn(r.num_tokens, HIDDEN).to(torch.bfloat16) for r in refs}
    rows_of = lambda r: u16(embs[r.media_key])
    height, lookup, is_img, runs = build_media_partials(refs, chunks, rows_of, SP, LOCAL)
    table = table_of(height, runs)
    want_rows, want_mask = build_injection(refs, chunks, rows_of, SP, LOCAL)

    summed = torch.zeros(lookup.shape[1], HIDDEN, dtype=torch.bfloat16)  # s_total, not one chunk
    for r in range(SP):
        summed += table[r][lookup[r].long()]

    assert torch.equal(is_img, want_mask.flatten())
    assert torch.equal(summed[is_img], want_rows.reshape(-1, HIDDEN)[is_img])
    assert not summed[~is_img].any(), "a row outside the mask got a contribution"
    assert (lookup > 0).sum() == is_img.sum(), "every image position is assigned exactly once"


def test_partials_are_contiguous_blocks_per_sp_row():
    """Each SP row's slots are 1..k with no holes, so filling it is one copy. A row taking a full block
    bounds the table height; rows past the last block take none."""
    chunks, refs = [PrefillChunk(0, 0, C)], [ref(0, 0, SP * 4, 1)]
    embs = {refs[0].media_key: torch.randn(refs[0].num_tokens, HIDDEN).to(torch.bfloat16)}
    height, lookup, _, _ = build_media_partials(refs, chunks, lambda r: u16(embs[r.media_key]), SP, LOCAL)
    for r in range(SP):
        slots = sorted(lookup[r][lookup[r] > 0].tolist())
        assert slots == list(range(1, len(slots) + 1)), (r, slots)
    assert max((lookup[r] > 0).sum().item() for r in range(SP)) + 1 <= height


def test_partials_text_only():
    assert build_media_partials([], [PrefillChunk(0, 0, C)], None, SP, LOCAL) is None


@pytest.mark.parametrize(
    "rows,want", [(1, 32), (32, 32), (33, 64), (73, 96), (289, 384), (577, 608), (1153, 1216), (2561, 3008)]
)
def test_table_height_snaps_to_the_ladder(rows, want):
    assert table_height(rows) == want


def test_table_is_bucketed_and_row_zero_is_never_written():
    """Row 0 is the "not this row's" sentinel and is read, so no run may write it. Rows above the slots in
    use are not read — lookup never points at them — so the persistent host table may leave them stale."""
    chunks, refs = [PrefillChunk(0, 0, C)], [ref(0, 10, 40, 1)]
    embs = {refs[0].media_key: torch.randn(refs[0].num_tokens, HIDDEN).to(torch.bfloat16)}
    height, lookup, _, runs = build_media_partials(refs, chunks, lambda r: u16(embs[r.media_key]), SP, LOCAL)
    assert height == table_height(int(lookup.max()) + 1)
    assert min(slot for _, slot, _ in runs) == 1, "the sentinel row must not be written"
