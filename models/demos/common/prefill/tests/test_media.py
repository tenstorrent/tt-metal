# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Media row injection layout from row ranges (host-only, no ttnn)."""

import re

import pytest
import torch

from models.demos.common.prefill.chunk_layout import rotated_chunk_positions
from models.demos.common.prefill.media import (
    MediaItem,
    MediaRows,
    build_injection,
    chunk_row_range,
    items_overlapping,
    media_rows_for_chunk,
    stub_embeddings,
    stub_row_source,
)

HIDDEN = 16


def item(start, n, fill):
    return MediaItem(start, n, bytes([fill]) * 32)


def embeddings_for(items):
    return {m.hash: stub_embeddings(m.hash, m.num_tokens, HIDDEN) for m in items}


def slices(items, embs, start, end):
    return media_rows_for_chunk(items, start, end, lambda m, lo, hi: embs[m.hash][lo:hi])


def whole(items, embs):
    return [MediaRows(m, 0, embs[m.hash]) for m in items]


def scatter_back(rows, mask, chunk_start, sp, chunk_local, seq_len):
    """Global position -> injected row, and which positions were injected."""
    out = torch.zeros(seq_len, HIDDEN, dtype=rows.dtype)
    hit = torch.zeros(seq_len, dtype=torch.bool)
    positions = torch.tensor(rotated_chunk_positions(chunk_start, sp, chunk_local))
    sel = mask & (positions < seq_len)
    out[positions[sel]] = rows[sel]
    hit[positions[sel]] = True
    return out, hit


LAYOUTS = [
    pytest.param([(10, 40)], id="one_image_inside_one_chunk"),
    pytest.param([(3, 20), (40, 30), (100, 12)], id="several_images_one_chunk"),
    pytest.param([(200, 300)], id="image_spans_chunks"),
    pytest.param([(256, 64)], id="image_starts_on_boundary"),
    pytest.param([(5, 10), (120, 400), (600, 8)], id="mixed"),
]


@pytest.mark.parametrize("sp,chunk_local", [(1, 64), (4, 32), (8, 32)])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_chunks_inject_each_image_row_exactly_once(sp, chunk_local, layout):
    items = [item(s, n, i + 1) for i, (s, n) in enumerate(layout)]
    embs = embeddings_for(items)
    chunk = sp * chunk_local
    seq_len = max(m.end_position for m in items) + 17
    expected = torch.zeros(seq_len, HIDDEN, dtype=torch.bfloat16)
    expected_hit = torch.zeros(seq_len, dtype=torch.bool)
    for m in items:
        expected[m.start_position : m.end_position] = embs[m.hash]
        expected_hit[m.start_position : m.end_position] = True

    got = torch.zeros_like(expected)
    hit_count = torch.zeros(seq_len, dtype=torch.int)
    for start in range(0, seq_len, chunk):
        end = min(start + chunk, seq_len)
        res = build_injection(slices(items, embs, start, end), start, end, sp, chunk_local)
        if res is None:
            assert not expected_hit[start:end].any()
            continue
        rows, mask = res
        assert rows.shape == (sp, chunk_local, HIDDEN) and mask.shape == (sp, chunk_local)
        out, hit = scatter_back(rows, mask, start, sp, chunk_local, seq_len)
        got[hit] = out[hit]
        hit_count += hit.int()

    assert torch.equal(hit_count.bool(), expected_hit)
    assert hit_count.max() <= 1
    assert torch.equal(got, expected)


@pytest.mark.parametrize("sp,chunk_local", [(1, 64), (4, 32), (8, 32)])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_exact_slices_match_whole_embeddings(sp, chunk_local, layout):
    items = [item(s, n, i + 1) for i, (s, n) in enumerate(layout)]
    embs = embeddings_for(items)
    chunk = sp * chunk_local
    for start in range(0, max(m.end_position for m in items) + chunk, chunk):
        end = start + chunk
        exact = build_injection(slices(items, embs, start, end), start, end, sp, chunk_local)
        full = build_injection(whole(items_overlapping(items, start, end), embs), start, end, sp, chunk_local)
        assert (exact is None) == (full is None)
        if exact is not None:
            assert torch.equal(exact[0], full[0]) and torch.equal(exact[1], full[1])


def test_chunk_row_range():
    m = item(100, 50, 1)
    assert chunk_row_range(m, 0, 128) == (0, 28)
    assert chunk_row_range(m, 128, 256) == (28, 50)
    assert chunk_row_range(m, 96, 224) == (0, 50)
    assert chunk_row_range(m, 120, 130) == (20, 30)


def test_chunk_row_range_rejects_no_overlap(expect_error):
    with expect_error(AssertionError, "does not overlap"):
        chunk_row_range(item(100, 50, 1), 150, 256)


def test_source_is_asked_only_for_the_chunks_rows():
    items = [item(40, 50, 1)]
    asked = []
    source = stub_row_source(HIDDEN)

    def recording(m, lo, hi):
        asked.append((lo, hi))
        return source(m, lo, hi)

    for start in (0, 64):
        media_rows_for_chunk(items, start, start + 64, recording)
    assert asked == [(0, 24), (24, 50)]


def test_stub_rows_do_not_depend_on_the_range():
    m = item(0, 30, 9)
    source = stub_row_source(HIDDEN)
    assert torch.equal(torch.cat([source(m, 0, 12), source(m, 12, 30)]), stub_embeddings(m.hash, 30, HIDDEN))


def test_missing_row_names_the_item(expect_error):
    items = [item(10, 40, 3)]
    short = MediaRows(items[0], 0, embeddings_for(items)[items[0].hash][:39])
    with expect_error(AssertionError, re.escape("rows [0,39) for MediaItem(start_position=10")):
        build_injection([short], 0, 64, 1, 64)


def test_media_rows_rejects_rows_past_the_item(expect_error):
    with expect_error(AssertionError, re.escape("rows [5,15)")):
        MediaRows(item(0, 10, 1), 5, torch.zeros(10, HIDDEN))


def test_rotated_resume_start():
    sp, chunk_local = 4, 32
    start = 96  # tile-aligned, not chunk-aligned
    items = [item(100, 50, 7)]
    embs = embeddings_for(items)
    rows, mask = build_injection(
        slices(items, embs, start, start + sp * chunk_local), start, start + sp * chunk_local, sp, chunk_local
    )
    out, hit = scatter_back(rows, mask, start, sp, chunk_local, 400)
    assert torch.equal(hit.nonzero().flatten(), torch.arange(100, 150))
    assert torch.equal(out[100:150], embs[items[0].hash])


def test_resume_mid_image_needs_only_the_tail():
    items = [item(50, 100, 5)]
    embs = embeddings_for(items)
    tail = MediaRows(items[0], 30, embs[items[0].hash][30:])
    rows, mask = build_injection([tail], 80, 208, 1, 128)
    assert mask[0, :70].all() and not mask[0, 70:].any()
    assert torch.equal(rows[0, :70], embs[items[0].hash][30:])


def test_rows_past_actual_end_are_not_injected():
    items = [item(50, 100, 3)]
    embs = embeddings_for(items)
    rows, mask = build_injection(slices(items, embs, 0, 80), 0, 80, 1, 128)
    assert mask[0, 50:80].all() and not mask[0, 80:].any() and not mask[0, :50].any()


def test_text_only_chunk_returns_none():
    assert build_injection([], 0, 128, 1, 128) is None
    items = [item(500, 10, 1)]
    assert media_rows_for_chunk(items, 0, 128, stub_row_source(HIDDEN)) == []


def test_items_overlapping_is_half_open():
    a, b = item(0, 64, 1), item(64, 64, 2)
    assert items_overlapping([b, a], 0, 64) == [a]
    assert items_overlapping([b, a], 63, 65) == [a, b]
    assert items_overlapping([b, a], 128, 256) == []


def test_media_item_rejects_bad_hash(expect_error):
    with expect_error(AssertionError, "media hash is 31 bytes"):
        MediaItem(0, 4, b"\x00" * 31)


def test_stub_is_deterministic_per_hash():
    h1, h2 = b"\x01" * 32, b"\x02" * 32
    assert torch.equal(stub_embeddings(h1, 8, HIDDEN), stub_embeddings(h1, 8, HIDDEN))
    assert not torch.equal(stub_embeddings(h1, 8, HIDDEN), stub_embeddings(h2, 8, HIDDEN))
