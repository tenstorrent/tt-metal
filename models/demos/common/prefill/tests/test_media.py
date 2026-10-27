# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Media row injection layout and the encoder cache (host-only, no ttnn)."""

import pytest
import torch

from models.demos.common.prefill.chunk_layout import rotated_chunk_positions
from models.demos.common.prefill.media import (
    EncoderCache,
    MediaItem,
    build_injection,
    items_overlapping,
    stub_embeddings,
)

HIDDEN = 16


def item(start, n, fill):
    return MediaItem(start, n, bytes([fill]) * 32)


def embeddings_for(items):
    return {m.hash: stub_embeddings(m.hash, m.num_tokens, HIDDEN) for m in items}


def scatter_back(rows, mask, chunk_start, sp, chunk_local, seq_len):
    """Undo the rotation: global position -> injected row, plus which positions were injected."""
    out = torch.zeros(seq_len, HIDDEN, dtype=rows.dtype)
    hit = torch.zeros(seq_len, dtype=torch.bool)
    positions = torch.tensor(rotated_chunk_positions(chunk_start, sp, chunk_local))
    sel = mask & (positions < seq_len)
    out[positions[sel]] = rows[sel]
    hit[positions[sel]] = True
    return out, hit


@pytest.mark.parametrize("sp,chunk_local", [(1, 64), (4, 32), (8, 32)])
@pytest.mark.parametrize(
    "layout",
    [
        pytest.param([(10, 40)], id="one_image_inside_one_chunk"),
        pytest.param([(3, 20), (40, 30), (100, 12)], id="several_images_one_chunk"),
        pytest.param([(200, 300)], id="image_spans_chunks"),
        pytest.param([(256, 64)], id="image_starts_on_boundary"),
        pytest.param([(5, 10), (120, 400), (600, 8)], id="mixed"),
    ],
)
def test_chunks_inject_each_image_row_exactly_once(sp, chunk_local, layout):
    """Every image position is written once, by the chunk that holds it, with its own row; text positions never."""
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
        res = build_injection(embs, items, start, end, sp, chunk_local)
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


def test_rotated_resume_start():
    """A chunk that starts mid-slab (multi-turn resume) still lands each row at its true position."""
    sp, chunk_local = 4, 32
    start = 96  # tile-aligned, not chunk-aligned
    items = [item(100, 50, 7)]
    embs = embeddings_for(items)
    rows, mask = build_injection(embs, items, start, start + sp * chunk_local, sp, chunk_local)
    out, hit = scatter_back(rows, mask, start, sp, chunk_local, 400)
    assert torch.equal(hit.nonzero().flatten(), torch.arange(100, 150))
    assert torch.equal(out[100:150], embs[items[0].hash])


def test_rows_past_actual_end_are_not_injected():
    """The last chunk's padding rows stay text even if an image would extend there."""
    items = [item(50, 100, 3)]
    embs = embeddings_for(items)
    rows, mask = build_injection(embs, items, 0, 80, 1, 128)
    assert mask[0, 50:80].all() and not mask[0, 80:].any() and not mask[0, :50].any()


def test_text_only_chunk_returns_none():
    assert build_injection({}, [], 0, 128, 1, 128) is None
    items = [item(500, 10, 1)]
    assert build_injection(embeddings_for(items), items, 0, 128, 1, 128) is None


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


def _bytes_of(n_tokens):
    return n_tokens * HIDDEN * 2  # bf16


class CountingEncoder:
    def __init__(self, n_tokens=8):
        self.calls = 0
        self.n_tokens = n_tokens

    def __call__(self, media_hash):
        def run():
            self.calls += 1
            return stub_embeddings(media_hash, self.n_tokens, HIDDEN)

        return run


def test_cache_hit_skips_encoder_and_is_bit_identical():
    """(a) a re-prefill after partial KV eviction hits and skips the encoder; (e) hit and miss give the same rows."""
    cache, enc, h = EncoderCache(_bytes_of(8) * 4), CountingEncoder(), b"\x0a" * 32
    miss = cache.get_or_compute(h, enc(h))
    hit = cache.get_or_compute(h, enc(h))
    assert enc.calls == 1
    assert torch.equal(miss, hit)
    assert (cache.hits, cache.misses) == (1, 1)


def test_cache_is_keyed_by_media_hash_only():
    """(b) the same image in another session, with other text and cache_salt, hits: the key carries neither.
    (c) a different image with the same text misses."""
    cache, enc = EncoderCache(_bytes_of(8) * 4), CountingEncoder()
    img_a, img_b = b"\xaa" * 32, b"\xbb" * 32
    cache.get_or_compute(img_a, enc(img_a))
    cache.get_or_compute(img_a, enc(img_a))
    assert enc.calls == 1
    cache.get_or_compute(img_b, enc(img_b))
    assert enc.calls == 2
    assert (cache.hits, cache.misses) == (1, 2)


def test_cache_evicts_lru_at_capacity():
    """(d) eviction happens at the configured byte capacity, oldest-used first."""
    cache, enc = EncoderCache(_bytes_of(8) * 2), CountingEncoder()
    a, b, c = b"\x01" * 32, b"\x02" * 32, b"\x03" * 32
    cache.get_or_compute(a, enc(a))
    cache.get_or_compute(b, enc(b))
    cache.get_or_compute(a, enc(a))  # a is now most recent
    cache.get_or_compute(c, enc(c))
    assert cache.evictions == 1
    assert a in cache and c in cache and b not in cache
    assert cache.bytes == _bytes_of(8) * 2 <= cache.capacity_bytes


def test_cache_skips_items_larger_than_capacity():
    cache, enc = EncoderCache(_bytes_of(8) - 1), CountingEncoder()
    h = b"\x05" * 32
    cache.get_or_compute(h, enc(h))
    cache.get_or_compute(h, enc(h))
    assert enc.calls == 2 and len(cache) == 0 and cache.evictions == 0


def test_stats_line_is_parseable():
    cache = EncoderCache(1024)
    line = cache.stats_line()
    assert line.startswith("ENCODER_CACHE ")
    fields = dict(kv.split("=") for kv in line.split()[1:])
    assert set(fields) == {"hits", "misses", "evictions", "entries", "bytes", "capacity_bytes"}
