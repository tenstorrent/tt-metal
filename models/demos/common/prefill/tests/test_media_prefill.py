# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 media per chunk: ring -> encoder cache -> injected rows (host-only, no ttnn)."""

import os

import pytest
import torch

from models.demos.common.prefill.media import MediaItem, build_injection, stub_embeddings
from models.demos.common.prefill.runners.media_prefill import MediaPrefill, encoder_from_env, stub_encoder
from models.demos.common.prefill.runners.media_ring import MediaRing
from models.demos.common.prefill.tests.media_ring_writer import EngineWriter

HIDDEN, SP, CHUNK_LOCAL = 32, 2, 32


class CountingEncoder:
    def __init__(self):
        self.seen = []
        self.inner = stub_encoder(HIDDEN)

    def __call__(self, payload):
        self.seen.append(bytes(payload.pixels))
        return self.inner(payload)


@pytest.fixture
def media(request):
    name = f"/tt_media_prefill_pytest_{os.getpid()}_{request.node.name[:40]}"
    ring = MediaRing(name, desc_capacity=16, slab_bytes=1 << 16)
    writer = EngineWriter(name)
    enc = CountingEncoder()
    mp = MediaPrefill(ring, enc, sp=SP, chunk_local=CHUNK_LOCAL, cache_bytes=1 << 20, timeout_s=1)
    yield mp, writer, enc
    writer.close()
    ring.close()


def meta(slot, start, end, count):
    return {"slot_id": slot, "actual_start": start, "actual_end": end, "media_count": count}


def img(side, seed):
    return bytes((seed * 5 + i) % 256 for i in range(side * side * 3))


def test_text_chunk_is_untouched(media):
    mp, _, enc = media
    assert mp.rows_for(meta(0, 0, 64, 0)) is None
    assert enc.seen == []


def test_chunk_rows_match_build_injection(media):
    mp, writer, _ = media
    writer.write_chunk(1, 0, [(4, 10, img(14, 1)), (30, 20, img(14, 2))])
    rows, mask = mp.rows_for(meta(1, 0, 64, 2))
    # EngineWriter hashes item i as bytes([i + 1]) * 32.
    items = [MediaItem(4, 10, b"\x01" * 32), MediaItem(30, 20, b"\x02" * 32)]
    embs = {m.hash: stub_embeddings(m.hash, m.num_tokens, HIDDEN) for m in items}
    want_rows, want_mask = build_injection(embs, items, 0, 64, SP, CHUNK_LOCAL)
    assert torch.equal(mask, want_mask) and torch.equal(rows, want_rows)
    assert int(mask.sum()) == 30


def test_image_spanning_chunks_is_encoded_once(media):
    """The second chunk carries the same image again; the encoder cache answers it without the pixels."""
    mp, writer, enc = media
    pixels = img(14, 7)
    writer.write_chunk(0, 0, [(40, 50, pixels)])
    writer.write_chunk(0, 64, [(40, 50, pixels)])
    first = mp.rows_for(meta(0, 0, 64, 1))
    second = mp.rows_for(meta(0, 64, 128, 1))
    assert enc.seen == [pixels]
    assert int(first[1].sum()) == 24 and int(second[1].sum()) == 26
    assert (mp.cache.hits, mp.cache.misses) == (1, 1)


def test_ring_is_released_after_each_chunk(media):
    mp, writer, _ = media
    for k in range(40):  # far more than the 16-record ring
        writer.write_chunk(0, k * 64, [(k * 64, 4, img(10, k))])
        assert mp.rows_for(meta(0, k * 64, k * 64 + 64, 1)) is not None
    assert mp.media_chunks == 40


def test_encoder_needs_the_stub_until_the_real_one_lands(monkeypatch, expect_error):
    monkeypatch.delenv("M3_MEDIA_STUB_EMBEDDINGS", raising=False)
    with expect_error(NotImplementedError, "M3_MEDIA_STUB_EMBEDDINGS"):
        encoder_from_env(HIDDEN)
    monkeypatch.setenv("M3_MEDIA_STUB_EMBEDDINGS", "1")
    assert callable(encoder_from_env(HIDDEN))
