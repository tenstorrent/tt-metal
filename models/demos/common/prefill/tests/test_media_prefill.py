# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Stage-0 media per chunk: ring -> row source -> injected rows (host-only, no ttnn)."""

import os

import pytest
import torch

from models.demos.common.prefill.media import MediaItem, build_injection, media_rows_for_chunk, stub_row_source
from models.demos.common.prefill.runners.media_prefill import MediaPrefill, row_source_from_env
from models.demos.common.prefill.runners.media_ring import MediaRing
from models.demos.common.prefill.tests.media_ring_writer import EngineWriter

HIDDEN, SP, CHUNK_LOCAL = 32, 2, 32


class RecordingSource:
    def __init__(self):
        self.asked = []
        self.inner = stub_row_source(HIDDEN)

    def __call__(self, item, lo, hi):
        self.asked.append((item.hash, lo, hi))
        return self.inner(item, lo, hi)


@pytest.fixture
def media(request):
    name = f"/tt_media_prefill_pytest_{os.getpid()}_{request.node.name[:40]}"
    ring = MediaRing(name, desc_capacity=16, slab_bytes=1 << 16)
    writer = EngineWriter(name)
    source = RecordingSource()
    mp = MediaPrefill(ring, source, sp=SP, chunk_local=CHUNK_LOCAL, timeout_s=1)
    yield mp, writer, source
    writer.close()
    ring.close()


def meta(slot, start, end, count):
    return {"slot_id": slot, "actual_start": start, "actual_end": end, "media_count": count}


def img(side, seed):
    return bytes((seed * 5 + i) % 256 for i in range(side * side * 3))


def test_text_chunk_is_untouched(media):
    mp, _, source = media
    assert mp.rows_for(meta(0, 0, 64, 0)) is None
    assert source.asked == []


def test_chunk_rows_match_build_injection(media):
    mp, writer, _ = media
    writer.write_chunk(1, 0, [(4, 10, img(14, 1)), (30, 20, img(14, 2))])
    rows, mask = mp.rows_for(meta(1, 0, 64, 2))
    # EngineWriter hashes item i as bytes([i + 1]) * 32.
    items = [MediaItem(4, 10, b"\x01" * 32), MediaItem(30, 20, b"\x02" * 32)]
    want_rows, want_mask = build_injection(
        media_rows_for_chunk(items, 0, 64, stub_row_source(HIDDEN)), 0, 64, SP, CHUNK_LOCAL
    )
    assert torch.equal(mask, want_mask) and torch.equal(rows, want_rows)
    assert int(mask.sum()) == 30


def test_image_spanning_chunks_asks_each_chunk_for_its_rows(media):
    mp, writer, source = media
    pixels = img(14, 7)
    writer.write_chunk(0, 0, [(40, 50, pixels)])
    writer.write_chunk(0, 64, [(40, 50, pixels)])
    first = mp.rows_for(meta(0, 0, 64, 1))
    second = mp.rows_for(meta(0, 64, 128, 1))
    assert [(lo, hi) for _, lo, hi in source.asked] == [(0, 24), (24, 50)]
    assert int(first[1].sum()) == 24 and int(second[1].sum()) == 26


def test_ring_is_released_after_each_chunk(media):
    mp, writer, _ = media
    for k in range(40):  # far more than the 16-record ring
        writer.write_chunk(0, k * 64, [(k * 64, 4, img(10, k))])
        assert mp.rows_for(meta(0, k * 64, k * 64 + 64, 1)) is not None
    assert mp.media_chunks == 40


def test_producer_metadata_size_follows_prefill_media_set_after_import(monkeypatch):
    from models.demos.common.prefill.runners import prefill_producer as producer

    try:
        monkeypatch.setenv("PREFILL_MEDIA", "1")
        producer._load_env_config()
        assert len(producer._pack_metadata(1, 2, 3)) == producer.CHUNK_METADATA_SIZE_BYTES == 16
    finally:
        monkeypatch.setenv("PREFILL_MEDIA", "0")
        producer._load_env_config()
    assert len(producer._pack_metadata(1, 2, 3)) == 12


def test_row_source_needs_the_stub_until_a_real_one_exists(monkeypatch, expect_error):
    monkeypatch.delenv("M3_MEDIA_STUB_EMBEDDINGS", raising=False)
    with expect_error(NotImplementedError, "M3_MEDIA_STUB_EMBEDDINGS"):
        row_source_from_env(HIDDEN)
    monkeypatch.setenv("M3_MEDIA_STUB_EMBEDDINGS", "1")
    assert callable(row_source_from_env(HIDDEN))
