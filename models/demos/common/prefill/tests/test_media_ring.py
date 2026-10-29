# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Runner side of the media ring against a writer that follows the engine's protocol (host-only, no ttnn)."""

import os
import threading
import time

import pytest

from models.demos.common.prefill.runners import media_ring as mr
from models.demos.common.prefill.tests.media_ring_writer import EngineWriter


def pixels(side, seed):
    return bytes((seed + 31 * i) % 256 for i in range(side * side * 3))


@pytest.fixture
def ring(request):
    name = f"/tt_media_ring_pytest_{os.getpid()}_{request.node.name[:40]}".replace("[", "_").replace("]", "_")
    params = getattr(request, "param", {})
    r = mr.MediaRing(name, **params)
    w = EngineWriter(name)
    yield r, w
    w.close()
    r.close()
    assert not os.path.exists("/dev/shm" + name)


def test_reads_a_chunk_back_exactly(ring):
    r, w = ring
    a, b = pixels(28, 1), pixels(14, 2)
    w.write_chunk(3, 64, [(70, 64, a), (140, 16, b)])
    got = r.try_read_chunk(3, 64, 2)
    assert [bytes(p.pixels) for p in got] == [a, b]
    d = got[0].desc
    assert (d.slot_id, d.chunk_start, d.item_index, d.item_count) == (3, 64, 0, 2)
    assert (d.height, d.width, d.channels, d.grid_h, d.grid_w) == (28, 28, 3, 2, 2)
    assert got[1].desc.item() == mr.MediaItem(140, 16, b"\x02" * 32)
    r.release()


def test_text_chunk_never_waits(ring):
    r, _ = ring
    assert r.try_read_chunk(0, 0, 0) == []
    assert r.read_chunk(0, 0, 0, timeout_s=0) == []


def test_unpublished_media_is_not_visible(ring):
    r, _ = ring
    assert r.try_read_chunk(0, 0, 1) is None


@pytest.mark.parametrize("ring", [{"desc_capacity": 4, "slab_bytes": 1000}], indirect=True)
def test_payloads_wrap_the_slab(ring):
    r, w = ring
    for k in range(25):
        img = pixels(10, k)  # 300 bytes in a 1000-byte slab
        w.write_chunk(0, k * 64, [(k * 64, 4, img)])
        (got,) = r.try_read_chunk(0, k * 64, 1)
        assert bytes(got.pixels) == img
        assert got.desc.payload_pos % 1000 + 300 <= 1000
        r.release()


def test_media_for_another_chunk_is_fatal(ring, expect_error):
    r, w = ring
    w.write_chunk(2, 128, [(130, 4, pixels(10, 1))])
    with expect_error(mr.MediaRingError, "expected item 0/1 of slot 2 chunk 0"):
        r.try_read_chunk(2, 0, 1)


def test_waits_for_media_published_after_its_metadata(ring):
    r, w = ring
    img = pixels(14, 9)

    def late_writer():
        time.sleep(0.02)
        w.write_chunk(1, 0, [(5, 8, img)])

    t = threading.Thread(target=late_writer)
    t.start()
    t0 = time.monotonic()
    (got,) = r.read_chunk(1, 0, 1, timeout_s=2)
    t.join()
    assert time.monotonic() - t0 >= 0.015
    assert bytes(got.pixels) == img


def test_missing_media_times_out(ring, expect_error):
    r, _ = ring
    with expect_error(mr.MediaRingError, "did not arrive"):
        r.read_chunk(0, 0, 1, timeout_s=0.01)


def test_more_items_than_the_ring_holds_is_fatal(ring, expect_error):
    r, _ = ring
    with expect_error(mr.MediaRingError, "ring holds 256"):
        r.read_chunk(0, 0, 257, timeout_s=0.01)


@pytest.mark.parametrize("ring", [{"desc_capacity": 8, "slab_bytes": 700}], indirect=True)
def test_release_lets_the_writer_reuse_the_slab(ring):
    r, w = ring
    w.write_chunk(0, 0, [(0, 1, pixels(10, 1))])
    w.write_chunk(0, 64, [(64, 1, pixels(10, 2))])
    done = threading.Event()
    t = threading.Thread(target=lambda: (w.write_chunk(0, 128, [(128, 1, pixels(10, 3))], 5), done.set()))
    t.start()
    assert not done.wait(0.05), "the writer overwrote a payload the runner still holds"
    (first,) = r.try_read_chunk(0, 0, 1)
    assert bytes(first.pixels) == pixels(10, 1)
    r.release()
    assert done.wait(5)
    t.join()


def test_create_replaces_a_stale_segment(ring):
    r, w = ring
    w.write_chunk(0, 0, [(0, 1, pixels(10, 1))])
    fresh = mr.MediaRing(r.name)
    try:
        assert fresh.try_read_chunk(0, 0, 1) is None
    finally:
        fresh.close(unlink=False)
