# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Runner side of the media ring against a writer that follows the engine's protocol (host-only, no ttnn)."""

import ctypes
import mmap
import os
import threading
import time

import pytest

from models.demos.common.prefill.runners import media_ring as mr


class EngineWriter:
    """The engine's write protocol (tt-d-gen MediaRingWriter), for driving the reader."""

    def __init__(self, name):
        fd = os.open("/dev/shm" + name, os.O_RDWR)
        try:
            self.mm = mmap.mmap(fd, os.fstat(fd).st_size)
        finally:
            os.close(fd)
        magic, _, self.cap, _, self.S, _, self.slab_offset = mr._HEADER.unpack_from(self.mm, 0)
        assert magic == mr.MAGIC
        self.c = {off: ctypes.c_uint64.from_buffer(self.mm, off) for off in (64, 128, 192, 256)}

    def write_chunk(self, slot, chunk_start, items, timeout_s=1.0):
        pos, end = [], self.c[192].value
        start = end
        for _, _, pixels in items:
            n = len(pixels)
            if end % self.S + n > self.S:
                end += self.S - end % self.S
            pos.append(end)
            end += n
        assert end - start <= self.S
        head = self.c[64].value
        deadline = time.monotonic() + timeout_s
        while end - self.c[256].value > self.S or head + len(items) - self.c[128].value > self.cap:
            assert time.monotonic() < deadline, "ring stayed full"
            time.sleep(1e-4)
        for i, ((start_pos, n_tok, pixels), p) in enumerate(zip(items, pos)):
            off = self.slab_offset + p % self.S
            self.mm[off : off + len(pixels)] = pixels
            h = w = int((len(pixels) // 3) ** 0.5)
            mr._DESC.pack_into(
                self.mm,
                mr.PAGE + ((head + i) % self.cap) * mr.DESC_BYTES,
                slot,
                chunk_start,
                i,
                len(items),
                start_pos,
                n_tok,
                bytes([i + 1]) * 32,
                h,
                w,
                3,
                h // 14,
                w // 14,
                mr.DTYPE_U8,
                p,
                len(pixels),
            )
        self.c[192].value = end
        self.c[64].value = head + len(items)

    def close(self):
        self.c.clear()
        self.mm.close()


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
