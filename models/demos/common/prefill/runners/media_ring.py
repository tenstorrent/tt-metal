# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Runner side of the media ring: the shm segment the d-gen prefill engine writes each chunk's images into.

The runner creates /tt_prefill_media_<service_id>; the engine attaches and writes. Layout and protocol are in
tt-d-gen docs/multimodal_media_ipc.md and must match engine/include/engine/pipeline/media_ring.hpp byte for byte.
Host-only: no ttnn.
"""

import ctypes
import mmap
import os
import struct
import time
from dataclasses import dataclass

from models.demos.common.prefill.media import MediaItem

MAGIC = 0x4149444D
VERSION = 1
DTYPE_U8 = 0
PAGE = 4096
DESC_BYTES = 96

_HEADER = struct.Struct("<IIIIQQQ")  # magic, version, desc_capacity, desc_bytes, slab_bytes, desc_offset, slab_offset
_DESC = struct.Struct("<6I32s6IQQ")
assert _DESC.size == DESC_BYTES

_DESC_HEAD, _DESC_TAIL, _SLAB_HEAD, _SLAB_TAIL = 64, 128, 192, 256

DEFAULT_DESC_CAPACITY = 256
DEFAULT_SLAB_BYTES = 64 << 20


def media_ring_shm_name(service_id: str) -> str:
    return f"/tt_prefill_media_{service_id}"


def _shm_path(name: str) -> str:
    assert name.startswith("/") and "/" not in name[1:], f"bad shm name {name!r}"
    return "/dev/shm" + name


def _segment_bytes(desc_capacity: int, slab_bytes: int) -> int:
    return -(-(PAGE + desc_capacity * DESC_BYTES) // PAGE) * PAGE + slab_bytes


class MediaRingError(RuntimeError):
    pass


@dataclass(frozen=True)
class MediaDescriptor:
    slot_id: int
    chunk_start: int
    item_index: int
    item_count: int
    start_position: int
    num_tokens: int
    hash: bytes
    height: int
    width: int
    channels: int
    grid_h: int
    grid_w: int
    dtype: int
    payload_pos: int
    payload_bytes: int

    def item(self) -> MediaItem:
        return MediaItem(self.start_position, self.num_tokens, self.hash)


@dataclass(frozen=True)
class MediaPayload:
    desc: MediaDescriptor
    pixels: memoryview  # height*width*channels uint8, a view into the slab; valid until MediaRing.release()


class MediaRing:
    """The runner's end: creates the segment, reads chunks in submit order, releases them."""

    def __init__(self, name: str, desc_capacity: int = DEFAULT_DESC_CAPACITY, slab_bytes: int = DEFAULT_SLAB_BYTES):
        assert desc_capacity > 0 and slab_bytes > 0
        self.name = name
        self.desc_capacity = desc_capacity
        self.slab_bytes = slab_bytes
        self.slab_offset = _segment_bytes(desc_capacity, slab_bytes) - slab_bytes
        path = _shm_path(name)
        if os.path.exists(path):
            os.unlink(path)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, self.slab_offset + slab_bytes)
            self._mm = mmap.mmap(fd, self.slab_offset + slab_bytes)
        finally:
            os.close(fd)
        _HEADER.pack_into(self._mm, 0, 0, VERSION, desc_capacity, DESC_BYTES, slab_bytes, PAGE, self.slab_offset)
        self._counters = {
            off: ctypes.c_uint64.from_buffer(self._mm, off) for off in (_DESC_HEAD, _DESC_TAIL, _SLAB_TAIL)
        }
        ctypes.c_uint32.from_buffer(self._mm, 0).value = MAGIC
        self._held_records = 0
        self._held_slab_end = 0

    def _load(self, off: int) -> int:
        return self._counters[off].value

    def _descriptor(self, index: int) -> MediaDescriptor:
        off = PAGE + (index % self.desc_capacity) * DESC_BYTES
        return MediaDescriptor(*_DESC.unpack_from(self._mm, off))

    def try_read_chunk(self, slot_id: int, chunk_start: int, count: int):
        """This chunk's `count` payloads once all are published, else None. Raises if they belong to another chunk."""
        assert self._held_records == 0, "release() the previous chunk first"
        if count == 0:
            return []
        if count > self.desc_capacity:
            raise MediaRingError(f"{count} media items in one chunk, the ring holds {self.desc_capacity}")
        tail = self._load(_DESC_TAIL)
        if self._load(_DESC_HEAD) - tail < count:
            return None
        descs = [self._descriptor(tail + i) for i in range(count)]
        for i, d in enumerate(descs):
            if (d.slot_id, d.chunk_start, d.item_index, d.item_count) != (slot_id, chunk_start, i, count):
                raise MediaRingError(
                    f"expected item {i}/{count} of slot {slot_id} chunk {chunk_start}, found item "
                    f"{d.item_index}/{d.item_count} of slot {d.slot_id} chunk {d.chunk_start}"
                )
            if (
                d.dtype != DTYPE_U8
                or d.payload_bytes != d.height * d.width * d.channels
                or d.payload_pos % self.slab_bytes + d.payload_bytes > self.slab_bytes
            ):
                raise MediaRingError(f"bad payload for {d}")
        view = memoryview(self._mm)
        out = []
        for d in descs:
            start = self.slab_offset + d.payload_pos % self.slab_bytes
            out.append(MediaPayload(d, view[start : start + d.payload_bytes]))
        self._held_records = count
        self._held_slab_end = out[-1].desc.payload_pos + out[-1].desc.payload_bytes
        return out

    def read_chunk(self, slot_id: int, chunk_start: int, count: int, timeout_s: float):
        """Blocks until the chunk's media is published; raises after timeout_s."""
        deadline = time.monotonic() + timeout_s
        while (got := self.try_read_chunk(slot_id, chunk_start, count)) is None:
            if time.monotonic() >= deadline:
                raise MediaRingError(
                    f"{self.name}: {count} media items for slot {slot_id} chunk {chunk_start} did not arrive "
                    f"in {timeout_s} s"
                )
            time.sleep(50e-6)
        return got

    def release(self) -> None:
        """Frees what the last read returned. Copy any pixels you keep before calling it."""
        if self._held_records == 0:
            return
        self._counters[_SLAB_TAIL].value = self._held_slab_end
        self._counters[_DESC_TAIL].value = self._load(_DESC_TAIL) + self._held_records
        self._held_records = 0

    def close(self, unlink: bool = True) -> None:
        self._counters.clear()
        self._mm.close()
        if unlink and os.path.exists(_shm_path(self.name)):
            os.unlink(_shm_path(self.name))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
