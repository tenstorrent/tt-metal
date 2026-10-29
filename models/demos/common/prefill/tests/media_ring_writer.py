# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Test-only writer for the media ring that follows the engine's protocol."""

import ctypes
import mmap
import os
import time

from models.demos.common.prefill.runners import media_ring as mr


class EngineWriter:
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
