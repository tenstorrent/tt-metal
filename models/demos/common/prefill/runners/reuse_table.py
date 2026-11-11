# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-slot prefill reuse counts, shared with the tt-d-gen prefill worker (tt-blaze DS41F-0152).

A runtime that reuses part of a prompt itself (V4.1 prefix reuse: chunks retained on device or restored from the host store)
publishes, per slot, how many of the prompt's tokens it did NOT compute. tt-d-gen reads the count when the burst's last chunk
retires and reports it to the decode side as the source's cache hit, so the request's usage `cached_tokens` covers the prefill
side's reuse as well as the ring's. Without the table (a runtime that never reuses) tt-d-gen reads 0, as before.

Layout of /dev/shm/tt_prefill_reuse_<service id>: a 16-byte header (magic, version, slot count, reserved; little-endian
uint32) and one uint32 per slot. The runtime writes a slot's count before that chunk's layer acks, so the count is final by the
time tt-d-gen sees the last chunk's acks. The reader opens the file per read, so a runner restart (a new file) is picked up.
"""

import mmap
import os
import struct

MAGIC = 0x53555254
VERSION = 1
HEADER = struct.Struct("<IIII")
_COUNT = struct.Struct("<I")


def shm_name(service_id: str) -> str:
    return f"/tt_prefill_reuse_{service_id}"


class ReuseTable:
    def __init__(self, name: str, n_slots: int):
        self.path = f"/dev/shm/{name.lstrip('/')}"
        self.n_slots = int(n_slots)
        if os.path.exists(self.path):
            os.remove(self.path)  # a prior run's table: never let the reader see its counts
        size = HEADER.size + _COUNT.size * self.n_slots
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_EXCL, 0o644)
        try:
            os.ftruncate(fd, size)
            self._mm = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        # zeroed by ftruncate; the magic goes in last, so a reader that sees it sees a whole table
        HEADER.pack_into(self._mm, 0, 0, VERSION, self.n_slots, 0)
        HEADER.pack_into(self._mm, 0, MAGIC, VERSION, self.n_slots, 0)

    def set(self, slot: int, tokens: int) -> None:
        assert 0 <= slot < self.n_slots, (slot, self.n_slots)
        _COUNT.pack_into(self._mm, HEADER.size + _COUNT.size * slot, max(0, min(int(tokens), 0xFFFFFFFF)))

    def get(self, slot: int) -> int:
        return _COUNT.unpack_from(self._mm, HEADER.size + _COUNT.size * slot)[0]

    def close(self, unlink: bool = True) -> None:
        self._mm.close()
        if unlink and os.path.exists(self.path):
            os.remove(self.path)
