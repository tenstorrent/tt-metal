# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Producer end of the prefill results ring: the first output token of each prompt whose final chunk asked for one.

The engine (tt-d-gen PrefillPipeline) attaches as the consumer. POSIX shared memory, little-endian:
  0   u32 magic 0x31525054, u32 version 1, u32 capacity (power of two), u32 entry_bytes 32
  16  u64 write_count   producer, stored after the entry it publishes
  24  u64 read_count    consumer; the producer never overwrites an entry not yet read
  64  entries[capacity]: u64 seq (write index + 1), u32 slot_id, u32 actual_end, u32 token_id, u32 status (0 ok),
      u64 request_seq
"""

from __future__ import annotations

import mmap
import os
import struct
import time

MAGIC = 0x31525054
VERSION = 1
ENTRY_BYTES = 32
HEADER_BYTES = 64
_HEADER = struct.Struct("<4I")
_COUNT = struct.Struct("<Q")
_ENTRY = struct.Struct("<Q4IQ")


def results_shm_name(ack_shm_name: str) -> str:
    return f"{ack_shm_name}_results"


class PrefillResultRing:
    def __init__(self, name: str, capacity: int = 64):
        assert capacity & (capacity - 1) == 0, "capacity must be a power of two"
        self.name = name
        self.capacity = capacity
        path = "/dev/shm/" + name.lstrip("/")
        if os.path.exists(path):
            os.unlink(path)
        size = HEADER_BYTES + capacity * ENTRY_BYTES
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
            self.buf = mmap.mmap(fd, size)
        finally:
            os.close(fd)
        _COUNT.pack_into(self.buf, 16, 0)
        _COUNT.pack_into(self.buf, 24, 0)
        # The magic goes in last: a consumer that sees it sees a zeroed ring.
        _HEADER.pack_into(self.buf, 0, MAGIC, VERSION, capacity, ENTRY_BYTES)
        self.written = 0

    def push(
        self, slot_id: int, actual_end: int, token_id: int, request_seq: int, status: int = 0, timeout_s: float = 60.0
    ) -> None:
        deadline = time.monotonic() + timeout_s
        while self.written - _COUNT.unpack_from(self.buf, 24)[0] >= self.capacity:
            if time.monotonic() > deadline:
                raise TimeoutError(f"prefill results ring {self.name}: consumer stopped reading")
            time.sleep(0.0005)
        off = HEADER_BYTES + (self.written % self.capacity) * ENTRY_BYTES
        _ENTRY.pack_into(self.buf, off, self.written + 1, slot_id, actual_end, token_id, status, request_seq)
        self.written += 1
        _COUNT.pack_into(self.buf, 16, self.written)

    def close(self) -> None:
        self.buf.close()
