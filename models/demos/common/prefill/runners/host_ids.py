# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-side copy of the prefill chunks' token ids (tt-blaze DS41F-0163 WS13): the engine writes each chunk it pushes into the
H2D stream service ALSO into a ring in /dev/shm, so the runner can see the chunks queued behind the one it is computing
(today their ids only exist on the device, and a device read waits for the chunk in flight). The runner uses them only to
START work early (the next chunk's Engram rows); every value that reaches the model still comes from the device tensors,
and an early result is used only when it equals what the chunk computes on arrival.

Writer: tt-d-gen engine/src/pipeline/prefill_host_ids.cpp (TT_DGEN_PREFILL_HOST_IDS=1); ``HostIdsWriter`` below is the same
layout for test producers (the V4.1 device-check driver). Layout of /dev/shm/tt_prefill_ids_<service id>, little-endian:
    header, 64 bytes:  u32 magic, u32 version, u32 n_entries, u32 chunk_size, u64 session, u64 write_seq, 32 bytes reserved
    entry k at 64 + k * entry_bytes (entry_bytes = 32 + 4 * chunk_size, rounded up to 64):
        u64 seq_word, u32 slot_word, u32 actual_start, u32 actual_end, u32 n_ids, u64 reserved, u32 ids[chunk_size]
Chunk number s (0-based, in push order) lives in entry s % n_entries. seq_word is a seqlock: 2s+1 while the writer fills the
entry, 2s+2 once complete; header write_seq = the number of complete entries. A reader takes an entry only if seq_word reads
2s+2 before AND after copying it. session changes whenever a writer (re)initializes the file.
"""

from __future__ import annotations

import mmap
import os
import struct

import numpy as np

MAGIC = 0x53444948  # "HIDS"
VERSION = 1
HEADER_BYTES = 64
ENTRY_HEADER_BYTES = 32
_HDR = struct.Struct("<IIIIQQ")  # magic, version, n_entries, chunk_size, session, write_seq
_ENT = struct.Struct("<QIIII")  # seq_word, slot_word, actual_start, actual_end, n_ids


def host_ids_path(service_id: str) -> str:
    return f"/dev/shm/tt_prefill_ids_{service_id}"


def entry_bytes(chunk_size: int) -> int:
    return (ENTRY_HEADER_BYTES + 4 * chunk_size + 63) // 64 * 64


class HostIdsReader:
    """Read side. ``match`` finds the entry of the chunk the runner just took from the H2D service (by its header), ``entry``
    returns a later one if the engine has written it already. A missing, foreign or re-initialized file never raises:
    lookups return None and the caller computes as before."""

    def __init__(self, path: str):
        self.path = path
        self._mm = None
        self._fd = None
        self._session = None
        self.n = self.chunk = self.ebytes = 0
        self.cursor = 0  # the next chunk number expected

    def _open(self) -> bool:
        if self._mm is not None:
            return True
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC)
        except OSError:
            return False
        try:
            size = os.fstat(fd).st_size
            if size < HEADER_BYTES:
                os.close(fd)
                return False
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ)
        except (OSError, ValueError):
            os.close(fd)
            return False
        magic, version, n, chunk, session, _ = _HDR.unpack_from(mm, 0)
        if magic != MAGIC or version != VERSION or n == 0 or chunk == 0 or size < HEADER_BYTES + n * entry_bytes(chunk):
            mm.close()
            os.close(fd)
            return False
        self._fd, self._mm = fd, mm
        self.n, self.chunk, self.ebytes = n, chunk, entry_bytes(chunk)
        self._session = session
        return True

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            os.close(self._fd)
        self._mm = self._fd = None

    def _header(self):
        """(session, write_seq), re-opening the file if a writer re-created it; None without a usable file."""
        if not self._open():
            return None
        magic, version, n, chunk, session, wseq = _HDR.unpack_from(self._mm, 0)
        if magic != MAGIC or version != VERSION or n != self.n or chunk != self.chunk:
            self.close()  # re-initialized with another geometry (or being initialized): try again next time
            return None
        if session != self._session:
            self._session = session
            self.cursor = 0
        return session, wseq

    def entry(self, s: int):
        """Chunk number ``s`` as (slot_word, actual_start, actual_end, ids[:n_ids] uint32 copy), or None (not written yet,
        overwritten, or being written)."""
        if s < 0 or self._header() is None:
            return None
        off = HEADER_BYTES + (s % self.n) * self.ebytes
        want = 2 * s + 2
        seq1, word, start, end, n_ids = _ENT.unpack_from(self._mm, off)
        if seq1 != want or n_ids > self.chunk:
            return None
        ids = np.frombuffer(self._mm, dtype="<u4", count=n_ids, offset=off + ENTRY_HEADER_BYTES).copy()
        (seq2,) = struct.unpack_from("<Q", self._mm, off)
        if seq2 != seq1:
            return None
        return word, start, end, ids

    def match(self, slot_word: int, actual_start: int, actual_end: int):
        """The chunk number of the chunk with this header: the first such entry at or after the cursor among the last
        n_entries written. Moves the cursor past it. None if there is none."""
        hdr = self._header()
        if hdr is None:
            return None
        wseq = hdr[1]
        lo = max(self.cursor, wseq - self.n)
        for s in range(lo, wseq):
            e = self.entry(s)
            if e is not None and e[0] == slot_word and e[1] == actual_start and e[2] == actual_end:
                self.cursor = s + 1
                return s
        return None


class HostIdsWriter:
    """Write side for test producers (the engine has its own, same layout). Re-initializes the file: a new session."""

    def __init__(self, path: str, chunk_size: int, n_entries: int = 64):
        self.path, self.chunk, self.n = path, int(chunk_size), int(n_entries)
        self.ebytes = entry_bytes(self.chunk)
        size = HEADER_BYTES + self.n * self.ebytes
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
        os.ftruncate(fd, size)
        self._fd = fd
        self._mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
        struct.pack_into("<I", self._mm, 0, 0)  # magic 0 while (re)initializing
        self._mm[4:size] = b"\0" * (size - 4)
        session = int.from_bytes(os.urandom(8), "little") | 1
        _HDR.pack_into(self._mm, 0, 0, VERSION, self.n, self.chunk, session, 0)
        struct.pack_into("<I", self._mm, 0, MAGIC)
        self.seq = 0

    def write(self, slot_word: int, actual_start: int, actual_end: int, ids) -> int:
        ids = np.asarray(ids, dtype="<u4").reshape(-1)
        assert ids.size <= self.chunk, (ids.size, self.chunk)
        s = self.seq
        off = HEADER_BYTES + (s % self.n) * self.ebytes
        struct.pack_into("<Q", self._mm, off, 2 * s + 1)
        _ENT.pack_into(
            self._mm,
            off,
            2 * s + 1,
            slot_word & 0xFFFFFFFF,
            actual_start & 0xFFFFFFFF,
            actual_end & 0xFFFFFFFF,
            ids.size,
        )
        self._mm[off + ENTRY_HEADER_BYTES : off + ENTRY_HEADER_BYTES + 4 * ids.size] = ids.tobytes()
        struct.pack_into("<Q", self._mm, off, 2 * s + 2)
        self.seq = s + 1
        struct.pack_into("<Q", self._mm, 24, self.seq)  # header write_seq
        return s

    def close(self) -> None:
        self._mm.close()
        os.close(self._fd)
