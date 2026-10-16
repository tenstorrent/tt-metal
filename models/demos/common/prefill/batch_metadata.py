# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Prefill H2D/D2D metadata wire, mirroring tt-d-gen's PrefillMetadata and PrefillBatchMetadata."""

from __future__ import annotations

import struct
from typing import Iterable, Sequence

from models.demos.common.prefill.prefill_entry import PrefillEntry

RECORD_WORDS = 3
RECORD_BYTES = 4 * RECORD_WORDS
MAX_WIRE_CHUNKS = 16  # kMaxPrefillChunksPerBatch
BATCH_HEADER_WORDS = 2
BATCH_METADATA_WORDS = BATCH_HEADER_WORDS + RECORD_WORDS * MAX_WIRE_CHUNKS
BATCH_METADATA_BYTES = 4 * BATCH_METADATA_WORDS
INVALID_SLOT = 0xFFFFFFFF
SHUTDOWN_WORD = -1
WARMUP_WORD = -2

_U32 = 0xFFFFFFFF


def metadata_size_bytes(max_chunks: int) -> int:
    _check_max_chunks(max_chunks)
    return RECORD_BYTES if max_chunks == 1 else BATCH_METADATA_BYTES


def metadata_words(max_chunks: int) -> int:
    return metadata_size_bytes(max_chunks) // 4


def encode_words(entries: Sequence[PrefillEntry], max_chunks: int) -> list[int]:
    _check_max_chunks(max_chunks)
    if not 1 <= len(entries) <= max_chunks:
        raise ValueError(f"{len(entries)} entries for a forward of at most {max_chunks} chunks")
    for e in entries:
        for name, v in (("slot", e.slot), ("start", e.start), ("end", e.end)):
            if not 0 <= v <= _U32:
                raise ValueError(f"{name}={v} does not fit a uint32")
    if max_chunks == 1:
        e = entries[0]
        return [e.slot, e.start, e.end]
    words = [len(entries), 0]
    for e in entries:
        words += [e.slot, e.start, e.end]
    words += [INVALID_SLOT, 0, 0] * (MAX_WIRE_CHUNKS - len(entries))
    return words


def encode(entries: Sequence[PrefillEntry], max_chunks: int) -> bytes:
    words = encode_words(entries, max_chunks)
    return struct.pack(f"<{len(words)}I", *words)


def decode(words: Iterable[int], max_chunks: int) -> list[PrefillEntry]:
    """Accepts signed (int32 view) or unsigned words."""
    _check_max_chunks(max_chunks)
    w = [int(x) & _U32 for x in words]
    if len(w) < metadata_words(max_chunks):
        raise ValueError(
            f"{len(w)} metadata words, need {metadata_words(max_chunks)} for {max_chunks} chunks per forward"
        )
    if max_chunks == 1:
        return [PrefillEntry(w[0], w[1], w[2])]
    n = w[0]
    if not 1 <= n <= max_chunks:
        raise ValueError(f"num_chunks={n} outside [1, {max_chunks}]")
    entries = []
    for k in range(n):
        base = BATCH_HEADER_WORDS + RECORD_WORDS * k
        entries.append(PrefillEntry(w[base], w[base + 1], w[base + 2]))
    return entries


def sentinel_words(word: int, max_chunks: int) -> list[int]:
    return [word & _U32] * metadata_words(max_chunks)


def sentinel(word: int, max_chunks: int) -> bytes:
    return struct.pack(f"<{metadata_words(max_chunks)}I", *sentinel_words(word, max_chunks))


def is_sentinel(words: Sequence[int], word: int) -> bool:
    """Checks the first three words, which no real push matches under either layout."""
    want = word & _U32
    return len(words) >= RECORD_WORDS and all((int(x) & _U32) == want for x in words[:RECORD_WORDS])


def _check_max_chunks(max_chunks: int) -> None:
    if not 1 <= max_chunks <= MAX_WIRE_CHUNKS:
        raise ValueError(f"max_chunks={max_chunks} outside [1, {MAX_WIRE_CHUNKS}]")
