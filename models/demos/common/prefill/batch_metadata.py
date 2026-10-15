# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The prefill H2D / D2D metadata wire, mirrored from tt-d-gen's engine/include/engine/pipeline/pipeline_types.hpp.

One chunk per forward ships ``PrefillMetadata``: three uint32 words ``(slot_id, actual_start, actual_end)``.
Several chunks per forward ship ``PrefillBatchMetadata``: ``num_chunks``, a reserved word, then
``MAX_WIRE_CHUNKS`` records of the same three words; record k describes the chunk whose tokens sit at
chunk index k of the payload, and records past ``num_chunks`` carry ``INVALID_SLOT`` with a zero range.
The service's metadata size selects the layout. A sentinel is the first three words all equal to
``SHUTDOWN_WORD`` or ``WARMUP_WORD``, which reads the same way under both layouts.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Iterable, Sequence

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


@dataclass(frozen=True)
class ChunkRecord:
    slot_id: int
    actual_start: int
    actual_end: int


def metadata_size_bytes(max_chunks: int) -> int:
    """Bytes the H2D/D2D services carry per push for ``max_chunks`` chunks per forward."""
    _check_max_chunks(max_chunks)
    return RECORD_BYTES if max_chunks == 1 else BATCH_METADATA_BYTES


def metadata_words(max_chunks: int) -> int:
    return metadata_size_bytes(max_chunks) // 4


def encode_words(records: Sequence[ChunkRecord], max_chunks: int) -> list[int]:
    """The unsigned words of one push carrying ``records``, laid out for ``max_chunks`` chunks per forward."""
    _check_max_chunks(max_chunks)
    if not 1 <= len(records) <= max_chunks:
        raise ValueError(f"{len(records)} records for a forward of at most {max_chunks} chunks")
    for r in records:
        for name, v in (("slot_id", r.slot_id), ("actual_start", r.actual_start), ("actual_end", r.actual_end)):
            if not 0 <= v <= _U32:
                raise ValueError(f"{name}={v} does not fit a uint32")
    if max_chunks == 1:
        r = records[0]
        return [r.slot_id, r.actual_start, r.actual_end]
    words = [len(records), 0]
    for r in records:
        words += [r.slot_id, r.actual_start, r.actual_end]
    words += [INVALID_SLOT, 0, 0] * (MAX_WIRE_CHUNKS - len(records))
    return words


def encode(records: Sequence[ChunkRecord], max_chunks: int) -> bytes:
    words = encode_words(records, max_chunks)
    return struct.pack(f"<{len(words)}I", *words)


def decode(words: Iterable[int], max_chunks: int) -> list[ChunkRecord]:
    """The live records of one push. ``words`` may be signed (a torch int32 view) or unsigned."""
    _check_max_chunks(max_chunks)
    w = [int(x) & _U32 for x in words]
    if len(w) < metadata_words(max_chunks):
        raise ValueError(
            f"{len(w)} metadata words, need {metadata_words(max_chunks)} for {max_chunks} chunks per forward"
        )
    if max_chunks == 1:
        return [ChunkRecord(w[0], w[1], w[2])]
    n = w[0]
    if not 1 <= n <= max_chunks:
        raise ValueError(f"num_chunks={n} outside [1, {max_chunks}]")
    records = []
    for k in range(n):
        base = BATCH_HEADER_WORDS + RECORD_WORDS * k
        records.append(ChunkRecord(w[base], w[base + 1], w[base + 2]))
    return records


def sentinel_words(word: int, max_chunks: int) -> list[int]:
    """Every word of the push set to ``word`` (``SHUTDOWN_WORD`` or ``WARMUP_WORD``)."""
    return [word & _U32] * metadata_words(max_chunks)


def sentinel(word: int, max_chunks: int) -> bytes:
    return struct.pack(f"<{metadata_words(max_chunks)}I", *sentinel_words(word, max_chunks))


def is_sentinel(words: Sequence[int], word: int) -> bool:
    """True when the first three words all read ``word``: the record under the single-chunk layout, or
    ``num_chunks``, the reserved word and the first slot under the batched one. No real push has a
    negative count or a negative slot, so this cannot match a chunk."""
    want = word & _U32
    return len(words) >= RECORD_WORDS and all((int(x) & _U32) == want for x in words[:RECORD_WORDS])


def _check_max_chunks(max_chunks: int) -> None:
    if not 1 <= max_chunks <= MAX_WIRE_CHUNKS:
        raise ValueError(f"max_chunks={max_chunks} outside [1, {MAX_WIRE_CHUNKS}]")
