# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunks of a packed prefill forward and their H2D / D2D metadata, tt-d-gen's PrefillBatchMetadata.

A forward of ``forward_tokens_budget`` rows holds ``forward_chunks = forward_tokens_budget // chunk_size`` (F)
chunks in pack order. At F = 1 its metadata is the single-chunk ``(slot_id, actual_start, actual_end)`` record,
``struct.pack("<3I", ...)``. Above that it is ``{num_chunks, reserved}`` followed by F records, little-endian
uint32, ``8 + 12 * F`` bytes; records past ``num_chunks`` are ``(INVALID_SLOT, 0, 0)``. Sentinels fill every word
with ``SHUTDOWN_WORD`` / ``WARMUP_WORD``.

Pure Python: the runner and host tests import it without a device.
"""

import struct
from dataclasses import dataclass

CHUNK_WORDS = 3
CHUNK_BYTES = 4 * CHUNK_WORDS
HEADER_WORDS = 2
INVALID_SLOT = 0xFFFFFFFF
SHUTDOWN_WORD = -1
WARMUP_WORD = -2
# The KV-cache writer's tile grid: a chunk starts on a tile row.
CHUNK_START_ALIGN = 32


@dataclass(frozen=True)
class PrefillChunk:
    """[actual_start, actual_end): the chunk's real tokens, half-open, in absolute KV positions of slot ``slot_id``.
    actual_start is also the prefix already cached (the attention core's ``cached_len``)."""

    slot_id: int  # KV-cache slot of the request
    actual_start: int  # absolute KV position of the chunk's first row
    actual_end: int  # past the last real token; == actual_start is an empty chunk (no attention, no KV write)

    @property
    def is_empty(self) -> bool:
        return self.actual_end == self.actual_start

    @property
    def is_sentinel(self) -> bool:
        return self.slot_id in (SHUTDOWN_WORD, WARMUP_WORD) and self.actual_start == self.actual_end == self.slot_id


EMPTY_CHUNK = PrefillChunk(0, 0, 0)


def metadata_size_bytes(forward_chunks: int) -> int:
    if forward_chunks < 1:
        raise ValueError(
            f"a forward of {forward_chunks} chunks; PREFILL_FORWARD_TOKENS_BUDGET must be >= the chunk size"
        )
    return CHUNK_BYTES if forward_chunks == 1 else 4 * (HEADER_WORDS + CHUNK_WORDS * forward_chunks)


def pad_chunks(chunks, forward_chunks: int) -> list:
    """``chunks`` followed by empty chunks up to ``forward_chunks``."""
    chunks = list(chunks)
    if len(chunks) > forward_chunks:
        raise ValueError(f"{len(chunks)} chunks do not fit a forward of {forward_chunks}")
    return chunks + [EMPTY_CHUNK] * (forward_chunks - len(chunks))


def forward_metadata_words(chunks, forward_chunks: int) -> list:
    """The metadata of a forward of ``chunks`` (empty ones at the tail) as int32 words, as the device tensor
    holds them."""
    if not chunks:
        raise ValueError("a forward needs at least one chunk")
    if len(chunks) > forward_chunks:
        raise ValueError(f"{len(chunks)} chunks do not fit a forward of {forward_chunks}")
    c = chunks[0]
    if c.is_sentinel:
        return sentinel_words(c.slot_id, forward_chunks)
    if forward_chunks == 1:
        return [c.slot_id, c.actual_start, c.actual_end]
    real = [c for c in chunks if not c.is_empty]
    if not real:
        raise ValueError(f"a forward of only empty chunks: {chunks}")
    if chunks[: len(real)] != real:
        raise ValueError(f"empty chunks must come after every non-empty one: {chunks}")
    words = [len(real), 0] + [w for c in real for w in (c.slot_id, c.actual_start, c.actual_end)]
    words += [INVALID_SLOT, 0, 0] * (forward_chunks - len(real))
    return [w - (1 << 32) if w >= 1 << 31 else w for w in words]


def _pack_words(words) -> bytes:
    return struct.pack(f"<{len(words)}i", *words)


def pack_forward_metadata(chunks, forward_chunks: int) -> bytes:
    return _pack_words(forward_metadata_words(chunks, forward_chunks))


def sentinel_words(word: int, forward_chunks: int) -> list:
    return [word] * (metadata_size_bytes(forward_chunks) // 4)


def sentinel_metadata(word: int, forward_chunks: int) -> bytes:
    return _pack_words(sentinel_words(word, forward_chunks))


def decode_forward_metadata(words, forward_chunks: int) -> list:
    """The forward's ``forward_chunks`` chunks, empty ones at the tail, from its metadata words read as int32.
    A sentinel decodes to chunks of its word. Words past the metadata (MTP's provided-levels word on D2D) are
    ignored."""
    n = metadata_size_bytes(forward_chunks) // 4
    if len(words) < n:
        raise ValueError(f"metadata has {len(words)} words, a forward of {forward_chunks} chunks needs {n}")
    head = PrefillChunk(*words[:CHUNK_WORDS])
    if forward_chunks == 1 or head.is_sentinel:
        return [head] * forward_chunks
    num_chunks = words[0]
    if not 1 <= num_chunks <= forward_chunks:
        raise ValueError(f"num_chunks={num_chunks} outside [1, {forward_chunks}]")
    # A producer still sending one chunk per forward fails these instead of decoding garbage.
    if words[1] != 0:
        raise ValueError(f"reserved header word is {words[1]}, not 0: is the producer sending {forward_chunks} chunks?")
    records = [
        words[HEADER_WORDS + CHUNK_WORDS * k : HEADER_WORDS + CHUNK_WORDS * (k + 1)] for k in range(forward_chunks)
    ]
    for k in range(num_chunks, forward_chunks):
        if records[k][0] & 0xFFFFFFFF != INVALID_SLOT or records[k][1:] != [0, 0]:
            raise ValueError(f"record {k} past num_chunks={num_chunks} is {records[k]}, not an unused record")
    return [PrefillChunk(*r) for r in records[:num_chunks]] + [EMPTY_CHUNK] * (forward_chunks - num_chunks)


def validate_chunks(chunks, *, chunk_size: int, forward_chunks: int, num_users: int, max_seq_len: int) -> None:
    """The order contract of one forward.

    Per slot, its chunks are one contiguous run of consecutive chunks and only the run's last chunk may be
    partial. Empty chunks come after every non-empty one, and at least one chunk is non-empty. Runs of
    different slots may come in any order.
    """
    if len(chunks) != forward_chunks:
        raise ValueError(f"a forward holds {forward_chunks} chunks, got {len(chunks)}")
    if chunks[0].is_empty:
        raise ValueError("a forward needs at least one non-empty chunk (empties go at the tail)")
    prev = None
    seen = set()
    for c in chunks:
        if c.is_empty:
            prev = c
            continue
        if prev is not None and prev.is_empty:
            raise ValueError(f"{c}: non-empty chunk after an empty one")
        if not 0 <= c.slot_id < num_users:
            raise ValueError(f"{c}: slot out of range [0, {num_users})")
        if c.actual_start < 0:
            raise ValueError(f"{c}: negative start")
        if c.actual_start % CHUNK_START_ALIGN:
            raise ValueError(f"{c}: start must be a multiple of {CHUNK_START_ALIGN}")
        if not c.actual_start < c.actual_end <= c.actual_start + chunk_size:
            raise ValueError(f"{c}: more than one {chunk_size}-token chunk")
        if c.actual_start + chunk_size > max_seq_len:
            raise ValueError(f"{c}: chunk exceeds the per-user cache {max_seq_len}")
        if c.slot_id in seen:
            if prev.slot_id != c.slot_id:
                raise ValueError(f"{c}: slot {c.slot_id} appears in two separate runs")
            if prev.actual_end != prev.actual_start + chunk_size:
                raise ValueError(f"{prev}: a partial chunk must end its slot's run")
            if c.actual_start != prev.actual_end:
                raise ValueError(f"{c}: does not continue {prev}")
        seen.add(c.slot_id)
        prev = c


def packed_forward_refusals(
    forward_chunks: int,
    *,
    supports_packed_forward: bool,
    mtp_levels: int = 0,
    dflash: bool = False,
    use_trace: bool = False,
    d2h_acks: bool = False,
    indexer_indices: bool = False,
) -> list:
    """Why a configuration cannot run ``forward_chunks`` chunks per forward (empty list: it can)."""
    if forward_chunks == 1:
        return []
    reasons = []
    if not supports_packed_forward:
        reasons.append("the model adapter does not set supports_packed_forward")
    if mtp_levels:
        reasons.append("PREFILL_MTP_LEVELS > 0 (MTP lookahead is per chunk)")
    if dflash:
        reasons.append("PREFILL_DFLASH=1")
    if use_trace:
        reasons.append("PREFILL_USE_TRACE=1 (per-chunk arguments are not trace inputs)")
    if d2h_acks:
        reasons.append("PREFILL_LAYER_ACK_D2H=1")
    if indexer_indices:
        reasons.append(
            "a multi-rank runtime passes indexer indices between ranks, which a packed forward does not carry"
        )
    return reasons


def check_packed_forward_support(forward_chunks: int, **kwargs) -> None:
    reasons = packed_forward_refusals(forward_chunks, **kwargs)
    if reasons:
        raise ValueError(
            f"PREFILL_FORWARD_TOKENS_BUDGET of {forward_chunks} chunks is not supported here: " + "; ".join(reasons)
        )
