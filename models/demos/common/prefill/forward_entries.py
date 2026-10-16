# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunk entries of a packed prefill forward and their H2D / D2D metadata.

A forward holds ``forward_chunks`` (F) entries of ``chunk_size`` tokens each. Its metadata is F little-endian
``(slot, start, end)`` uint32 triples (12 * F bytes); F = 1 is the single-chunk ``struct.pack("<3I", ...)``.
``[start, end)`` is the absolute KV-position range of the entry's real tokens, and ``start == end`` is an empty
entry (no attention, no KV write). Sentinels fill every word with ``SHUTDOWN_WORD`` / ``WARMUP_WORD``.

Pure Python: the producer, the runner and host tests import it without a device.
"""

import struct
from dataclasses import dataclass

ENTRY_WORDS = 3
ENTRY_BYTES = 4 * ENTRY_WORDS
SHUTDOWN_WORD = -1
WARMUP_WORD = -2
# The KV-cache writer's tile grid: an entry starts on a tile row.
ENTRY_START_ALIGN = 32


@dataclass(frozen=True)
class PrefillEntry:
    slot: int  # KV-cache user slot
    start: int  # absolute KV position of the chunk's first row (the prefix already cached)
    end: int  # past the last real token; start == end is an empty entry (no attention, no KV write)

    @property
    def is_empty(self) -> bool:
        return self.end == self.start


EMPTY_ENTRY = PrefillEntry(0, 0, 0)


def metadata_size_bytes(forward_chunks: int) -> int:
    return ENTRY_BYTES * forward_chunks


def pad_entries(entries, forward_chunks: int) -> list:
    """``entries`` followed by empty entries up to ``forward_chunks``."""
    entries = list(entries)
    if len(entries) > forward_chunks:
        raise ValueError(f"{len(entries)} entries do not fit a forward of {forward_chunks}")
    return entries + [EMPTY_ENTRY] * (forward_chunks - len(entries))


def pack_forward_metadata(entries) -> bytes:
    return struct.pack("<" + "3I" * len(entries), *(w for e in entries for w in (e.slot, e.start, e.end)))


def sentinel_words(word: int, forward_chunks: int) -> list:
    return [word] * (ENTRY_WORDS * forward_chunks)


def sentinel_metadata(word: int, forward_chunks: int) -> bytes:
    words = sentinel_words(word, forward_chunks)
    return struct.pack(f"<{len(words)}i", *words)


def decode_forward_metadata(words, forward_chunks: int) -> list:
    """The forward's entries from its metadata words read as int32. Words past the F triples (MTP's
    provided-levels word on D2D) are ignored."""
    words = [int(w) for w in words]
    n = ENTRY_WORDS * forward_chunks
    if len(words) < n:
        raise ValueError(f"metadata has {len(words)} words, a forward of {forward_chunks} entries needs {n}")
    return [PrefillEntry(*words[i : i + ENTRY_WORDS]) for i in range(0, n, ENTRY_WORDS)]


def validate_forward_entries(entries, *, chunk_size: int, forward_chunks: int, num_users: int, max_seq_len: int):
    """The order contract of one forward.

    Per slot, its entries are one contiguous run of consecutive chunks and only the run's last entry may be
    partial. Empty entries come after every non-empty one, and at least one entry is non-empty. Runs of
    different slots may come in any order.
    """
    if len(entries) != forward_chunks:
        raise ValueError(f"a forward holds {forward_chunks} entries, got {len(entries)}")
    if entries[0].is_empty:
        raise ValueError("a forward needs at least one non-empty entry (empties go at the tail)")
    prev = None
    seen = set()
    for e in entries:
        if e.is_empty:
            prev = e
            continue
        if prev is not None and prev.is_empty:
            raise ValueError(f"{e}: non-empty entry after an empty one")
        if not 0 <= e.slot < num_users:
            raise ValueError(f"{e}: slot out of range [0, {num_users})")
        if e.start < 0:
            raise ValueError(f"{e}: negative start")
        if e.start % ENTRY_START_ALIGN:
            raise ValueError(f"{e}: start must be a multiple of {ENTRY_START_ALIGN}")
        if not e.start < e.end <= e.start + chunk_size:
            raise ValueError(f"{e}: more than one {chunk_size}-token chunk")
        if e.start + chunk_size > max_seq_len:
            raise ValueError(f"{e}: chunk exceeds the per-user cache {max_seq_len}")
        if e.slot in seen:
            if prev.slot != e.slot:
                raise ValueError(f"{e}: slot {e.slot} appears in two separate runs")
            if prev.end != prev.start + chunk_size:
                raise ValueError(f"{prev}: a partial entry must end its slot's run")
            if e.start != prev.end:
                raise ValueError(f"{e}: does not continue {prev}")
        seen.add(e.slot)
        prev = e


def packed_forward_refusals(
    forward_chunks: int,
    *,
    supports_packed_forward: bool,
    mtp_levels: int = 0,
    dflash: bool = False,
    use_trace: bool = False,
    d2h_acks: bool = False,
) -> list:
    """Why a configuration cannot run ``forward_chunks`` entries per forward (empty list: it can)."""
    if forward_chunks < 1:
        return [f"PREFILL_FORWARD_CHUNKS must be >= 1, got {forward_chunks}"]
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
        reasons.append("PREFILL_USE_TRACE=1 (per-entry arguments are not trace inputs)")
    if d2h_acks:
        reasons.append("PREFILL_LAYER_ACK_D2H=1")
    return reasons


def check_packed_forward_support(forward_chunks: int, **kwargs) -> None:
    reasons = packed_forward_refusals(forward_chunks, **kwargs)
    if reasons:
        raise ValueError(f"PREFILL_FORWARD_CHUNKS={forward_chunks} is not supported here: " + "; ".join(reasons))
