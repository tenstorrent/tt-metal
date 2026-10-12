# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunk entries of a packed prefill forward.

A forward of ``forward_tokens`` rows holds ``forward_tokens // chunk_size`` entries in pack order. Each SP chip
holds every entry's ``chunk_size // sp`` rows back to back, in the entry's rotated per-chip layout.
"""

from dataclasses import dataclass
from functools import lru_cache

from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens, rotated_chunk_real_counts


@dataclass(frozen=True)
class PrefillEntry:
    slot: int  # KV-cache user slot
    start: int  # absolute KV position of the chunk's first row (the prefix already cached)
    end: int  # past the last real token; start == end is an empty entry (no attention, no KV write)

    @property
    def is_empty(self) -> bool:
        return self.end == self.start


def validate_entries(entries, *, chunk_size: int, forward_chunks: int, num_users: int, max_seq_len: int) -> None:
    """A slot's entries are one contiguous run of consecutive chunks; only the run's last entry may be partial;
    empty entries come after every non-empty one."""
    assert len(entries) == forward_chunks, f"a forward holds {forward_chunks} entries, got {len(entries)}"
    assert not entries[0].is_empty, "a forward needs at least one non-empty entry (empties go at the tail)"
    prev = None
    seen = set()
    for e in entries:
        if e.is_empty:
            prev = e
            continue
        assert prev is None or not prev.is_empty, f"{e}: non-empty entry after an empty one"
        assert 0 <= e.slot < num_users, f"{e}: slot out of range [0, {num_users})"
        assert e.start % 32 == 0, f"{e}: start must be a multiple of 32 (the KV-cache writer's tile grid)"
        assert e.start < e.end <= e.start + chunk_size, f"{e}: more than one {chunk_size}-token chunk"
        assert e.start + chunk_size <= max_seq_len, f"{e}: chunk exceeds the per-user cache {max_seq_len}"
        if e.slot in seen:
            assert prev.slot == e.slot, f"{e}: slot {e.slot} appears in two separate runs"
            assert prev.end == prev.start + chunk_size, f"{prev}: a partial entry must end its slot's run"
            assert e.start == prev.end, f"{e}: does not continue {prev}"
        seen.add(e.slot)
        prev = e


def packed_real_rows(entries, sp: int, chunk_local: int) -> tuple:
    """Per SP chip, the count of leading real rows of the packed forward, for the MoE padding config. A chip
    whose pad rows precede a later real row counts all its rows as real."""
    return _packed_real_rows(tuple(entries), sp, chunk_local)


@lru_cache(maxsize=1024)  # every MoE layer asks for the same forward
def _packed_real_rows(entries: tuple, sp: int, chunk_local: int) -> tuple:
    counts = [_entry_real_counts(e.start % (sp * chunk_local), e.end - e.start, sp, chunk_local) for e in entries]
    rows = []
    for c in range(sp):
        lead = 0
        for k, count in enumerate(counts):
            lead += count[c]
            if count[c] < chunk_local:
                if any(later[c] for later in counts[k + 1 :]):
                    lead = len(entries) * chunk_local
                break
        rows.append(lead)
    return tuple(rows)


def _entry_real_counts(slab_offset: int, n_real: int, sp: int, chunk_local: int) -> list:
    if n_real == sp * chunk_local:
        return [chunk_local] * sp
    return rotated_chunk_real_counts(slab_offset, n_real, sp, chunk_local)


def pack_forward_tokens(chunk_tokens: list, starts: list, sp: int) -> list:
    """Chip-major token order of a packed forward: chip c's rows are slice c of every entry's rotated chunk
    (rotate_chunk_tokens at the entry's start), in entry order. ``chunk_tokens`` are natural-order chunks."""
    per_chip = [[] for _ in range(sp)]
    for tokens, start in zip(chunk_tokens, starts, strict=True):
        rotated = rotate_chunk_tokens(list(tokens), start, sp)
        s = len(rotated) // sp
        for c in range(sp):
            per_chip[c] += rotated[c * s : (c + 1) * s]
    return [t for chip in per_chip for t in chip]
