# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Chunk entries of a prefill forward."""

from dataclasses import dataclass

from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens


@dataclass(frozen=True)
class PrefillEntry:
    slot: int
    start: int  # absolute KV position of the chunk's first row
    end: int  # past the last real token; start == end is an empty entry

    @property
    def is_empty(self) -> bool:
        return self.end == self.start


EMPTY_ENTRY = PrefillEntry(0, 0, 0)


def pad_entries(entries, forward_chunks: int) -> list:
    """Fills a partial forward with empty entries."""
    assert len(entries) <= forward_chunks, f"{len(entries)} entries for a forward of {forward_chunks}"
    return list(entries) + [EMPTY_ENTRY] * (forward_chunks - len(entries))


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
        assert e.start >= 0, f"{e}: negative start"
        assert e.start % 32 == 0, f"{e}: start must be a multiple of 32 (the KV-cache writer's tile grid)"
        assert e.start < e.end <= e.start + chunk_size, f"{e}: more than one {chunk_size}-token chunk"
        assert e.start + chunk_size <= max_seq_len, f"{e}: chunk exceeds the per-user cache {max_seq_len}"
        if e.slot in seen:
            assert prev.slot == e.slot, f"{e}: slot {e.slot} appears in two separate runs"
            assert prev.end == prev.start + chunk_size, f"{prev}: a partial entry must end its slot's run"
            assert e.start == prev.end, f"{e}: does not continue {prev}"
        seen.add(e.slot)
        prev = e


def pack_forward_tokens(chunk_tokens: list, starts: list, sp: int) -> list:
    """Chip-major token order of a forward: chip c holds slice c of every entry's rotated chunk, in entry order."""
    per_chip = [[] for _ in range(sp)]
    for tokens, start in zip(chunk_tokens, starts, strict=True):
        rotated = rotate_chunk_tokens(list(tokens), start, sp)
        s = len(rotated) // sp
        for c in range(sp):
            per_chip[c] += rotated[c * s : (c + 1) * s]
    return [t for chip in per_chip for t in chip]
