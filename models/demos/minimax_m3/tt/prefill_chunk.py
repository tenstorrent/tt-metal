# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunks of a packed prefill forward.

A forward of ``forward_tokens_budget`` rows holds ``forward_tokens_budget // chunk_size`` chunks in pack order.
Each SP chip holds every chunk's ``chunk_size // sp`` rows back to back, in the chunk's rotated per-chip layout.
"""

from dataclasses import dataclass
from functools import lru_cache

from models.demos.common.prefill.chunk_layout import rotate_chunk_tokens, rotated_chunk_real_counts


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


def validate_chunks(chunks, *, chunk_size: int, forward_chunks: int, num_users: int, max_seq_len: int) -> None:
    """A slot's chunks are one contiguous run of consecutive chunks; only the run's last chunk may be partial;
    empty chunks come after every non-empty one."""
    assert len(chunks) == forward_chunks, f"a forward holds {forward_chunks} chunks, got {len(chunks)}"
    assert not chunks[0].is_empty, "a forward needs at least one non-empty chunk (empties go at the tail)"
    prev = None
    seen = set()
    for c in chunks:
        if c.is_empty:
            prev = c
            continue
        assert prev is None or not prev.is_empty, f"{c}: non-empty chunk after an empty one"
        assert 0 <= c.slot_id < num_users, f"{c}: slot out of range [0, {num_users})"
        assert c.actual_start >= 0, f"{c}: negative start"
        assert c.actual_start % 32 == 0, f"{c}: start must be a multiple of 32 (the KV-cache writer's tile grid)"
        assert (
            c.actual_start < c.actual_end <= c.actual_start + chunk_size
        ), f"{c}: more than one {chunk_size}-token chunk"
        assert c.actual_start + chunk_size <= max_seq_len, f"{c}: chunk exceeds the per-user cache {max_seq_len}"
        if c.slot_id in seen:
            assert prev.slot_id == c.slot_id, f"{c}: slot {c.slot_id} appears in two separate runs"
            assert prev.actual_end == prev.actual_start + chunk_size, f"{prev}: a partial chunk must end its slot's run"
            assert c.actual_start == prev.actual_end, f"{c}: does not continue {prev}"
        seen.add(c.slot_id)
        prev = c


def packed_real_rows(chunks, sp: int, chunk_local: int) -> tuple:
    """Per SP chip, the count of leading real rows of the packed forward, for the MoE padding config. A chip
    whose pad rows precede a later real row counts all its rows as real."""
    # TODO: per-chunk real-row ranges in the MoE padding config, so a partial chunk followed by a non-empty one
    # does not route its pad rows.
    return _packed_real_rows(tuple(chunks), sp, chunk_local)


@lru_cache(maxsize=1024)  # every MoE layer asks for the same forward
def _packed_real_rows(chunks: tuple, sp: int, chunk_local: int) -> tuple:
    counts = [
        _chunk_real_counts(c.actual_start % (sp * chunk_local), c.actual_end - c.actual_start, sp, chunk_local)
        for c in chunks
    ]
    rows = []
    for chip in range(sp):
        lead = 0
        for k, count in enumerate(counts):
            lead += count[chip]
            if count[chip] < chunk_local:
                if any(later[chip] for later in counts[k + 1 :]):
                    lead = len(chunks) * chunk_local
                break
        rows.append(lead)
    return tuple(rows)


def _chunk_real_counts(slab_offset: int, n_real: int, sp: int, chunk_local: int) -> list:
    if n_real == sp * chunk_local:
        return [chunk_local] * sp
    return rotated_chunk_real_counts(slab_offset, n_real, sp, chunk_local)


def pack_forward_tokens(chunk_tokens: list, starts: list, sp: int) -> list:
    """Chip-major token order of a packed forward: chip c's rows are slice c of every chunk's rotated tokens
    (rotate_chunk_tokens at the chunk's actual_start), in chunk order. ``chunk_tokens`` are natural-order chunks."""
    per_chip = [[] for _ in range(sp)]
    for tokens, start in zip(chunk_tokens, starts, strict=True):
        rotated = rotate_chunk_tokens(list(tokens), start, sp)
        s = len(rotated) // sp
        for c in range(sp):
            per_chip[c] += rotated[c * s : (c + 1) * s]
    return [t for chip in per_chip for t in chip]
