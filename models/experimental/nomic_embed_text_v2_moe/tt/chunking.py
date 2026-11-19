# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How a large batch is split into chunks that each run the whole backbone on their own.

A chunk is a run of whole sequences. Attention never crosses a sequence and the MoE layer routes every
token on its own, with no capacity limit, so a chunk's outputs are the ones it would get in any other
batch: the split changes which programs run, not what they compute.

Each chunk takes as many sequences as fit max_tokens and the last takes the rest, as DeepSeek V3's
prefill chunks at a fixed size.

Computed tokens count each sequence at its tile-padded length, B x ceil32(S), as every program does.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Sequence

import ttnn


def _padded(seqlen: int) -> int:
    """A sequence's tile-padded length; an empty one still takes a tile row."""
    return max(1, ttnn.core.divup(seqlen, ttnn.TILE_SIZE)) * ttnn.TILE_SIZE


@dataclass(frozen=True)
class Chunk:
    """Sequences start .. start + count of the batch (of the length-sorted order, for plan_sorted_chunks),
    run at seqlen tokens a sequence."""

    start: int
    count: int
    seqlen: int

    @property
    def tokens(self) -> int:
        return self.count * _padded(self.seqlen)


def _fill(lengths: Sequence[int], max_tokens: int) -> list[Chunk]:
    """Chunks of consecutive sequences, each run at its first sequence's length and holding as many
    sequences as fit max_tokens at that length (a sequence longer than that is a chunk of its own).

    lengths is non-increasing, so a chunk's first sequence is its longest.
    """
    chunks, start = [], 0
    while start < len(lengths):
        count = min(max(1, max_tokens // _padded(lengths[start])), len(lengths) - start)
        chunks.append(Chunk(start, count, lengths[start]))
        start += count
    return chunks


@cache
def plan_chunks(batch: int, seqlen: int, max_tokens: int) -> tuple[Chunk, ...]:
    """Split B sequences of S tokens into contiguous chunks of at most max_tokens computed tokens, all
    full but the last."""
    return tuple(_fill([seqlen] * batch, max_tokens))


def plan_sorted_chunks(lengths: Sequence[int], max_tokens: int) -> list[Chunk]:
    """Chunks of length-sorted sequences, longest first, each run at its own longest sequence's length.

    Sorted, a chunk's sequences are of similar length, so cutting the chunk to its longest one removes
    most of the padding a batch padded to its overall longest would compute.

    Args:
        lengths: real lengths, non-increasing.
        max_tokens: the most computed tokens a chunk may take.
    """
    if any(earlier < later for earlier, later in zip(lengths, lengths[1:])):
        raise ValueError("plan_sorted_chunks takes lengths sorted longest first")
    return _fill(list(lengths), max_tokens)
