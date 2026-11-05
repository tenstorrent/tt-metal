# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How a large batch is split into chunks that each run the whole backbone on their own.

A chunk is a run of whole sequences. Attention never crosses a sequence and the MoE layer routes every
token on its own, with no capacity limit, so a chunk's outputs are the ones it would get in any other
batch: the split changes which programs run, not what they compute.

The chunk sizes are tt_config.chunk_tokens, largest first: 4096, 2048 and 1024 computed tokens. An input
that fits the largest runs as one pass. A larger one runs as chunks of the largest size while the
sequences left fill one, then of the next size down while they fill one, and so on; the last chunk takes
what is left once they fill none. 4096 tokens is the pass the generated ops are tuned for and the cheapest
per token (1.40 us at 8x512 against 1.70 for one 8192-token pass, whose activations leave L1 and which
runs the stock norms and MoE glue), and 2048 and 1024 are the other two token counts they are tuned for:
an 8192-token input runs as 4096 + 4096, a 7168-token one as 4096 + 2048 + 1024.

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


def _sizes(chunk_tokens: Sequence[int]) -> tuple[int, ...]:
    sizes = tuple(sorted({int(size) for size in chunk_tokens}, reverse=True))
    if not sizes or sizes[-1] < 1:
        raise ValueError(f"chunk sizes must be positive token counts, got {tuple(chunk_tokens)}")
    return sizes


def _fill(lengths: Sequence[int], chunk_tokens: Sequence[int]) -> list[Chunk]:
    """Chunks of consecutive sequences, each run at its first sequence's length (lengths is non-increasing,
    so a chunk's first sequence is its longest).

    All of them in one chunk when they fit the largest size at the longest length. Otherwise each chunk
    holds as many sequences as fit the largest size that the sequences left fill at the current length, and
    the last chunk takes the rest once they fill none. A sequence longer than the largest size is a chunk of
    its own.
    """
    sizes = _sizes(chunk_tokens)
    if not lengths:
        return []
    if len(lengths) * _padded(lengths[0]) <= sizes[0]:
        return [Chunk(0, len(lengths), lengths[0])]
    chunks, start = [], 0
    while start < len(lengths):
        padded = _padded(lengths[start])
        left = len(lengths) - start
        fits = [size // padded for size in sizes]  # sequences a chunk of each size holds at this length
        filled = [held for held in fits if 0 < held <= left]
        if filled:
            count = filled[0]  # the largest size the sequences left fill
        elif fits[0] == 0:
            count = 1  # longer than the largest size: a chunk of its own
        else:
            count = left  # the rest, which fills no size
        chunks.append(Chunk(start, count, lengths[start]))
        start += count
    return chunks


@cache
def plan_chunks(batch: int, seqlen: int, chunk_tokens: tuple[int, ...]) -> tuple[Chunk, ...]:
    """Split B sequences of S tokens into contiguous chunks of the chunk_tokens sizes (computed tokens,
    largest first); see the module docstring."""
    return tuple(_fill([seqlen] * batch, chunk_tokens))


def plan_sorted_chunks(lengths: Sequence[int], chunk_tokens: Sequence[int]) -> list[Chunk]:
    """Chunks of length-sorted sequences, longest first, each run at its own longest sequence's length.

    Sorted, a chunk's sequences are of similar length, so cutting the chunk to its longest one removes
    most of the padding a batch padded to its overall longest would compute.

    Args:
        lengths: real lengths, non-increasing.
        chunk_tokens: the chunk sizes in computed tokens; the largest is the most a chunk may take.
    """
    if any(earlier < later for earlier, later in zip(lengths, lengths[1:])):
        raise ValueError("plan_sorted_chunks takes lengths sorted longest first")
    return _fill(list(lengths), chunk_tokens)
