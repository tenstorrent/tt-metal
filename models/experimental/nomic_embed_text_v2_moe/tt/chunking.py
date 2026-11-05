# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""How a large batch is split into chunks that each run the whole backbone on their own.

A chunk is a run of whole sequences. Attention never crosses a sequence and the MoE layer routes every
token on its own, with no capacity limit, so a chunk's outputs are the ones it would get in any other
batch: the split changes which programs run, not what they compute.

An input of at most max_tokens computed tokens (tt_config.max_chunk_tokens, 4096) runs as one pass. A larger
one runs as chunks of max_tokens while the sequences left fill one, and the rest as one more chunk; a rest
under min_tokens (tt_config.min_chunk_tokens, 1024) is merged with the chunk before it and the two are split
evenly. 4096 tokens is the pass the generated ops are tuned for and the cheapest per token (1.40 us at 8x512
against 1.70 for one 8192-token pass, whose activations leave L1 and which runs the stock norms and MoE glue).
Every chunk costs a fixed ~1.3 ms on top of its tokens, so the rest stays one chunk rather than being split
further, and a chunk under ~1000 tokens runs the stock glue and costs about what 1024 tokens do, hence the
merge: 15x512 runs as 8 + 7 sequences, 9x512 as 5 + 4 (nomic-opt c06 chunking study, 2026-11-05: 10.7% less
device time than 4096 -> 2048 -> 1024 chunks over 14 large shapes).

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


def _fill(lengths: Sequence[int], max_tokens: int, min_tokens: int) -> list[Chunk]:
    """Chunks of consecutive sequences, each run at its first sequence's length (lengths is non-increasing,
    so a chunk's first sequence is its longest).

    Each chunk takes as many sequences as fit max_tokens at its first sequence's length (a sequence longer than
    that is a chunk of its own), and the last takes the rest. A last chunk under min_tokens is merged with the
    one before it and the two are split evenly by count, the first half never larger than that chunk was.
    """
    if max_tokens < 1:
        raise ValueError(f"max_tokens must be a positive token count, got {max_tokens}")
    chunks, start = [], 0
    while start < len(lengths):
        count = min(max(1, max_tokens // _padded(lengths[start])), len(lengths) - start)
        chunks.append(Chunk(start, count, lengths[start]))
        start += count
    if len(chunks) > 1 and chunks[-1].tokens < min_tokens:
        before, last = chunks[-2], chunks[-1]
        both = before.count + last.count
        first = min(before.count, (both + 1) // 2)
        if first < before.count:
            chunks[-2:] = [
                Chunk(before.start, first, before.seqlen),
                Chunk(before.start + first, both - first, lengths[before.start + first]),
            ]
    return chunks


@cache
def plan_chunks(batch: int, seqlen: int, max_tokens: int, min_tokens: int = 0) -> tuple[Chunk, ...]:
    """Split B sequences of S tokens into contiguous chunks of at most max_tokens computed tokens; see the module
    docstring. min_tokens 0 keeps the rest as it is (every chunk full but the last)."""
    return tuple(_fill([seqlen] * batch, max_tokens, min_tokens))


def plan_sorted_chunks(lengths: Sequence[int], max_tokens: int, min_tokens: int = 0) -> list[Chunk]:
    """Chunks of length-sorted sequences, longest first, each run at its own longest sequence's length.

    Sorted, a chunk's sequences are of similar length, so cutting the chunk to its longest one removes
    most of the padding a batch padded to its overall longest would compute.

    Args:
        lengths: real lengths, non-increasing.
        max_tokens: the most computed tokens a chunk may take.
        min_tokens: a last chunk under this many computed tokens is merged with the one before and the two split.
    """
    if any(earlier < later for earlier, later in zip(lengths, lengths[1:])):
        raise ValueError("plan_sorted_chunks takes lengths sorted longest first")
    return _fill(list(lengths), max_tokens, min_tokens)
