# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sequence packing: several whole texts in one row of the backbone, so a padded batch runs in fewer passes.

A padded batch computes every text at its chunk's longest length. Sorted and chunked (tt.chunking), 11 real
documents of 3,760 tokens still fill 6 x 512 + 5 x 287 slots, two passes (8.35 ms against 5.38 for one pass of
4,096 full-length tokens). Packed, the texts sit side by side in rows of `width` slots and the batch takes
ceil(rows / (max_tokens / width)) passes. Three properties keep each text's result its own:

- attention: tt.common.segment_attention_mask lets a query attend only the keys of its own text;
- positions: the model's rotary embedding makes q . k depend only on the distance between two tokens, so a text
  starting at offset o of a row scores exactly as it would at 0 and positions need not restart;
- everything else (norms, matmuls, the MoE layer) works token by token, and pooling sums each text's own slots.

A text is never split across rows. Rows are filled first-fit, longest text first.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

# Row widths tried, in whole tiles. 512 is the longest text (the tokenizer's limit); wider rows hold more texts of
# a few hundred tokens each but grow every row's attention with width squared.
WIDTHS = (512, 1024)


@dataclass(frozen=True)
class Segment:
    """Text `index` of the batch at slots offset .. offset + length of row `row` of its pass."""

    index: int
    row: int
    offset: int
    length: int


@dataclass(frozen=True)
class PackedPass:
    rows: int
    width: int
    segments: tuple[Segment, ...]

    @property
    def tokens(self) -> int:
        return self.rows * self.width


def _first_fit(lengths: Sequence[int], width: int) -> list[list[tuple[int, int]]]:
    """Rows of (index, length), longest text first, each into the first row with room."""
    rows: list[list[tuple[int, int]]] = []
    loads: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        length = max(1, int(lengths[index]))
        if length > width:
            raise ValueError(f"text {index} has {length} tokens, more than a row of {width}")
        for row, load in enumerate(loads):
            if load + length <= width:
                rows[row].append((index, length))
                loads[row] += length
                break
        else:
            rows.append([(index, length)])
            loads.append(length)
    return rows


def pack(lengths: Sequence[int], width: int, max_tokens: int) -> list[PackedPass]:
    """Pack texts of `lengths` real tokens into rows of `width` slots and the rows into passes of at most
    max_tokens slots (at least one row a pass)."""
    rows = _first_fit(lengths, width)
    per_pass = max(1, max_tokens // width)
    passes = []
    for first in range(0, len(rows), per_pass):
        segments = []
        for row, members in enumerate(rows[first : first + per_pass]):
            offset = 0
            for index, length in members:
                segments.append(Segment(index, row, offset, length))
                offset += length
        passes.append(PackedPass(min(per_pass, len(rows) - first), width, tuple(segments)))
    return passes


def best_packing(lengths: Sequence[int], max_tokens: int, widths: Sequence[int] = WIDTHS) -> list[PackedPass]:
    """The packing with the fewest passes over the widths that hold the longest text; among equals the one with
    the least attention work (rows x width^2 summed over passes), then the narrowest rows."""
    longest = max(int(length) for length in lengths)
    options = [pack(lengths, width, max_tokens) for width in widths if width >= longest]
    return min(options, key=lambda p: (len(p), sum(x.rows * x.width**2 for x in p), p[0].width))


# Device time model of a pass on one Blackhole p300c (nomic-v14, real text, tracy device time): a pass costs
# PASS_US plus TOKEN_US a computed token (fit on full-length rows: 2x512 2.30, 4x512 3.26, 8x512 5.35 ms), and a
# packed pass adds its attention on stock SDPA with the block-diagonal mask over the nomic_sdpa a chunked pass runs
# (12 calls a pass: 106 against 53 us a call at 8 x 512 rows, 180 at 4 x 1024). Measured against it (ms): chunked
# 7.86 / 13.25 / 7.12 / 24.02 vs model 7.69 / 12.65 / 6.94 / 24.1; packed 6.90 / 12.06 / 6.88 / 27.57 vs 6.85 /
# 11.96 / 6.85 / 27.4. Saving a pass is worth about PASS_US, so a packing of 1024-wide rows that saves one pass in
# five loses (51 documents: 24.0 chunked, 27.6 packed).
PASS_US = 1170.0
TOKEN_US = 1.02
PACKED_ATTENTION_US = {512: 630.0, 1024: 1500.0}


def chunked_cost_us(chunks) -> float:
    """Modelled device time of length-sorted chunks (tt.chunking.Chunk, .tokens tile-padded)."""
    return sum(PASS_US + TOKEN_US * chunk.tokens for chunk in chunks)


def packed_cost_us(passes: Sequence[PackedPass]) -> float:
    """Modelled device time of packed passes."""
    return sum(PASS_US + TOKEN_US * p.tokens + PACKED_ATTENTION_US[p.width] for p in passes)


def cheaper_packing(lengths: Sequence[int], max_tokens: int, chunks) -> list[PackedPass] | None:
    """The packing with the lowest modelled time over the widths that hold the longest text, if it beats the
    length-sorted chunks' modelled time; else None (run the chunks)."""
    longest = max(int(length) for length in lengths)
    options = [pack(lengths, width, max_tokens) for width in WIDTHS if width >= longest]
    best = min(options, key=packed_cost_us)
    return best if packed_cost_us(best) < chunked_cost_us(chunks) else None


def packed_inputs(input_ids: torch.Tensor, packed: PackedPass, pad_token_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_ids, segment_ids), both (rows, width): each text's tokens at its slots, pad_token_id elsewhere;
    segment_ids holds the text's index in the batch at its slots and -1 at slots no text fills."""
    ids = torch.full((packed.rows, packed.width), pad_token_id, dtype=input_ids.dtype)
    segments = torch.full((packed.rows, packed.width), -1, dtype=torch.long)
    for seg in packed.segments:
        ids[seg.row, seg.offset : seg.offset + seg.length] = input_ids[seg.index, : seg.length]
        segments[seg.row, seg.offset : seg.offset + seg.length] = seg.index
    return ids, segments


def pooling_rows(packed: PackedPass) -> torch.Tensor:
    """(texts, rows * width) 0/1: row t marks the slots of the pass's t-th segment on the flattened token axis, the
    packed_sum_pool weights of a packed pass (before tile padding of the text axis)."""
    weights = torch.zeros(len(packed.segments), packed.rows * packed.width)
    for t, seg in enumerate(packed.segments):
        start = seg.row * packed.width + seg.offset
        weights[t, start : start + seg.length] = 1
    return weights
