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

A text is never split across rows and starts on a tile (32 slots), so each 32-query tile of a row belongs to one
text: nomic_sdpa's segmented mode then runs each text's attention on its own keys at the speed of an unpacked pass
(55.8 us a call for 4 x 1024 packed rows against 52.4 for 8 x 512 full rows, and 242 for stock SDPA with the
block-diagonal mask). Rows are filled first-fit, longest text first.
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


TILE = 32


def _span(length: int) -> int:
    """Slots a text takes in a row: its length rounded up to whole tiles."""
    return -(-max(1, int(length)) // TILE) * TILE


def _first_fit(lengths: Sequence[int], width: int) -> list[list[tuple[int, int]]]:
    """Rows of (index, length), longest text first, each into the first row with room for its whole tiles."""
    rows: list[list[tuple[int, int]]] = []
    loads: list[int] = []
    for index in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        length = max(1, int(lengths[index]))
        if length > width:
            raise ValueError(f"text {index} has {length} tokens, more than a row of {width}")
        for row, load in enumerate(loads):
            if load + _span(length) <= width:
                rows[row].append((index, length))
                loads[row] += _span(length)
                break
        else:
            rows.append([(index, length)])
            loads.append(_span(length))
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
                offset += _span(length)
        passes.append(PackedPass(min(per_pass, len(rows) - first), width, tuple(segments)))
    return passes


def best_packing(lengths: Sequence[int], max_tokens: int, widths: Sequence[int] = WIDTHS) -> list[PackedPass]:
    """The packing with the fewest passes over the widths that hold the longest text; among equals the one with
    the least attention work (rows x width^2 summed over passes), then the narrowest rows."""
    longest = max(int(length) for length in lengths)
    options = [pack(lengths, width, max_tokens) for width in widths if width >= longest]
    return min(options, key=lambda p: (len(p), sum(x.rows * x.width**2 for x in p), p[0].width))


# Device time model of a pass on one Blackhole p300c (real text, tracy device time): a pass costs PASS_US plus
# TOKEN_US a computed token (fit on full-length rows: 2x512 2.30, 4x512 3.26, 8x512 5.35 ms; within 0.2 ms of eight
# measured chunked and packed batches), and a packed pass adds what its attention costs over the nomic_sdpa of an
# unpacked pass. On stock SDPA with the block-diagonal mask that was 630 / 1500 us a pass (rows of 512 / 1024),
# which made packing lose wherever it saved only one pass in several; nomic_sdpa's segmented mode costs 12 calls x
# 4.6 / 3.4 us (57.0 and 55.8 against 52.4 us a call).
PASS_US = 1170.0
TOKEN_US = 1.02
PACKED_ATTENTION_US = {512: 55.0, 1024: 41.0}
# packed passes on stock SDPA with the block-diagonal mask (tt_config.generated_sdpa_segments off)
STOCK_PACKED_ATTENTION_US = {512: 630.0, 1024: 1500.0}


def chunked_cost_us(chunks) -> float:
    """Modelled device time of length-sorted chunks (tt.chunking.Chunk, .tokens tile-padded)."""
    return sum(PASS_US + TOKEN_US * chunk.tokens for chunk in chunks)


def packed_cost_us(passes: Sequence[PackedPass], attention_us=PACKED_ATTENTION_US) -> float:
    """Modelled device time of packed passes (attention_us: a packed pass's attention cost by row width)."""
    return sum(PASS_US + TOKEN_US * p.tokens + attention_us[p.width] for p in passes)


def cheaper_packing(
    lengths: Sequence[int], max_tokens: int, chunks, attention_us=PACKED_ATTENTION_US
) -> list[PackedPass] | None:
    """The packing with the lowest modelled time over the widths that hold the longest text, if it beats the
    length-sorted chunks' modelled time; else None (run the chunks)."""
    longest = max(int(length) for length in lengths)
    options = [pack(lengths, width, max_tokens) for width in WIDTHS if width >= longest]
    best = min(options, key=lambda p: packed_cost_us(p, attention_us))
    return best if packed_cost_us(best, attention_us) < chunked_cost_us(chunks) else None


def packed_inputs(
    input_ids: torch.Tensor, packed: PackedPass, pad_token_id: int, fill: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """(input_ids, segment_ids), both (rows, width): each text's tokens at its slots; segment_ids holds the text's
    index in the batch at its slots and -1 at slots no text fills. Those hold pad_token_id, or with fill the row's
    own tokens repeated (their results are never used; one shared pad id reads one embedding row and routes every
    such slot to the same experts, tt.model._fill_padding)."""
    ids = torch.full((packed.rows, packed.width), pad_token_id, dtype=input_ids.dtype)
    segments = torch.full((packed.rows, packed.width), -1, dtype=torch.long)
    for seg in packed.segments:
        ids[seg.row, seg.offset : seg.offset + seg.length] = input_ids[seg.index, : seg.length]
        segments[seg.row, seg.offset : seg.offset + seg.length] = seg.index
    if fill:
        for row in range(packed.rows):
            used = segments[row] >= 0
            free = int((~used).sum())
            if free and bool(used.any()):
                real = ids[row, used]
                ids[row, ~used] = real.repeat(-(-free // real.numel()))[:free]
    return ids, segments


def row_segments(segment_ids: torch.Tensor) -> list[list[tuple[int, int]]]:
    """One list a row of (start, length): the runs of one text in packed segment_ids (-1 for slots no text fills),
    nomic_sdpa's segments argument."""
    out = []
    for row in segment_ids.tolist():
        runs, start = [], None
        for at, owner in enumerate(row + [-2]):
            if start is not None and owner != row[start]:
                runs.append((start, at - start))
                start = None
            if start is None and owner >= 0 and at < len(row):
                start = at
        out.append(runs)
    return out


def pooling_rows(packed: PackedPass) -> torch.Tensor:
    """(texts, rows * width) 0/1: row t marks the slots of the pass's t-th segment on the flattened token axis, the
    packed_sum_pool weights of a packed pass (before tile padding of the text axis)."""
    weights = torch.zeros(len(packed.segments), packed.rows * packed.width)
    for t, seg in enumerate(packed.segments):
        start = seg.row * packed.width + seg.offset
        weights[t, start : start + seg.length] = 1
    return weights


def planned_passes(lengths: Sequence[int], max_tokens: int, min_tokens: int = 0, attention_us=PACKED_ATTENTION_US):
    """(passes, computed tokens) of the plan embed() runs for texts of these real lengths: the length-sorted chunks,
    or the packing where tt.packing's model puts it below them."""
    from models.experimental.nomic_embed_text_v2_moe.tt.chunking import plan_sorted_chunks

    chunks = plan_sorted_chunks(sorted((max(1, int(n)) for n in lengths), reverse=True), max_tokens, min_tokens)
    packed = cheaper_packing(lengths, max_tokens, chunks, attention_us) if len(chunks) > 1 else None
    if packed is not None:
        return len(packed), sum(p.tokens for p in packed)
    return len(chunks), sum(chunk.tokens for chunk in chunks)


def texts_for_full_passes(
    lengths: Sequence[int], max_tokens: int, min_tokens: int = 0, min_fill: float = 0.9, flush: bool = False
) -> int:
    """For a serving loop: how many of the queued texts (real lengths, in arrival order) to embed now so that every
    pass embed() runs is at least min_fill full (computed tokens over passes x max_tokens). A pass costs ~1.2 ms
    whatever it holds, so a batch that spills a few texts into an almost empty last pass pays most of a pass for
    them (4,721 real tokens: 7.64 ms in two passes, 4,096: 5.38 ms in one). Returns the longest such prefix; with
    none, 0 (wait for more texts) unless flush (embed them all now: the latency budget is spent)."""
    best = 0
    for count in range(1, len(lengths) + 1):
        passes, computed = planned_passes(lengths[:count], max_tokens, min_tokens)
        if computed >= min_fill * passes * max_tokens:
            best = count
    if best == 0 and flush:
        return len(lengths)
    return best
