# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional vision attention through chunked and cached-prefix prefill.

Gemma 4 attends bidirectionally inside every contiguous run of image or video placeholders
(``use_bidirectional_attention="vision"``, HF's ``blockwise_overlay``), on top of the causal (full
layers) or causal-and-windowed (sliding layers) text mask. The whole-prompt single-chunk prefill
takes that overlay as an explicit SDPA mask (``build_mm_prefill_masks``); this module carries the
vision spans at their absolute prompt positions through every other prefill form so a continuation
after a prefix-cache hit, a later generator chunk and a long single call keep the same attention:

* ``vision_spans_of_prompt`` turns a prompt (or the uncached suffix of one, with its ``base``) into
  absolute ``(start, end)`` spans;
* ``plan_chunk_starts_outside_vision_spans`` keeps the generator's chunk boundaries out of a span
  (a span cut by a chunk end would be prefilled in two halves, the first without its future rows);
* ``MMPrefillVision`` is the per-call object the model hands every layer: ``mask(abs_start, length,
  window)`` builds the additive mask of any in-memory square (the chunk, the ``[tail | chunk]`` slice
  of a sliding layer, a window slice of a long prefill) and ``vision_row_tiles`` lists the chunk's
  row tiles that hold vision rows with the position each row must attend up to, for the paged
  full-attention path, whose keys live in the KV cache: those rows are recomputed with the paged
  decode SDPA, each row attending ``[0, cur_pos]`` with ``cur_pos`` = its span's last position.
"""

from __future__ import annotations

import torch
from loguru import logger

import ttnn
from models.demos.gemma4.tt.attention.prefill import (
    MASK_NEG,
    vision_block_ids,
    vision_block_spans,
    vision_placeholder_ids,
)

TILE_ROWS = 32


def vision_spans_of_prompt(tokens, image_token_id, video_token_id=None, *, base: int = 0):
    """Absolute ``(start, end)`` spans of the vision placeholder runs in ``tokens``.

    ``tokens`` is the prompt or, with ``base`` > 0, the uncached suffix the generator prefills
    (``base`` = the cached prefix it was cut at); padding ids are text and open no span.
    """
    ids = vision_placeholder_ids(image_token_id, video_token_id)
    blocks = vision_block_ids(torch.as_tensor(tokens).reshape(-1), ids)
    return tuple((int(base) + s, int(base) + e) for s, e in vision_block_spans(blocks))


def spans_in_window(spans, start: int, end: int):
    """The spans that intersect ``[start, end)``."""
    return tuple((s, e) for s, e in spans if s < end and e > start)


def plan_chunk_starts_outside_vision_spans(num_cached, seq_len, chunk_size, last_abs, spans, *, align=128):
    """Generator chunk starts for the suffix ``[num_cached, num_cached + seq_len)``.

    The grid is ``num_cached + k * chunk_size`` capped by the planned last-chunk start ``last_abs``
    (``_adjust_last_prefill_chunk``), as before. A grid boundary that falls strictly inside a vision
    span moves down to the span's start aligned to ``align``, so the whole span lands in the next
    chunk (which stays ``<= chunk_size`` long because it starts earlier, not later). A boundary that
    cannot move (the span starts in the chunk's first ``align`` rows, or the last chunk could not
    absorb the span) is kept and reported: that span is prefilled in two halves. The grid continues
    from a moved boundary, so a short chunk may precede the planned last start.
    """
    num_cached, seq_len, chunk_size, last_abs = int(num_cached), int(seq_len), int(chunk_size), int(last_abs)
    end = num_cached + seq_len
    spans = tuple((int(s), int(e)) for s, e in spans or ())
    starts = [num_cached]
    cur = num_cached
    while True:
        nxt = cur + chunk_size
        if nxt >= last_abs:
            break
        for s, e in spans:
            if s < nxt < e:
                cand = (s // align) * align
                if cand > cur:
                    nxt = cand
                else:
                    logger.warning(
                        "Gemma4 prefill chunking: the vision span [{}, {}) crosses the chunk boundary {} and "
                        "cannot move (chunk start {}); its first rows are prefilled without their later rows",
                        s,
                        e,
                        nxt,
                        cur,
                    )
                break
        starts.append(nxt)
        cur = nxt
    last = last_abs
    for s, e in spans:
        if s < last < e:
            cand = (s // align) * align
            if cand > cur and end - cand <= chunk_size:
                last = cand
            else:
                logger.warning(
                    "Gemma4 prefill chunking: the vision span [{}, {}) crosses the last chunk start {} and cannot "
                    "move; its first rows are prefilled without their later rows",
                    s,
                    e,
                    last,
                )
            break
    starts = [c for c in starts if c < last]
    starts.append(last)
    return starts


def allowed_matrix_abs(abs_start: int, length: int, window, spans) -> torch.Tensor:
    """Boolean ``[length, length]`` ``allowed[r, c]`` for an in-memory square whose row and column ``i``
    is absolute position ``abs_start + i``: causal (``c <= r``), within ``window`` when given
    (``c > r - window``, HF's sliding rule), OR both positions inside one vision span."""
    r = torch.arange(length).view(-1, 1)
    c = torch.arange(length).view(1, -1)
    allowed = c <= r
    if window is not None and int(window) > 0:
        allowed &= c > (r - int(window))
    pos = torch.arange(length) + int(abs_start)
    for s, e in spans:
        inside = (pos >= s) & (pos < e)
        allowed |= inside.view(-1, 1) & inside.view(1, -1)
    return allowed


def additive_mask(allowed: torch.Tensor) -> torch.Tensor:
    """``[1, 1, L, L]`` bf16 additive mask (0 allowed, ``MASK_NEG`` masked)."""
    length = int(allowed.shape[0])
    return (
        torch.where(allowed, torch.zeros((), dtype=torch.float32), torch.full((), MASK_NEG))
        .to(torch.bfloat16)
        .view(1, 1, length, length)
    )


def vision_row_tiles(chunk_start: int, seq_len: int, spans, *, tile: int = TILE_ROWS):
    """Row tiles of the chunk ``[chunk_start, chunk_start + seq_len)`` that hold vision rows.

    Returns ``[(r0, r1, cur_pos), ...]`` with ``r0``/``r1`` tile-aligned chunk-relative row bounds and
    ``cur_pos[j]`` the last absolute key position row ``r0 + j`` attends: its own position for a text
    row (plain causal) and the span's last position for a vision row (the whole span, both ways).
    """
    chunk_start, seq_len = int(chunk_start), int(seq_len)
    inside = spans_in_window(spans, chunk_start, chunk_start + seq_len)
    if not inside:
        return []
    tiles = []
    for t0 in range(0, seq_len, tile):
        t1 = min(t0 + tile, seq_len)
        if not spans_in_window(inside, chunk_start + t0, chunk_start + t1):
            continue
        cur = []
        for row in range(t0, t1):
            pos = chunk_start + row
            end = pos
            for s, e in inside:
                if s <= pos < e:
                    end = e - 1
                    break
            cur.append(end)
        tiles.append((t0, t1, cur))
    return tiles


class MMPrefillVision:
    """The vision spans of one prefill call and the device masks its layers ask for.

    ``spans`` are absolute; ``chunk_start`` / ``seq_len`` describe the chunk this call prefills (the
    padded chunk rows). Masks are built once per ``(abs_start, length, window)`` and freed by
    ``deallocate`` after the call: they are per-prefill device tensors (a persistent cache kept
    buffers alive under the decode traces' baked-in addresses).
    """

    def __init__(self, spans, chunk_start: int, seq_len: int, mesh_device):
        self.spans = tuple((int(s), int(e)) for s, e in spans)
        self.chunk_start = int(chunk_start)
        self.seq_len = int(seq_len)
        self.mesh_device = mesh_device
        self._masks = {}

    def in_chunk(self) -> bool:
        return bool(spans_in_window(self.spans, self.chunk_start, self.chunk_start + self.seq_len))

    def mask(self, abs_start: int, length: int, window):
        """Device additive mask ``[1, 1, length, length]`` of the in-memory square starting at absolute
        position ``abs_start`` (negative rows are the zero-padded tail of a short prefix: causal only)."""
        key = (int(abs_start), int(length), None if window is None else int(window))
        mask = self._masks.get(key)
        if mask is None:
            host = additive_mask(allowed_matrix_abs(key[0], key[1], key[2], self.spans))
            is_mesh = hasattr(self.mesh_device, "shape")
            mask = ttnn.from_torch(
                host,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device) if is_mesh else None,
            )
            self._masks[key] = mask
        return mask

    def vision_row_tiles(self):
        return vision_row_tiles(self.chunk_start, self.seq_len, self.spans)

    def deallocate(self):
        for mask in self._masks.values():
            mask.deallocate(True)
        self._masks = {}
