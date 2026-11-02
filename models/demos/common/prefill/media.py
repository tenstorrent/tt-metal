# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Which image embedding rows a prefill chunk injects at its image positions, laid out like the chunk's tokens."""

from dataclasses import dataclass
from typing import Optional, Protocol

import torch

from models.demos.common.prefill.chunk_layout import rotated_chunk_positions


@dataclass(frozen=True)
class MediaItem:
    start_position: int
    num_tokens: int
    media_id: bytes

    def __post_init__(self):
        assert self.num_tokens > 0, "media item with no tokens"

    @property
    def end_position(self) -> int:
        return self.start_position + self.num_tokens


@dataclass(frozen=True)
class MediaRows:
    """Rows [row_start, row_start + len(rows)) of an item's [num_tokens, hidden] embeddings."""

    item: MediaItem
    row_start: int
    rows: torch.Tensor

    def __post_init__(self):
        assert self.rows.dim() == 2, f"rows {tuple(self.rows.shape)} for {self.item}, want [n, hidden]"
        assert (
            0 <= self.row_start and self.row_end <= self.item.num_tokens
        ), f"rows [{self.row_start},{self.row_end}) for {self.item}"

    @property
    def row_end(self) -> int:
        return self.row_start + self.rows.shape[0]


class EmbeddingSource(Protocol):
    def rows(self, item: MediaItem, lo: int, hi: int) -> torch.Tensor:
        """Rows [lo, hi) of the item's [num_tokens, hidden] embeddings."""


class StubEmbeddingSource(EmbeddingSource):
    """Stands in for the encoder service: deterministic embeddings seeded from the media ID."""

    def __init__(self, hidden: int, dtype=torch.bfloat16):
        self.hidden = hidden
        self.dtype = dtype

    def rows(self, item: MediaItem, lo: int, hi: int) -> torch.Tensor:
        gen = torch.Generator().manual_seed(int.from_bytes(item.media_id[:8], "little"))
        return (torch.randn(item.num_tokens, self.hidden, generator=gen) * 0.02).to(self.dtype)[lo:hi]


def items_overlapping(items: list[MediaItem], start: int, end: int) -> list[MediaItem]:
    """Items with at least one position in [start, end), in prompt order."""
    return sorted(
        (m for m in items if m.start_position < end and m.end_position > start), key=lambda m: m.start_position
    )


def chunk_row_range(item: MediaItem, chunk_start: int, actual_end: int) -> tuple[int, int]:
    """The item's rows [lo, hi) whose positions fall inside [chunk_start, actual_end)."""
    lo = max(item.start_position, chunk_start) - item.start_position
    hi = min(item.end_position, actual_end) - item.start_position
    assert lo < hi, f"{item} does not overlap [{chunk_start},{actual_end})"
    return lo, hi


def media_rows_for_chunk(
    items: list[MediaItem], chunk_start: int, actual_end: int, source: EmbeddingSource
) -> list[MediaRows]:
    out = []
    for item in items_overlapping(items, chunk_start, actual_end):
        lo, hi = chunk_row_range(item, chunk_start, actual_end)
        out.append(MediaRows(item, lo, source.rows(item, lo, hi)))
    return out


def build_injection(
    slices: list[MediaRows],
    chunk_start: int,
    actual_end: int,
    sp: int,
    chunk_local: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """(rows [sp, chunk_local, hidden], mask [sp, chunk_local]) in the chunk's rotated per-chip order, or None
    for no slices. Each slice must cover its chunk_row_range."""
    if not slices:
        return None
    hidden, dtype = slices[0].rows.shape[-1], slices[0].rows.dtype
    positions = torch.tensor(rotated_chunk_positions(chunk_start, sp, chunk_local), dtype=torch.long)
    rows = torch.zeros(sp, chunk_local, hidden, dtype=dtype)
    mask = torch.zeros(sp, chunk_local, dtype=torch.bool)
    for s in slices:
        item = s.item
        assert (
            s.rows.shape[-1] == hidden and s.rows.dtype == dtype
        ), f"rows {tuple(s.rows.shape)} {s.rows.dtype} for {item}"
        lo, hi = chunk_row_range(item, chunk_start, actual_end)
        assert (
            s.row_start <= lo and hi <= s.row_end
        ), f"rows [{s.row_start},{s.row_end}) for {item}, chunk needs [{lo},{hi})"
        sel = (positions >= item.start_position + lo) & (positions < item.start_position + hi)
        assert not (mask & sel).any(), f"media items overlap at {item}"
        rows[sel] = s.rows[positions[sel] - item.start_position - s.row_start]
        mask |= sel
    return rows, mask
