# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side media handling for prefill: which image rows a chunk injects, the encoder cache, stub embeddings.

The adapter expands each image into num_tokens placeholder tokens starting at start_position. On stage 0 the
runner overwrites the token embeddings at those positions with the image's embedding rows. A chunk injects only
the rows whose positions fall inside it, laid out in the same rotated block-cyclic order as the chunk's tokens.
"""

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Optional

import torch

from models.demos.common.prefill.chunk_layout import rotated_chunk_positions

MEDIA_HASH_BYTES = 32


@dataclass(frozen=True)
class MediaItem:
    start_position: int
    num_tokens: int
    hash: bytes

    def __post_init__(self):
        assert len(self.hash) == MEDIA_HASH_BYTES, f"media hash is {len(self.hash)} bytes, want {MEDIA_HASH_BYTES}"
        assert self.num_tokens > 0, "media item with no tokens"

    @property
    def end_position(self) -> int:
        return self.start_position + self.num_tokens


def items_overlapping(items: list[MediaItem], start: int, end: int) -> list[MediaItem]:
    """Items with at least one position in [start, end), in prompt order."""
    return sorted(
        (m for m in items if m.start_position < end and m.end_position > start), key=lambda m: m.start_position
    )


def build_injection(
    embeddings: dict[bytes, torch.Tensor],
    items: list[MediaItem],
    chunk_start: int,
    actual_end: int,
    sp: int,
    chunk_local: int,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    """Dense rows and mask for one chunk, in the chunk's rotated per-chip order.

    Returns (rows [sp, chunk_local, hidden], mask [sp, chunk_local] bool), or None when no image overlaps
    [chunk_start, actual_end). embeddings maps each overlapping item's hash to its full [num_tokens, hidden]
    embedding; only the rows inside the chunk are used. Masked-off rows are zero.
    """
    hits = items_overlapping(items, chunk_start, actual_end)
    if not hits:
        return None
    first = embeddings[hits[0].hash]
    hidden, dtype = first.shape[-1], first.dtype
    positions = torch.tensor(rotated_chunk_positions(chunk_start, sp, chunk_local), dtype=torch.long)
    rows = torch.zeros(sp, chunk_local, hidden, dtype=dtype)
    mask = torch.zeros(sp, chunk_local, dtype=torch.bool)
    for item in hits:
        emb = embeddings[item.hash]
        assert emb.shape == (item.num_tokens, hidden), f"embedding {tuple(emb.shape)} for {item}"
        lo, hi = max(item.start_position, chunk_start), min(item.end_position, actual_end)
        sel = (positions >= lo) & (positions < hi)
        assert not (mask & sel).any(), f"media items overlap at {item}"
        rows[sel] = emb[positions[sel] - item.start_position]
        mask |= sel
    return rows, mask


def stub_embeddings(media_hash: bytes, num_tokens: int, hidden: int, dtype=torch.bfloat16) -> torch.Tensor:
    """Deterministic placeholder embeddings seeded from the media hash, until the vision encoder is wired in."""
    gen = torch.Generator().manual_seed(int.from_bytes(media_hash[:8], "little"))
    return (torch.randn(num_tokens, hidden, generator=gen) * 0.02).to(dtype)


class EncoderCache:
    """Per-worker LRU of image embeddings keyed by media hash, bounded by bytes. Lost on worker restart."""

    def __init__(self, capacity_bytes: int):
        assert capacity_bytes >= 0
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[bytes, torch.Tensor] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, media_hash: bytes) -> bool:
        return media_hash in self._entries

    def get_or_compute(self, media_hash: bytes, compute: Callable[[], torch.Tensor]) -> torch.Tensor:
        emb = self._entries.get(media_hash)
        if emb is not None:
            self._entries.move_to_end(media_hash)
            self.hits += 1
            return emb
        self.misses += 1
        emb = compute()
        self._insert(media_hash, emb)
        return emb

    def _insert(self, media_hash: bytes, emb: torch.Tensor) -> None:
        size = emb.numel() * emb.element_size()
        if size > self.capacity_bytes:
            return
        while self.bytes + size > self.capacity_bytes:
            _, old = self._entries.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
            self.evictions += 1
        self._entries[media_hash] = emb
        self.bytes += size

    def stats_line(self) -> str:
        return (
            f"ENCODER_CACHE hits={self.hits} misses={self.misses} evictions={self.evictions} "
            f"entries={len(self)} bytes={self.bytes} capacity_bytes={self.capacity_bytes}"
        )
