# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Host bookkeeping for prefix caching of Muse-Glimmer-30B (no ttnn, no device).

The 13 full-attention layers keep K/V in vLLM's paged pool, so a cached prefix block restores them. The 39 sliding
layers keep a ring per slot (window + 256 pad rows) that no block holds, so a hit also needs the ring as it was at the
resume position. The adapter snapshots a slot's ring at the end of every prefill into a retained ring and resumes a
later prefill from that snapshot; this module holds the pure index and registry logic. Design:
profile/claude/projects/muse-glimmer-30b/AGENTX_PREFIX_CACHE_DESIGN.md in the claude-profile repo.
"""

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass, field

import torch

# Resume positions are multiples of this: the chunked prefill SDPA needs chunk_start_idx % k_chunk == 0 (k_chunk 256),
# and a ring snapshot taken at prompt length P serves a resume at floor(P / 256) * 256 because the ring keeps
# window + 256 rows.
CACHE_ALIGN = 256


def enabled() -> bool:
    return os.environ.get("MUSE_PREFIX_CACHE", "0") == "1"


def retained_rings() -> int:
    """Retained snapshot rings per chip (24.4 MB each at 39 sliding layers, 36 blocks, BFP8; default 96 = 2.3 GB per chip with the segment snapshots, keep at least the lane count x 3,
    since every turn of a conversation adds a snapshot and the superseded ones leave only by LRU)."""
    return int(os.environ.get("MUSE_PREFIX_SNAPSHOTS", 96)) if enabled() else 0


def segment_rows() -> int:
    """MUSE_PREFIX_SEGMENT: rows per prefill segment (a multiple of 256) with a ring snapshot after each (default 16384 when the
    cache is on; 0 = off). 8192 would also catch branch points between 8K and 16K but needs twice the snapshots (untested)."""
    n = int(os.environ.get("MUSE_PREFIX_SEGMENT", 16384)) if enabled() else 0
    if n and n % CACHE_ALIGN:
        raise ValueError(f"MUSE_PREFIX_SEGMENT must be a multiple of {CACHE_ALIGN}, got {n}")
    return n


def chain_hashes(tokens, salt: bytes = b"") -> list:
    """digests[k] identifies tokens[: (k + 1) * CACHE_ALIGN]: blake2b chained over CACHE_ALIGN-token blocks."""
    t = torch.as_tensor(tokens).reshape(-1).to(torch.int32).numpy()
    out, prev = [], hashlib.blake2b(salt, digest_size=16).digest()
    for k in range(len(t) // CACHE_ALIGN):
        h = hashlib.blake2b(prev, digest_size=16)
        h.update(t[k * CACHE_ALIGN : (k + 1) * CACHE_ALIGN].tobytes())
        prev = h.digest()
        out.append(prev)
    return out


@dataclass
class PrefillResume:
    """Device-side inputs of a prefill that resumes at position `start` (> 0)."""

    start: int
    ring: int  # ring index of the slot whose sliding caches already hold the window before `start`
    full_fill_pt: object  # ttnn int32 [1, W] fill table of the full-attention layers (fill_row_without_cached)


@dataclass
class PrefillPlan:
    start: int  # first position computed by this prefill (0 = from scratch)
    snapshot_ring: int | None  # retained ring to copy into the slot's live ring before the prefill (None if start == 0)
    cached_tokens: int  # tokens vLLM reports as cached: their full-attention blocks must not be rewritten
    key: bytes | None = None  # registry key of the snapshot used


@dataclass
class RingSnapshotRegistry:
    """LRU map chain-hash -> retained ring index. Rings [first, first + capacity) of the sliding caches are retained."""

    first: int
    capacity: int
    entries: "OrderedDict[bytes, tuple]" = field(default_factory=OrderedDict)  # key -> (ring, boundary, prompt_len)
    hits: int = 0
    misses: int = 0
    stale_cached_tokens: int = 0

    def _free_ring(self) -> int:
        used = {r for r, _, _ in self.entries.values()}
        for r in range(self.first, self.first + self.capacity):
            if r not in used:
                return r
        raise RuntimeError("no free retained ring")

    def plan(self, tokens, cached_tokens: int, digests=None) -> PrefillPlan:
        """Resume position for a prompt whose first `cached_tokens` tokens vLLM already has in the paged pool."""
        digests = chain_hashes(tokens) if digests is None else digests
        top = min(cached_tokens, len(tokens) - 1) // CACHE_ALIGN  # boundaries at or below the cached length
        for k in range(top - 1, -1, -1):  # digests[k] is the boundary (k + 1) * CACHE_ALIGN
            e = self.entries.get(digests[k])
            if e is not None:
                self.entries.move_to_end(digests[k])
                self.hits += 1
                return PrefillPlan((k + 1) * CACHE_ALIGN, e[0], cached_tokens, digests[k])
        if cached_tokens:
            self.misses += 1
            self.stale_cached_tokens += cached_tokens
        return PrefillPlan(0, None, cached_tokens)

    def register(self, tokens, digests=None):
        """Reserve the retained ring that will hold the snapshot of the ring right after prefilling `tokens`.

        Returns (ring, key) or None when the prompt is shorter than one CACHE_ALIGN block or capacity is 0. The caller
        copies the slot's ring into `ring` (device work) and then calls commit(). A key already present is refreshed
        in place (same ring). The least recently used entry is evicted when full.
        """
        digests = chain_hashes(tokens) if digests is None else digests
        if not digests or self.capacity == 0:
            return None
        key = digests[-1]
        boundary = len(digests) * CACHE_ALIGN
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key][0], key
        if len(self.entries) >= self.capacity:
            self.entries.popitem(last=False)
        ring = self._free_ring()
        self.entries[key] = (ring, boundary, len(tokens))
        return ring, key

    def drop(self, key: bytes):
        self.entries.pop(key, None)


def shifted_page_row(page_row: torch.Tensor, start: int, block_size: int, width: int) -> torch.Tensor:
    """Page-table row for filling rows [start, ...): paged_fill_cache always begins at table column 0."""
    assert start % block_size == 0
    out = torch.zeros(width, dtype=torch.int32)
    row = page_row.reshape(-1)[start // block_size :].to(torch.int32)
    n = min(width, row.shape[0])
    out[:n] = row[:n]
    return out


def fill_row_without_cached(
    page_row: torch.Tensor, start: int, cached_tokens: int, block_size: int, width: int
) -> torch.Tensor:
    """Like shifted_page_row, but blocks that vLLM already caches (positions < cached_tokens) point at block 0, vLLM's
    never-allocated null block, so recomputed rows never overwrite blocks other requests may be reading."""
    out = shifted_page_row(page_row, start, block_size, width)
    n_cached = max(0, cached_tokens // block_size - start // block_size)
    out[: min(n_cached, width)] = 0
    return out


def shifted_ring_row(ring: int, start: int, ring_blocks: int, block_size: int, width: int) -> torch.Tensor:
    """Sliding table row for filling rows [start, ...) of a slot: chunk-local block j -> ring block (j + start/block)
    % ring_blocks. With cache_position_modulo = ring_blocks * block_size the fill wraps inside the ring."""
    assert start % block_size == 0
    off = start // block_size
    return (ring * ring_blocks + (torch.arange(width) + off) % ring_blocks).to(torch.int32)


def history_segments(start: int, window: int, block_size: int, ring_blocks: int, ring: int) -> list:
    """Cache block ranges holding rows [start - rows, start), in position order, as (first_block, n_blocks) pairs
    (absolute block indices into the sliding cache, at most two because the ring wraps once). rows = min(window, start)
    rounded to whole blocks; window must be a multiple of block_size."""
    assert start % block_size == 0 and window % block_size == 0
    n = min(window, start) // block_size
    first = start // block_size - n
    segs, j = [], first
    while j < first + n:
        r = j % ring_blocks
        take = min(ring_blocks - r, first + n - j)
        segs.append((ring * ring_blocks + r, take))
        j += take
    return segs


def resume_bucket(suffix_len: int, buckets, start: int, max_seq_len: int) -> int:
    """Padded length of a resumed prefill: the warm bucket, capped so start + pad stays inside the rope tables."""
    for b in buckets:
        if b >= suffix_len:
            return min(b, max_seq_len - start)
    raise ValueError(f"suffix of {suffix_len} tokens exceeds the largest bucket {buckets[-1]}")
