# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
"""LRU, threshold-based eviction for per-geometry device tensor caches.

The DRAM leak this closes: `TtConv1d`/`TtConvTranspose1d` cache prepared conv weights
(and, separately, which `(weight, bias, compute_config)` triple was verified correct) per
`(input_length, batch_size)` geometry -- real, measured growth across the 2026-09-21
regression run's four utterance lengths: 90.9 MB/bank after the first length, 134.1 MB/bank
after the fourth (see BRINGUP_STATUS.md). Nothing ever evicted an old geometry's entries,
so DRAM grows without bound as a session sees more distinct lengths.

Designed for reuse, not as a one-off patch for this specific leak: streaming (bounty
Stages 2/3) will churn through many more distinct geometries per session as chunk shapes
vary, and threshold-based eviction (check real DRAM pressure, evict the least-recently-used
entries only when actually needed) is the same mechanism that workload will need, not a
different one. Per the user's explicit instruction this round: eviction triggers on a real
free-DRAM threshold, not per-utterance or on any other proxy schedule.

Usage: a cache instance backs each distinct dict a class used to keep directly (e.g. one
for `_prep_cache`, a separate one for `_verified_config`, since they can have different
lifetimes for the same geometry key -- see TtConv1d/TtConvTranspose1d for exactly how
ownership is threaded between the two so nothing is double-freed or dropped while still
referenced).
"""
from __future__ import annotations

import os
from collections import OrderedDict
from typing import Any

import ttnn

# Free-DRAM-per-bank floor (MB) below which the LRU eviction kicks in on the next `put()`.
# One knob, shared by every GeometryWeightCache instance on a given device -- matches this
# package's existing COSYVOICE2_* env-var convention. Read at cache-construction time.
_DEFAULT_THRESHOLD_MB = 150


def dram_free_threshold_mb() -> int:
    return int(os.environ.get("COSYVOICE2_DRAM_FREE_THRESHOLD_MB", _DEFAULT_THRESHOLD_MB))


class GeometryWeightCache:
    """Maps an arbitrary hashable geometry key -> caller-defined metadata, backed by zero
    or more device tensors this cache instance owns and is responsible for freeing.

    Not every cached entry owns tensors: a fallback path (e.g. `prepare_conv_weights`
    unavailable, falling back to a conv's own permanent raw weight) caches metadata that
    references tensors owned elsewhere -- passing an empty `owned_tensors` list for those
    makes eviction a no-op for that entry's tensors (their bookkeeping is still dropped,
    but nothing is deallocated), which is exactly what must happen: a per-geometry cache
    must never free a tensor some OTHER geometry, or the instance itself, still needs.
    """

    def __init__(self, device, threshold_mb: int | None = None):
        self.device = device
        self._threshold_bytes = (threshold_mb if threshold_mb is not None else dram_free_threshold_mb()) * 1024 * 1024
        self._entries: OrderedDict[Any, tuple[list, Any]] = OrderedDict()

    def _free_bytes_per_bank(self) -> int:
        return ttnn.get_memory_view(self.device, ttnn.BufferType.DRAM).total_bytes_free_per_bank

    def get(self, key) -> Any | None:
        """Returns the cached metadata, or None on a miss. A hit refreshes the key's LRU
        position (most-recently-used)."""
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry[1]

    def __getitem__(self, key) -> Any:
        """Dict-style access for callers that know the key is present (raises KeyError
        otherwise) -- `get` is the miss-tolerant form everything in this package's
        production code actually uses."""
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(key)
        self._entries.move_to_end(key)
        return entry[1]

    def put(self, key, owned_tensors: list, metadata: Any) -> None:
        """Inserts (or replaces) an entry, then evicts least-recently-used entries (other
        than the one just inserted) while free DRAM is below threshold. Called on every
        insertion, not on a per-utterance schedule -- a session that never gets DRAM-tight
        never evicts anything, exactly as it shouldn't."""
        self._entries[key] = (list(owned_tensors), metadata)
        self._entries.move_to_end(key)
        self._evict_while_needed(protect=key)

    def discard(self, key) -> None:
        """Drops a key's bookkeeping WITHOUT deallocating its tensors -- for the case
        where ownership of those exact tensors is being transferred to another cache (or
        to the instance itself) rather than actually freed. Using `pop` here instead would
        be a double-free the moment the new owner is also evicted."""
        self._entries.pop(key, None)

    def pop(self, key) -> None:
        """Removes a key and deallocates the tensors it owns, if any. Safe to call on a
        missing key (no-op)."""
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        for t in entry[0]:
            ttnn.deallocate(t)

    def clear(self) -> None:
        """Deallocates every owned tensor and drops all entries -- the explicit,
        all-at-once counterpart to threshold-based eviction (e.g. for a `release_caches()`
        call at a natural session boundary)."""
        for key in list(self._entries.keys()):
            self.pop(key)

    def _evict_while_needed(self, protect) -> None:
        while self._free_bytes_per_bank() < self._threshold_bytes:
            victim = next((k for k in self._entries if k != protect), None)
            if victim is None:  # only `protect` (or nothing) left -- can't evict further
                return
            self.pop(victim)

    def __len__(self) -> int:
        return len(self._entries)
