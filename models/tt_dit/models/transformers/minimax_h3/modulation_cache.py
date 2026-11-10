# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Reuse of the per-block AdaLN modulation tables across requests.

A block's six modulation tables are a pure function of `temb`, and `temb` is a pure function of the per-slot timestep
levels (and the two-time endpoints) the pipeline uploads for a step. A served schedule repeats the same levels on every
request, so the `silu -> adaln_proj -> layout` chain (one weight-bound projection and ~25 small ops per block per
forward) recomputes identical tables. This cache keeps them, keyed on the host-side levels.

Exactness: a hit hands the blocks device copies (`ttnn.copy`, same spec) of the tables the unchanged
`_modulation_tables` produced for the same levels on an earlier forward; the device ops are deterministic, so the
copies are the bytes today's code would compute again.

Trace safety: a buffer allocated after a trace capture can be overwritten when that trace replays (the allocator warns
"Allocating device buffers is potentially unsafe due to the existence of an active trace"), and the pipeline replays
its audio-decode traces at the end of every request. So every buffer the cache keeps is allocated in one pool, once,
and only while no trace has been captured; afterwards a new key reuses the least recently used slot in place
(`ttnn.copy`), never allocating. If the pool could not be allocated before the first capture, the cache stays off and
every forward computes its tables as before. Traced forwards never use the cache (the caller passes no key).
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from typing import Any

import ttnn


class ModulationTableCache:
    """LRU of `capacity` slots, each holding `num_blocks` lists of tables (one list per block; a caller may add entries
    past the blocks, e.g. `temb`, and request them in any order).

    `trace_captured` reports whether any trace has been captured on the device yet; the pool is only allocated while
    it returns False.
    """

    def __init__(self, *, capacity: int, num_blocks: int, mesh_device: Any, trace_captured: Callable[[], bool]) -> None:
        self.capacity = max(0, int(capacity))
        self.num_blocks = num_blocks
        self.mesh_device = mesh_device
        self._trace_captured = trace_captured
        self._slots: list[list[list[Any]]] | None = None
        self._keys: OrderedDict[Hashable, int] = OrderedDict()
        self._free: list[int] = []
        self.disabled = self.capacity == 0
        self.hits = 0
        self.misses = 0

    @property
    def allocated(self) -> bool:
        return self._slots is not None

    def plan(self, key: Hashable | None) -> _Plan | None:
        """The plan for one forward, or None when the forward must compute its tables as before."""
        if key is None or self.disabled:
            return None
        if self._slots is None:
            return _Plan(self, key, slot=None, hit=False)
        slot = self._keys.get(key)
        if slot is not None:
            self._keys.move_to_end(key)
            self.hits += 1
            return _Plan(self, key, slot=slot, hit=True)
        self.misses += 1
        if self._free:
            slot = self._free.pop()
        else:
            _, slot = self._keys.popitem(last=False)
        return _Plan(self, key, slot=slot, hit=False)

    def clear(self) -> None:
        """Forget every key (the pool stays allocated). Call when the tables' inputs change, e.g. new weights."""
        if self._slots is not None:
            self._free = list(range(len(self._slots)))
        self._keys.clear()

    def _allocate(self, specs: Sequence[Sequence[Any]]) -> None:
        if self._trace_captured():
            self.disabled = True
            return
        if any(spec.memory_config.buffer_type != ttnn.BufferType.DRAM for block in specs for spec in block):
            self.disabled = True
            return
        self._slots = [
            [[ttnn.allocate_tensor_on_device(spec, self.mesh_device) for spec in block] for block in specs]
            for _ in range(self.capacity)
        ]
        self._free = list(range(self.capacity))


class _Plan:
    """Where one forward's tables come from: a filled slot (hit), a slot to fill (miss), or nowhere (pool not yet
    allocated)."""

    def __init__(self, cache: ModulationTableCache, key: Hashable, *, slot: int | None, hit: bool) -> None:
        self._cache = cache
        self._key = key
        self._slot = slot
        self._hit = hit
        self._specs: dict[int, list[Any]] | None = {} if slot is None else None
        self._filled = 0

    @property
    def slot(self) -> int | None:
        """The pool slot this forward reads (hit) or fills (miss); None before the pool exists."""
        return self._slot

    def slot_tables(self, index: int) -> list[Any]:
        """Entry `index` of this forward's slot: the pool's own buffers (filled once `tables(index, ...)` ran)."""
        return self._cache._slots[self._slot][index]

    def tables(self, index: int, compute: Callable[[], list[Any]]) -> list[Any]:
        """Block `index`'s tables. On a miss the block's own tables are returned (exactly today's tensors) after being
        copied into the slot."""
        cache = self._cache
        if self._hit:
            return cache._slots[self._slot][index]
        tables = compute()
        if self._slot is not None:
            for value, dest in zip(tables, cache._slots[self._slot][index], strict=True):
                ttnn.copy(value, dest)
            self._filled += 1
        elif self._specs is not None:
            self._specs[index] = [t.spec for t in tables]
        return tables

    def finish(self) -> None:
        """Call after the last block: registers a filled slot, or allocates the pool on the first keyed forward."""
        cache = self._cache
        if self._hit:
            return
        if self._slot is not None:
            if self._filled == cache.num_blocks:
                cache._keys[self._key] = self._slot
            else:
                cache._free.append(self._slot)
            return
        if self._specs is not None and len(self._specs) == cache.num_blocks and not cache.allocated:
            cache._allocate([self._specs[index] for index in range(cache.num_blocks)])
