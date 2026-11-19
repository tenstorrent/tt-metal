# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fixed-address request-state migration, with no model tensor host readback.

Independent per-slot KV allocations move with request state. Shared serving
pools remain in place while their page tables move. Allocate one scratch
RequestCache before trace capture; never swap caches or their tensor objects.
The scheduler's remap is destination -> source and covers every allocated slot.
Sampling state is intentionally owned separately (see slot_state_audit.md).
"""

from __future__ import annotations

import operator


def validate_remap(remap, width):
    """Validate a full permutation without importing TTNN or reading devices."""
    if getattr(getattr(remap, "device", None), "type", "cpu") != "cpu":
        raise ValueError("slot_remap must be host scheduler metadata")
    if hasattr(remap, "tolist"):
        remap = remap.tolist()
    try:
        values = tuple(operator.index(value) for value in remap)
    except (TypeError, ValueError) as error:
        raise ValueError("slot_remap must contain integer slot indices") from error
    if len(values) != width or sorted(values) != list(range(width)):
        raise ValueError(f"slot_remap must permute all {width} slots")
    return values


def permutation_cycles(remap):
    """Return nontrivial cycles in destination-to-source traversal order."""
    remap = validate_remap(remap, len(remap))
    visited, cycles = set(), []
    for start in range(len(remap)):
        if start in visited or remap[start] == start:
            continue
        cycle, slot = [], start
        while slot not in visited:
            visited.add(slot)
            cycle.append(slot)
            slot = remap[slot]
        cycles.append(tuple(cycle))
    return tuple(cycles)


def _state_tensors(cache):
    result = []
    for index, layer in enumerate(cache.layers):
        for name in ("conv", "recurrent", "ple_conv", "token_history"):
            tensor = getattr(layer, name, None)
            if tensor is not None:
                result.append((f"layers.{index}.{name}", tensor))
        attention = layer.attention
        if attention is not None:
            for name in ("position", "page_table", "k_cache", "v_cache", "raw_history", "compressed_keys"):
                if getattr(cache, "shared_kv", False) and name in ("k_cache", "v_cache"):
                    continue
                result.append((f"layers.{index}.attention.{name}", getattr(attention, name)))
    # hash_operations tokens/control/output are overwritten scratch or constants;
    # its history aliases token_history. Index-number arrays are immutable/shared.
    return tuple(result)


def _signature(cache):
    return tuple((name, tuple(tensor.shape), tensor.dtype, tensor.layout) for name, tensor in _state_tensors(cache))


def _device_copy(source, destination):
    import ttnn

    ttnn.copy(source, destination)


class SlotStateMover:
    """Cycle-copy request state through one setup-time scratch allocation.

    All operations enqueue on the default CQ, in order with decode. Call
    warmup before captures. apply() is a transition operation, never a per-token
    operation, and keeps every traced tensor address unchanged. Parent must
    validate all other submission arguments before calling this mutating API.
    """

    def __init__(self, caches, scratch=None, *, copy_tensor=None):
        self.caches = tuple(caches)
        if not self.caches:
            raise ValueError("At least one cache is required")
        if len(self.caches) > 1 and scratch is None:
            raise ValueError("Multiple slots require an independent setup-time scratch cache")
        self.scratch = scratch
        self.copy_tensor = _device_copy if copy_tensor is None else copy_tensor
        self.copy_replays = {}
        signature, capacity = _signature(self.caches[0]), self.caches[0].capacity
        all_caches = self.caches + (() if scratch is None else (scratch,))
        seen = set()
        for cache in all_caches:
            if cache.capacity != capacity or _signature(cache) != signature:
                raise ValueError("All slots and scratch must have identical capacity, shape, dtype and layout")
            for name, tensor in _state_tensors(cache):
                if id(tensor) in seen:
                    raise ValueError(f"Aliased mutable cache tensor: {name}")
                seen.add(id(tensor))

    def _copy_tensors(self, source, destination):
        for (_, src), (_, dst) in zip(_state_tensors(source), _state_tensors(destination), strict=True):
            self.copy_tensor(src, dst)

    def _copy(self, source, destination):
        replay = self.copy_replays.get((id(source), id(destination)))
        if replay is None:
            self._copy_tensors(source, destination)
        else:
            replay()
        destination.position = source.position  # Scheduler metadata only.

    def prepare_copy_traces(self, capture):
        """Bind bounded scratch transfers after generator-wide compilation.

        capture(body, signature) returns a nonblocking replay callable owned by
        the generator. Only existing persistent tensors are read/written; no
        state, bindings or Python positions are changed by capture itself.
        Ordinary slot-to-slot migration keeps the warmed eager copy path.
        """
        if self.scratch is None or self.copy_replays:
            return
        for slot, cache in enumerate(self.caches):
            for source, destination, direction in (
                (cache, self.scratch, "to_scratch"),
                (self.scratch, cache, "from_scratch"),
            ):
                self.copy_replays[id(source), id(destination)] = capture(
                    lambda source=source, destination=destination: self._copy_tensors(source, destination),
                    ("state_copy", slot, direction),
                )

    def warmup(self):
        """Compile every persistent tensor copy shape without changing live state."""
        if self.scratch is not None:
            self._copy(self.caches[0], self.scratch)
            self._copy(self.scratch, self.caches[0])

    def apply(self, remap):
        remap = validate_remap(remap, len(self.caches))
        for cycle in permutation_cycles(remap):
            self._copy(self.caches[cycle[0]], self.scratch)
            for destination, source in zip(cycle, cycle[1:]):
                self._copy(self.caches[source], self.caches[destination])
            self._copy(self.scratch, self.caches[cycle[-1]])
        return remap


def remap_token_buffer(tokens, remap, *, warmup=False):
    """Gather resident [1,1,1,32] uint32 feedback entirely on device.

    Warm with identity and warmup=True before capture. This compiles each slice
    offset and the 32-input concat/copy. The concat completes before overwrite;
    sliced inputs are kept alive until after that copy. Tail rows stay identity.
    """
    import ttnn

    remap = validate_remap(remap, len(remap))
    if tuple(tokens.shape) != (1, 1, 1, 32) or len(remap) > 32:
        raise ValueError("Expected physical-32 token feedback and at most 32 slots")
    if not warmup and all(source == destination for destination, source in enumerate(remap)):
        return
    full = remap + tuple(range(len(remap), 32))
    rows = [tokens[:, :, :, source : source + 1] for source in full]
    gathered = ttnn.concat(rows, dim=3)
    ttnn.copy(gathered, tokens)


def remap_slot_mapping(mapping, remap):
    """Reindex slot-keyed host metadata without changing values or inventing rows.

    Appropriate for page-table upload shadows and per-request HostSampler
    objects. Such metadata never replaces the resident cache/feedback tensors.
    """
    remap = validate_remap(remap, len(remap))
    if any(not 0 <= key < len(remap) for key in mapping):
        raise ValueError("Slot mapping contains an out-of-range key")
    return {destination: mapping[source] for destination, source in enumerate(remap) if source in mapping}
