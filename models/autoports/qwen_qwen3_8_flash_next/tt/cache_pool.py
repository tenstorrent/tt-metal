# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Setup-time shared paged KV storage; no TTNN import until allocation.

Each allocator spec is the pinned plugin's (shape, dtype, tensor_idx), already
TP-folded to [global_blocks, 1, 32, 256]. Specs enumerate indexed-attention
layers, not recurrent layers. Repeated tensor_idx means the SAME K/V pair;
those layers must use disjoint global block IDs from distinct block tables.

This module does not make the existing generator pool-aware by itself. Bind
before prepare, rebuild its slot mover excluding shared K/V, and never zero
shared K/V during request reset. Pool lifetimes must exceed every bound trace.
No cache-prefix reuse: the recurrent/indexer state is still request-owned.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass


def _integer(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer") from error


def dtype_name(dtype):
    name = str(dtype).lower().split(".")[-1]
    if name not in ("bfloat16", "bfloat8_b"):
        raise ValueError("KV storage supports BF16 or BFP8")
    return name


@dataclass(frozen=True)
class PoolSpec:
    shape: tuple[int, int, int, int]
    tensor_idx: int
    dtype: str = "bfloat16"

    @property
    def bytes_per_rank(self):
        return 2 * self.shape[0] * 8 * (2048 if self.dtype == "bfloat16" else 1088)


def normalize_specs(per_layer_specs):
    """Validate the complete allocation plan before allocating anything."""
    specs, unique = [], {}
    for entry in per_layer_specs:
        if len(entry) != 3:
            raise ValueError("Expected (shape, dtype, tensor_idx) per indexed-attention layer")
        shape, dtype, tensor_idx = entry
        shape = tuple(_integer(x, "shape") for x in shape)
        tensor_idx = _integer(tensor_idx, "tensor_idx")
        if len(shape) != 4 or not 1 <= shape[0] <= 2**31 - 1 or shape[1:] != (1, 32, 256):
            raise ValueError("Accepted TP4 KV pool shape is (global_blocks, 1, 32, 256)")
        dtype = dtype_name(dtype)
        if tensor_idx < 0:
            raise ValueError("tensor_idx must be nonnegative")
        spec = PoolSpec(shape, tensor_idx, dtype)
        if tensor_idx in unique and unique[tensor_idx] != spec:
            raise ValueError("Shared tensor_idx requires identical shape and dtype")
        unique[tensor_idx] = spec
        specs.append(spec)
    if not specs:
        raise ValueError("At least one indexed-attention pool spec is required")
    return tuple(specs)


def _allocate_pair(mesh, spec):
    import ttnn

    return tuple(
        ttnn.zeros(
            spec.shape,
            device=mesh,
            dtype=getattr(ttnn, spec.dtype),
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for _ in range(2)
    )


class SharedKVPool:
    """Own one stable K/V allocation per unique tensor_idx.

    allocate_pair is a CPU-test seam: (mesh, PoolSpec) -> (K, V). bind() changes
    only K/V tensor references; all request state and page-table allocations
    remain fixed. Displaced tensors are returned, never implicitly destroyed,
    so the owner can deallocate them only after checking other references.
    Use freeze() immediately before prepare/capture to reject later binding.
    """

    def __init__(self, per_layer_specs, mesh=None, *, allocate_pair=None):
        self.specs = normalize_specs(per_layer_specs)
        allocator = _allocate_pair if allocate_pair is None else allocate_pair
        self.pairs = {}
        self.frozen = False
        self.bound = []
        for spec in self.specs:
            if spec.tensor_idx not in self.pairs:
                pair = tuple(allocator(mesh, spec))
                if len(pair) != 2 or pair[0] is pair[1]:
                    raise ValueError("Allocator must return independent K and V tensors")
                for tensor in pair:
                    if tuple(tensor.shape) != spec.shape or dtype_name(tensor.dtype) != spec.dtype:
                        raise ValueError("Allocator returned an incompatible KV tensor")
                    if "tile" not in str(tensor.layout).lower():
                        raise ValueError("Pool tensors must retain the accepted tiled layout")
                self.pairs[spec.tensor_idx] = pair
        identities = [id(t) for pair in self.pairs.values() for t in pair]
        if len(set(identities)) != len(identities):
            raise ValueError("Different tensor_idx allocations must not alias")

    @property
    def bytes_per_rank(self):
        return sum(spec.bytes_per_rank for spec in {s.tensor_idx: s for s in self.specs}.values())

    @property
    def shared_tensor_ids(self):
        """Exclude precisely these objects from slot copies and request resets."""
        return frozenset(id(t) for pair in self.pairs.values() for t in pair)

    def pair_for_attention_layer(self, index):
        """For a future direct state constructor, avoiding private KV temporaries."""
        return self.pairs[self.specs[index].tensor_idx]

    def bind(self, caches):
        """Bind existing empty request/scratch caches atomically before prepare.

        Scratch aliases the same pool and must never receive K/V copies.
        Canonical prefill executes attention through scratch after copying the
        current request's private state and page table; scratch borrows those
        pages and has no independent simultaneous request ownership.
        Page-table width remains per-request logical capacity,
        independent of global pool size. Existing page-table content is not a
        scheduler allocation: callers must load valid global IDs before use.
        """
        if self.frozen:
            raise RuntimeError("Cannot bind shared pools after freeze/prepare")
        caches = tuple(caches)
        edits = []
        for cache in caches:
            if cache.position != 0:
                raise ValueError("Only empty request caches may be rebound")
            states = [layer.attention for layer in cache.layers if layer.attention is not None]
            if len(states) != len(self.specs):
                raise ValueError("Pool specs must enumerate every indexed-attention layer in decoder order")
            for state, spec in zip(states, self.specs, strict=True):
                if state.page_size != 32 or tuple(state.page_table.shape) != (1, state.capacity // 32):
                    raise ValueError("Request page-table geometry must retain its page32 context contract")
                if state.capacity < cache.capacity or state.capacity % 128:
                    raise ValueError("Request state capacity must cover context and remain 128-aligned")
                edits.append((state, self.pairs[spec.tensor_idx]))
        displaced = {}
        for state, pair in edits:
            for name, tensor in zip(("k_cache", "v_cache"), pair, strict=True):
                old = getattr(state, name)
                if old is not None and id(old) not in self.shared_tensor_ids:
                    displaced[id(old)] = old
                setattr(state, name, tensor)
        for cache in caches:
            if all(cache is not previous for previous in self.bound):
                self.bound.append(cache)
            cache.shared_kv = True
            cache.kv_bound = True
            cache.shared_kv_pool = self  # Keep the allocation owner alive with every request/trace.
        return tuple(displaced.values())

    def freeze(self):
        self.frozen = True

    def warmup_tables(self, *, batch_size, capacity, tokens_per_slot=64):
        """Reserve temporary disjoint pages before the scheduler starts.

        Current prepare leaves slot0 at position32 after its last prefill,
        then runs up to32 decode widths without resetting. Two pages per slot
        cover that warmup. These tables are temporary, not scheduler ownership;
        install actual scheduler tables before any real request. Canonical
        prefill scratch borrows the current request's pages through its copied
        table, so it needs no independent block reservation.
        """
        batch_size = _integer(batch_size, "batch_size")
        tokens_per_slot = _integer(tokens_per_slot, "tokens_per_slot")
        if tokens_per_slot < 1 or tokens_per_slot > capacity:
            raise ValueError("Warmup token count must fit request capacity")
        pages, offsets, tables = (tokens_per_slot + 31) // 32, {}, []
        for spec in self.specs:
            start = offsets.get(spec.tensor_idx, 0)
            end = start + pages * batch_size
            if end > spec.shape[0]:
                raise ValueError("Global pool cannot hold disjoint warmup pages for all fixed slots")
            tables.append([list(range(start + row * pages, start + (row + 1) * pages)) for row in range(batch_size)])
            offsets[spec.tensor_idx] = end
        return self.normalize_tables(
            batch_size=batch_size,
            capacity=capacity,
            page_tables_by_layer=tables,
            lengths=[tokens_per_slot] * batch_size,
        )

    def normalize_tables(self, *, batch_size, capacity, page_table=None, page_tables_by_layer=None, lengths=None):
        """Return CPU int32 [B, ceil128(capacity)/32] tables without device writes.

        Either supply one common table, or an explicit table per attention
        layer in allocation order. Short tables are zero-padded internally.
        When lengths are given, scheduler -1 padding beyond the live prefix is
        replaced with zero; valid future block reservations are preserved. The caller must supply a full
        live-slot snapshot to additionally rule out overlap with dormant rows.
        """
        import torch

        batch_size, capacity = _integer(batch_size, "batch_size"), _integer(capacity, "capacity")
        if not 1 <= batch_size <= 32 or capacity < 1:
            raise ValueError("Expected 1..32 slots and positive request capacity")
        if (page_table is None) == (page_tables_by_layer is None):
            raise ValueError("Supply exactly one common table or per-layer tables")
        tables = [page_table] * len(self.specs) if page_tables_by_layer is None else page_tables_by_layer
        if len(tables) != len(self.specs):
            raise ValueError("Expected one table per indexed-attention layer")
        if lengths is not None:
            lengths = tuple(_integer(n, "length") for n in lengths)
            if len(lengths) != batch_size or any(n < 0 or n > capacity for n in lengths):
                raise ValueError("Lengths must cover every slot and lie within request capacity")
        width = ((capacity + 127) // 128) * 4
        result = []
        for table, spec in zip(tables, self.specs, strict=True):
            if getattr(getattr(table, "device", None), "type", "cpu") != "cpu":
                raise ValueError("Page tables must be host scheduler metadata")
            table = table.tolist() if hasattr(table, "tolist") else table
            if len(table) != batch_size:
                raise ValueError("Page-table rows must match batch_size")
            output = torch.zeros(batch_size, width, dtype=torch.int32)
            for row, values in enumerate(table):
                if len(values) > width:
                    raise ValueError("Page table exceeds the request's physical context width")
                count = len(values) if lengths is None else (lengths[row] + 31) // 32
                if count > len(values):
                    raise ValueError("Page table does not cover the valid token length")
                blocks = []
                for index, value in enumerate(values):
                    block = _integer(value, "global block ID")
                    if lengths is not None and index >= count and block == -1:
                        block = 0
                    if not 0 <= block < spec.shape[0]:
                        raise ValueError("Global block ID is outside the allocated pool")
                    blocks.append(block)
                if blocks:
                    output[row, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
            result.append(output)
        if lengths is not None:
            self.validate_active_pages(result, lengths)
        return tuple(result)

    def validate_active_pages(self, per_layer_tables, lengths):
        """CPU scheduler-metadata validation of a COMPLETE live-slot snapshot.

        tables[attention_layer][slot][logical_page] are global block IDs.
        lengths are valid token counts INCLUDING the token about to be written.
        Unused table padding is ignored. Every live prefix page must be unique
        within each shared tensor_idx; KV-only prefix sharing is unsupported.
        Run on scheduling changes, not every token. This does not read a device.
        """
        lengths = tuple(_integer(n, "length") for n in lengths)
        if any(n < 0 for n in lengths) or len(per_layer_tables) != len(self.specs):
            raise ValueError("Expected nonnegative lengths and one table per indexed-attention layer")
        seen = set()
        for layer, (table, spec) in enumerate(zip(per_layer_tables, self.specs, strict=True)):
            if getattr(getattr(table, "device", None), "type", "cpu") != "cpu":
                raise ValueError("Page tables must be host scheduler metadata")
            table = table.tolist() if hasattr(table, "tolist") else table
            if len(table) != len(lengths):
                raise ValueError("Every table must include all fixed slots")
            for slot, (row, length) in enumerate(zip(table, lengths, strict=True)):
                pages = (length + 31) // 32
                if pages > len(row):
                    raise ValueError("Page table does not cover the valid token length")
                for value in row[:pages]:
                    block = _integer(value, "global block ID")
                    if not 0 <= block < spec.shape[0]:
                        raise ValueError("Global block ID is outside the allocated pool")
                    key = (spec.tensor_idx, block)
                    if key in seen:
                        raise ValueError(f"Overlapping live KV page at attention layer {layer}, slot {slot}")
                    seen.add(key)
        return True
