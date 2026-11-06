# SPDX-License-Identifier: Apache-2.0
"""Opt-in model-owned checkpoints with native APC disabled.

Experimental: must pass coordinator's raw-copy, model exactness and cost gates.
"""

import json

import torch

import ttnn

from .private_cache_copy import CopyWorkspace, copy_blocks, page_geometry
from .private_prefix_index import CheckpointIndex, bank_reserved_bytes


class PrivatePrefixCache:
    def __init__(self, adapter, byte_limit, node_limit=128, free_reserve=4 << 30):
        self.adapter, self.gen, self.mesh = adapter, adapter.gen, adapter.mesh
        if self.gen.counters["captures"]:
            raise RuntimeError("Private-cache arenas must be prepared before the first trace")
        namespace = json.dumps(
            dict(
                precision=adapter.model.precision_config,
                layers=adapter.model.layer_indices,
                batch=adapter.batch,
                context=adapter.context,
                pack_terminal=self.gen.pack_prefill_terminal,
                checkpoint=str(adapter.checkpoint),
                tensor_specs=[
                    [
                        (name, list(value.shape), list(value.padded_shape), str(value.dtype), str(value.layout))
                        for name in ("conv", "recurrent", "key", "value")
                        if (value := getattr(state, name, None)) is not None
                    ]
                    for state in self.gen.kv_cache
                ],
                physical_chunk=4096,
                policy="canonical-4k-exact-scheduled-end-arena-v2",
            ),
            sort_keys=True,
        )
        self.index = CheckpointIndex(namespace, byte_limit, node_limit)
        self.free_reserve = free_reserve
        self.banks = ttnn.get_memory_view(self.mesh, ttnn.BufferType.DRAM).num_banks
        self.alignment = ttnn.get_dram_alignment()
        self.mapping_row_bytes = max(32, self.alignment)
        self.scratch_limit = 64 << 20
        self.pending_scratch = 0
        self.pending = []
        self.touched = False
        self.hits = self.misses = self.insertions = self.declined = 0
        self.declined_keys = {}
        self.lost_hits = {}
        self.decline_tracking_overflow = 0
        self._prepare_arenas(byte_limit, node_limit)

    def _prepare_arenas(self, byte_limit, node_limit):
        # Setup only: fixed backing tensors and every raw-copy program must
        # exist before Generator.prepare_decode captures its first trace.
        self.state_specs = list(self._state_specs())
        logits = self.gen.prefill_output
        lp, lb = page_geometry(logits)
        if lp % self.adapter.batch:
            raise ValueError("Logits do not have independently addressable rows")
        self.logits_pages = lp // self.adapter.batch
        # _copy uses eight physical pages only for KV blocks. Current model
        # state rows are conv=3, recurrence=192, logits=1940 pages. Fail setup
        # rather than silently selecting a mapping shape not primed below.
        if self.logits_pages == 8 or any(n == 8 for _, _, _, n, _ in self.state_specs):
            raise ValueError("State/logit page geometry collides with KV mapping selector")
        self.kv_specs = [
            (layer, name, value)
            for layer, state in enumerate(self.gen.kv_cache)
            for name in ("key", "value")
            if (value := getattr(state, name, None)) is not None
        ]
        self.mapping_reserved = sum(self._reserved(n, self.mapping_row_bytes) for n in (1, 8192))

        def arena_bytes(nodes):
            return (
                sum(self._reserved(nodes * n, size) for _, _, _, n, size in self.state_specs)
                + self._reserved(nodes * self.logits_pages, lb)
                + sum(self._reserved(nodes * 128 * 8, page_geometry(v)[1]) for _, _, v in self.kv_specs)
                + self.mapping_reserved
            )

        self.node_capacity = next((n for n in range(node_limit, 0, -1) if arena_bytes(n) <= byte_limit), 0)
        if not self.node_capacity:
            raise ValueError("Private-cache cap cannot fit one prepared 4K checkpoint")
        self.kv_capacity = self.node_capacity * 128
        self.arena_reserved = arena_bytes(self.node_capacity)
        before = self.adapter.memory()["free_bytes_per_device"]
        if before < self.arena_reserved + self.free_reserve:
            raise RuntimeError("Insufficient memory for fixed private-cache arenas plus reserve")
        self.state_arenas = {
            (layer, name): self._allocate(source, [self.node_capacity, *tuple(source.shape)[1:]])
            for layer, name, source, _, _ in self.state_specs
        }
        self.kv_arenas = {
            (layer, name): self._allocate(source, [self.kv_capacity, *tuple(source.shape)[1:]])
            for layer, name, source in self.kv_specs
        }
        self.logits_arena = self._allocate(logits, [self.node_capacity, *tuple(logits.shape)[1:]])
        self.workspace = CopyWorkspace(self.mesh)
        after_storage = self.adapter.memory()["free_bytes_per_device"]
        if before - after_storage != self.arena_reserved:
            raise RuntimeError("Actual arena reservation differs from bank ledger")
        programs_before = self.mesh.num_program_cache_entries()
        for layer, name, source, n, _ in self.state_specs:
            self._prime(source, self.state_arenas[layer, name], n, False)
        for layer, name, source in self.kv_specs:
            self._prime(source, self.kv_arenas[layer, name], 8, True)
        self._prime(logits, self.logits_arena, self.logits_pages, False)
        ttnn.synchronize_device(self.mesh)
        self.pending.clear()
        after_prime = self.adapter.memory()["free_bytes_per_device"]
        if after_prime < self.free_reserve:
            raise RuntimeError("Transport preparation consumed the required free reserve")
        self.node_cursor = self.kv_cursor = 0
        self.adapter.event(
            "private_prefix_arenas",
            max_bytes=byte_limit,
            storage_reserved_bytes=self.arena_reserved,
            actual_storage_reserved_bytes=before - after_storage,
            prepared_program_bytes=after_storage - after_prime,
            new_programs=self.mesh.num_program_cache_entries() - programs_before,
            node_capacity=self.node_capacity,
            kv_blocks=self.kv_capacity,
            mapping_reserved_bytes=self.mapping_reserved,
            memory=self.adapter.memory(),
        )

    def _prime(self, live, arena, pages_per_block, large):
        for source, destination in ((live, arena), (arena, live)):
            copy_blocks(
                source,
                destination,
                [(0, 0)],
                pages_per_block=pages_per_block,
                keepalive=self.pending,
                workspace=self.workspace,
                large_mapping=large,
                prime=True,
            )

    @staticmethod
    def token_bytes(row, end):
        # Native adapter tokens are CPU int32; canonical int64 bytes avoid mutable
        # storage aliasing and preserve exact token identity.
        if row.device.type != "cpu" or row.dtype not in (torch.int32, torch.int64):
            raise ValueError("Private-cache key requires CPU integer token IDs")
        if end < 0 or row.numel() < end:
            raise ValueError("Token buffer does not cover requested checkpoint")
        return row[:end].detach().to(torch.int64).contiguous().numpy().tobytes()

    def lookup(self, tokens, end):
        key = self.token_bytes(tokens, end)
        node = self.index.lookup(key)
        if node is None:
            self.misses += 1
            reason = self.declined_keys.get(self.index._key(key))
            if reason is not None:
                self.lost_hits[reason] = self.lost_hits.get(reason, 0) + 1
                self.adapter.event(
                    "private_prefix_lost_hit", end=end, reason=reason, basis="previous_declined_prefix_hash_match"
                )
        else:
            self.hits += 1
        return node

    def _allocate(self, source, shape):
        return ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), source.dtype, source.layout, self.mesh, ttnn.DRAM_MEMORY_CONFIG
        )

    def _reserved(self, pages, page_bytes):
        return bank_reserved_bytes(pages, page_bytes, self.banks, self.alignment)

    def _begin_transfer(self, scratch):
        if scratch > self.scratch_limit:
            raise ValueError("Checkpoint mapping exceeds bounded scratch")
        if self.pending_scratch + scratch > self.scratch_limit:
            self.finish()
        # Scratch now means bounded HOST map buffers. Device maps are fixed
        # arena-owned tensors; no request-time allocations are permitted.

    def _copy(self, source, destination, pairs, pages_per_block):
        self._begin_transfer((8192 if pages_per_block == 8 else 1) * self.mapping_row_bytes)
        copy_blocks(
            source,
            destination,
            pairs,
            pages_per_block=pages_per_block,
            keepalive=self.pending,
            workspace=self.workspace,
            large_mapping=(pages_per_block == 8),
        )
        self.touched = True
        self.pending_scratch += (8192 if pages_per_block == 8 else 1) * self.mapping_row_bytes

    def _state_specs(self):
        for layer, state in enumerate(self.gen.kv_cache):
            for name in ("conv", "recurrent"):
                tensor = getattr(state, name, None)
                if tensor is not None:
                    pages, size = page_geometry(tensor)
                    if pages % self.adapter.batch:
                        raise ValueError("State is not independently addressable by outer row")
                    yield layer, name, tensor, pages // self.adapter.batch, size

    def capture(self, tokens, end, slot, pages):
        key = self.token_bytes(tokens, end)
        if not end or end % self.index.alignment or self.index.lookup(key) is not None:
            return
        parent = self.index.parent(key)
        first = parent.end if parent is not None else 0
        count = (end - first) // 32
        if pages.numel() < end // 32:
            raise ValueError("Page map does not cover checkpoint")
        specs = self.state_specs
        logits = self.gen.prefill_output
        logits_pages = self.logits_pages
        needed = (
            sum(n * size for _, _, _, n, size in specs)
            + logits_pages * page_geometry(logits)[1]
            + sum(count * 8 * page_geometry(v)[1] for _, _, v in self.kv_specs)
        )
        reason = None
        if self.node_cursor >= self.node_capacity:
            reason = "state_arena_full"
        elif self.kv_cursor + count > self.kv_capacity:
            reason = "kv_arena_full"
        elif count > 8192:
            reason = "mapping_capacity"
        elif not self.index.can_admit(key, needed):
            reason = "index_limit"
        if reason is not None:
            self.declined += 1
            if len(self.declined_keys) < 1024 or self.index._key(key) in self.declined_keys:
                self.declined_keys[self.index._key(key)] = reason
            else:
                self.decline_tracking_overflow += 1
            self.adapter.event(
                "private_prefix_decline",
                slot=slot,
                end=end,
                reason=reason,
                required_used_bytes=needed,
                arena_reserved_bytes=self.arena_reserved,
                node_cursor=self.node_cursor,
                kv_cursor=self.kv_cursor,
            )
            return
        row, base = self.node_cursor, self.kv_cursor
        payload = dict(states=[], kv=[], logits=None, first=first)
        for layer, name, source, n, _ in specs:
            target = self.state_arenas[layer, name]
            self._copy(source, target, [(slot, row)], n)
            payload["states"].append((layer, name, target, n, row))
        source_pages = [int(p) for p in pages[first // 32 : end // 32].tolist()]
        for layer, name, source in self.kv_specs:
            target = self.kv_arenas[layer, name]
            self._copy(source, target, [(page, base + i) for i, page in enumerate(source_pages)], 8)
            payload["kv"].append((layer, name, target, base))
        self._copy(logits, self.logits_arena, [(slot, row)], logits_pages)
        payload["logits"] = self.logits_arena, logits_pages, row
        node = self.index.insert(key, payload, needed, parent)
        if node is None:
            raise RuntimeError("Admission changed during serialized snapshot insertion")
        self.node_cursor += 1
        self.kv_cursor += count
        self.insertions += 1
        self.adapter.event(
            "private_prefix_insert",
            slot=slot,
            end=end,
            parent_end=first,
            owned_bytes=self.index.owned_bytes,
            entries=len(self.index.nodes),
            memory=self.adapter.memory(),
        )

    def restore(self, node, slot, pages, valid_start=0):
        if pages.numel() < node.end // 32:
            raise ValueError("Page map does not cover restored checkpoint")
        if not 0 <= valid_start <= node.end or valid_start % 32:
            raise ValueError("Invalid proven prefix boundary")
        chain = []
        current = node
        while current is not None:
            chain.append(current)
            current = current.parent
        for entry in reversed(chain):
            first = max(valid_start, entry.payload["first"])
            if first >= entry.end:
                continue
            destination_pages = [int(p) for p in pages[first // 32 : entry.end // 32].tolist()]
            offset = (first - entry.payload["first"]) // 32
            for layer, name, source, base in entry.payload["kv"]:
                self._copy(
                    source,
                    getattr(self.gen.kv_cache[layer], name),
                    [(base + offset + i, page) for i, page in enumerate(destination_pages)],
                    8,
                )
        for layer, name, source, n, row in node.payload["states"]:
            self._copy(source, getattr(self.gen.kv_cache[layer], name), [(row, slot)], n)
        source, n, row = node.payload["logits"]
        self._copy(source, self.gen.prefill_output, [(row, slot)], n)
        self.gen._resident_inputs_valid = False
        self.adapter.event(
            "private_prefix_hit",
            slot=slot,
            end=node.end,
            valid_start=valid_start,
            copied_kv_tokens=node.end - valid_start,
            owned_bytes=self.index.owned_bytes,
        )
        return self.gen.prefill_output

    def finish(self):
        # Intentionally conservative first prototype. This fence is part of the
        # measured hit/insertion cost, including intermediate prefill callbacks.
        fenced = self.touched
        if fenced:
            ttnn.synchronize_device(self.mesh)
            self.pending.clear()
            self.workspace.inflight.clear()
            self.pending_scratch = 0
            self.touched = False
        self.adapter.event(
            "private_prefix_stats",
            hits=self.hits,
            misses=self.misses,
            insertions=self.insertions,
            declined=self.declined,
            reserved_bytes=self.arena_reserved,
            used_payload_bytes=self.index.owned_bytes,
            nodes=len(self.index.nodes),
            scratch_limit=self.scratch_limit,
            mapping_reuse_fences=self.workspace.reuse_fences,
            lost_hits_by_reason=self.lost_hits,
            decline_tracking_overflow=self.decline_tracking_overflow,
            banks=self.banks,
            alignment=self.alignment,
            fenced=fenced,
            memory=self.adapter.memory(),
        )
        return fenced

    def clear(self):
        self.finish()
        ttnn.synchronize_device(self.mesh)
        self.index.clear_after_device_fence()
        self.declined_keys.clear()
        self.node_cursor = self.kv_cursor = 0
