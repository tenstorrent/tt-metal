# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PROTOTYPE planner for the sparse KVPE push (high_bw_all_gather(push_runs_tensor=...)).

Instead of gathering the whole populated prefix onto every chip, each owner chip pushes only the KVPE rows the
requester's top-k indices reference, into the requester's gather buffer laid out compactly; sparse_sdpa then
reads that buffer through indices rewritten to compact slots (no block-cyclic remap).

The plan is built on device by default (ttnn.experimental.sparse_push_plan, then an all-gather of its tables; no host
round trip, trace-safe). TT_SPARSE_KV_PUSH_PLAN=host builds it on the host from the indices instead (eager only).
One plan serves every layer that reuses the same indices (GLM-5.3 shared layers), since only the KV slot differs
between them. The indexer writes every chunk's indices into the same buffer, so a layer that computes new indices
must call invalidate().
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from loguru import logger

import ttnn

SENTINEL = 0xFFFFFFFF
_STREAM_HEADER_WORDS = 16  # stream payload starts at byte 64


@dataclass
class SparseKvPushPlan:
    streams: ttnn.Tensor  # host plan: per-device [1, 1, links, words] push lists; device plan: all-gathered tables
    compact_indices: ttnn.Tensor  # indices rewritten to compact slots, same distribution as `indices`
    push_kwargs: dict  # extra high_bw_all_gather arguments selecting the push variant
    split_prefix: bool = False  # built with the chunk start: push_part 1 / 2 split the push around it


def _shard_over_mesh(tensor, mesh_device):
    """Declare `tensor` sharded on dim 2 over every chip in row-major order (what the full-mesh gather expects)."""
    dist = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
    tensor.update_tensor_topology(ttnn.TensorTopology(dist, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords))


def _owner_local(n, stripe_chunk, num_devices):
    """Owner chip and local cache row of natural row n in the TP-deduped block-cyclic KVPE cache."""
    blk = n // stripe_chunk
    return blk % num_devices, (blk // num_devices) * stripe_chunk + n % stripe_chunk


def build_plan_host(indices, num_devices, stripe_chunk, links, pages_per_packet, buffer_rows, banks):
    """indices: [num_devices, rows, topk] natural positions (SENTINEL = masked), chip-major.

    Returns (streams [num_devices, 1, links, words] int32, compact [num_devices, rows, topk] uint32-as-int64,
    the most compact-buffer rows any chip uses).
    Owner o's rows for requester r sit in r's buffer at base(o, r) in local-row order; packet j of that stream
    starts at base + (j % banks) + (j // banks) * P * banks with its P rows `banks` pages apart, matching
    kernels/sparse_push_reader.cpp.
    """
    unit = pages_per_packet * banks
    streams = [[[] for _ in range(links)] for _ in range(num_devices)]
    load = np.zeros((num_devices, links), dtype=np.int64)
    compact = np.full(indices.shape, SENTINEL, dtype=np.int64)
    used = 0
    for dest in range(num_devices):
        idx = indices[dest]
        valid = idx != SENTINEL
        picked = idx[valid]
        rows = np.flatnonzero(np.bincount(picked))  # sorted distinct rows
        owner, local = _owner_local(rows, stripe_chunk, num_devices)
        order = np.lexsort((local, owner))  # owner-major, local-row order within an owner
        counts = np.bincount(owner, minlength=num_devices)
        bases = np.concatenate(([0], np.cumsum(-(-counts // unit) * unit)))
        assert bases[-1] <= buffer_rows, f"sparse push: compact buffer overflow ({bases[-1]} > {buffer_rows} rows)"
        used = max(used, int(bases[-1]))
        starts = np.concatenate(([0], np.cumsum(counts)))
        k = np.arange(len(rows)) - np.repeat(starts[:-1], counts)
        j, i = k // pages_per_packet, k % pages_per_packet
        slot_by_row = np.empty(len(rows), dtype=np.int64)
        slot_by_row[order] = np.repeat(bases[:-1], counts) + (j % banks) + (j // banks) * unit + i * banks
        lut = np.empty(rows[-1] + 1 if len(rows) else 0, dtype=np.int64)
        lut[rows] = slot_by_row
        compact[dest][valid] = lut[picked]
        sorted_local = local[order]
        for o in np.nonzero(counts)[0]:
            w = int(np.argmin(load[o]))
            streams[o][w].append((dest, int(bases[o]), sorted_local[starts[o] : starts[o + 1]]))
            load[o, w] += counts[o]
    words = _STREAM_HEADER_WORDS + max(
        sum(3 + len(l) for _, _, l in streams[o][w]) for o in range(num_devices) for w in range(links)
    )
    words = max(1024, 1 << (words - 1).bit_length())  # few distinct page sizes, so few kernel recompiles
    host = np.zeros((num_devices, 1, links, words), dtype=np.int64)
    for o in range(num_devices):
        for w in range(links):
            host[o, 0, w, 0] = len(streams[o][w])
            pos = _STREAM_HEADER_WORDS
            for dest, base, local in streams[o][w]:
                host[o, 0, w, pos : pos + 3] = (dest, base, len(local))
                host[o, 0, w, pos + 3 : pos + 3 + len(local)] = local
                pos += 3 + len(local)
    return host.astype(np.int32), compact, used


class SparseKvPushPlanner:
    """Builds and caches the push plan for one set of injected indices; shared by all MLA layers of a mesh."""

    def __init__(self, mesh_device, links: int):
        self.mesh_device = mesh_device
        self.links = links
        self.banks = mesh_device.dram_grid_size().x
        self._last: SparseKvPushPlan | None = None
        self._key = None
        self.host_seconds = 0.0  # cumulative host planning time
        # Perf A/B only: when `memo` is a dict, plans are kept per (memo_tag, generation) so re-running the same chunk
        # reuses them without reading the indices back (stand-in for on-device planning).
        self.memo = None
        self.memo_tag = None
        self.generation = 0
        self.on_device = os.getenv("TT_SPARSE_KV_PUSH_PLAN", "device") != "host"
        self._tables = None  # all-gathered plan tables, allocated once
        self._buffers = {}  # (parity, rows, width, dtype) -> compact buffer; consecutive layers alternate parity
        self._starts = {}  # chunk start -> 1-element device tensor
        # layer index -> plan whose prefix part an earlier layer already pushed into that layer's buffer
        self.prefetched = {}
        self._progress = None

    def progress(self):
        """Per-sender [packets sent, finished, turn] pages written by a fused prefix push, read by the resume."""
        if self._progress is None:
            self._progress = ttnn.from_torch(
                torch.zeros(1, 1, 4 * self.links * 2, 16, dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self._progress

    def push_buffer(self, parity, rows, like):
        """Compact buffer for layers of this parity; the previous layer's buffer stays readable while it fills."""
        key = (parity, rows, like.shape[-1], like.dtype)
        if key not in self._buffers:
            self._buffers[key] = ttnn.from_torch(
                torch.zeros(1, 1, rows, like.shape[-1]),
                dtype=like.dtype,
                layout=like.layout,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self._buffers[key]

    def chunk_start(self, start: int):
        if start not in self._starts:
            self._starts[start] = ttnn.from_torch(
                torch.tensor([[[[start]]]], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        return self._starts[start]

    def invalidate(self):
        self._last = None
        self.generation += 1
        self.prefetched.clear()

    def buffer_rows(self, rows: int, page_bytes: int) -> int:
        """Compact buffer rows for up to `rows` distinct rows: each owner's stream pads to whole bank-lane units."""
        unit = self._pages_per_packet(page_bytes) * self.banks
        return rows + self.mesh_device.get_num_devices() * unit

    @staticmethod
    def _pages_per_packet(page_bytes: int) -> int:
        return max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)

    def plan(
        self,
        indices: ttnn.Tensor,
        *,
        stripe_chunk: int,
        buffer_rows: int,
        page_bytes: int,
        max_rows: int,
        chunk_start: int | None = None,
    ) -> SparseKvPushPlan:
        key = (indices.buffer_address(), stripe_chunk, buffer_rows, page_bytes, chunk_start)
        if self._last is not None and self._key == key:
            return self._last
        if self.on_device:
            start = None if chunk_start is None else self.chunk_start(chunk_start)
            self._key, self._last = key, self._plan_device(indices, stripe_chunk, page_bytes, max_rows, start)
            return self._last
        memo_key = (self.memo_tag, self.generation, key[1:])
        if self.memo is not None and memo_key in self.memo:
            self._key, self._last = key, self.memo[memo_key]
            return self._last
        t0 = time.perf_counter()
        # Indices are sequence-sharded over every chip in row-major order, so chip c holds rows c*rows..
        host_idx = ttnn.to_torch(indices, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))
        host_idx = host_idx.to(torch.int64).numpy().astype(np.int64) & 0xFFFFFFFF
        n_dev = self.mesh_device.get_num_devices()
        pages_per_packet = self._pages_per_packet(page_bytes)
        rows = host_idx.shape[2] // n_dev
        per_chip = host_idx.reshape(n_dev, rows, host_idx.shape[-1])
        streams, compact, used = build_plan_host(
            per_chip, n_dev, stripe_chunk, self.links, pages_per_packet, buffer_rows, self.banks
        )
        common = dict(
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        streams_tt = ttnn.from_torch(
            torch.from_numpy(streams), mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0), **common
        )
        compact_tt = ttnn.from_torch(
            torch.from_numpy(compact.reshape(host_idx.shape).astype(np.uint32).view(np.int32)),
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=2),
            **common,
        )
        self._key = key
        self._last = SparseKvPushPlan(streams=streams_tt, compact_indices=compact_tt, push_kwargs={})
        if self.memo is not None:
            self.memo[memo_key] = self._last
        dt = time.perf_counter() - t0
        self.host_seconds += dt
        logger.info(f"sparse KV push plan: {dt:.3f}s on host, up to {used} buffer rows per chip")
        return self._last

    def _plan_device(self, indices, stripe_chunk, page_bytes, max_rows, chunk_start=None):
        n_dev = self.mesh_device.get_num_devices()
        table, compact = ttnn.experimental.sparse_push_plan(
            indices,
            num_devices=n_dev,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            pages_per_packet=self._pages_per_packet(page_bytes),
            banks=self.banks,
            chunk_start=chunk_start,
        )
        _shard_over_mesh(table, self.mesh_device)
        rows = table.shape[2] * n_dev
        if self._tables is None or self._tables.shape[2] != rows:
            self._tables = ttnn.from_torch(
                torch.zeros(1, 1, rows, table.shape[3], dtype=torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=self._tables, cluster_axis=None, num_links=self.links
        )
        ttnn.deallocate(table)
        return SparseKvPushPlan(
            streams=self._tables,
            compact_indices=compact,
            push_kwargs=dict(push_stripe_chunk=stripe_chunk, push_max_rows=max_rows),
            split_prefix=chunk_start is not None,
        )
