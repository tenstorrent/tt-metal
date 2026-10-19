# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PROTOTYPE host planner for the sparse KVPE push (high_bw_all_gather(push_runs_tensor=...)).

Instead of gathering the whole populated prefix onto every chip, each owner chip pushes only the KVPE rows the
requester's top-k indices reference, into the requester's gather buffer laid out compactly; sparse_sdpa then
reads that buffer through indices rewritten to compact slots (no block-cyclic remap).

This v1 builds the plan on the host from the indices (eager only, not trace-safe). One plan serves every layer
that reuses the same indices (GLM-5.3 shared layers), since only the KV slot differs between them.
"""

from dataclasses import dataclass

import numpy as np
import torch

import ttnn

SENTINEL = 0xFFFFFFFF
_STREAM_HEADER_WORDS = 16  # stream payload starts at byte 64


@dataclass
class SparseKvPushPlan:
    indices: ttnn.Tensor  # the injected indices this plan was built from (kept so the cache key stays valid)
    streams: ttnn.Tensor  # per-device [1, 1, links, words] uint32 push lists
    compact_indices: ttnn.Tensor  # indices rewritten to compact slots, same distribution as `indices`


def _owner_local(n, stripe_chunk, num_devices):
    """Owner chip and local cache row of natural row n in the TP-deduped block-cyclic KVPE cache."""
    blk = n // stripe_chunk
    return blk % num_devices, (blk // num_devices) * stripe_chunk + n % stripe_chunk


def build_plan_host(indices, num_devices, stripe_chunk, links, pages_per_packet, buffer_rows, banks):
    """indices: [num_devices, rows, topk] natural positions (SENTINEL = masked), chip-major.

    Returns (streams [num_devices, 1, links, words] int32, compact [num_devices, rows, topk] uint32-as-int64).
    Owner o's rows for requester r sit in r's buffer at base(o, r) in local-row order; packet j of that stream
    starts at base + (j % banks) + (j // banks) * P * banks with its P rows `banks` pages apart, matching
    kernels/sparse_push_reader.cpp.
    """
    unit = pages_per_packet * banks
    streams = [[[] for _ in range(links)] for _ in range(num_devices)]
    load = np.zeros((num_devices, links), dtype=np.int64)
    compact = np.full(indices.shape, SENTINEL, dtype=np.int64)
    for dest in range(num_devices):
        idx = indices[dest]
        valid = idx != SENTINEL
        rows = np.unique(idx[valid])
        owner, local = _owner_local(rows, stripe_chunk, num_devices)
        slot_of = {}
        base = 0
        for o in range(num_devices):
            sel = owner == o
            o_rows, o_local = rows[sel], local[sel]
            order = np.argsort(o_local)
            o_rows, o_local = o_rows[order], o_local[order]
            k = np.arange(len(o_local))
            j, i = k // pages_per_packet, k % pages_per_packet
            slots = base + (j % banks) + (j // banks) * unit + i * banks
            slot_of.update(zip(o_rows.tolist(), slots.tolist()))
            if len(o_local):
                w = int(np.argmin(load[o]))
                streams[o][w].append((dest, base, o_local))
                load[o, w] += len(o_local)
            base += -(-len(o_local) // unit) * unit
        assert base <= buffer_rows, f"sparse push: compact buffer overflow ({base} > {buffer_rows} rows)"
        lut = np.vectorize(slot_of.__getitem__, otypes=[np.int64])
        compact[dest][valid] = lut(idx[valid])
    words = _STREAM_HEADER_WORDS + max(
        sum(3 + len(l) for _, _, l in streams[o][w]) for o in range(num_devices) for w in range(links)
    )
    words += -words % 16
    host = np.zeros((num_devices, 1, links, words), dtype=np.int64)
    for o in range(num_devices):
        for w in range(links):
            host[o, 0, w, 0] = len(streams[o][w])
            pos = _STREAM_HEADER_WORDS
            for dest, base, local in streams[o][w]:
                host[o, 0, w, pos : pos + 3] = (dest, base, len(local))
                host[o, 0, w, pos + 3 : pos + 3 + len(local)] = local
                pos += 3 + len(local)
    return host.astype(np.int32), compact


class SparseKvPushPlanner:
    """Builds and caches the push plan for one set of injected indices; shared by all MLA layers of a mesh."""

    def __init__(self, mesh_device, links: int):
        self.mesh_device = mesh_device
        self.links = links
        self.banks = mesh_device.dram_grid_size().x
        self._last: SparseKvPushPlan | None = None

    def buffer_rows(self, rows: int, page_bytes: int) -> int:
        """Compact buffer rows for up to `rows` distinct rows: each owner's stream pads to whole bank-lane units."""
        unit = self._pages_per_packet(page_bytes) * self.banks
        return rows + self.mesh_device.get_num_devices() * unit

    @staticmethod
    def _pages_per_packet(page_bytes: int) -> int:
        return max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes)

    def plan(self, indices: ttnn.Tensor, *, stripe_chunk: int, buffer_rows: int, page_bytes: int) -> SparseKvPushPlan:
        if self._last is not None and self._last.indices is indices:
            return self._last
        n_dev = self.mesh_device.get_num_devices()
        pages_per_packet = self._pages_per_packet(page_bytes)
        # Indices are sequence-sharded over every chip in row-major order, so chip c holds rows c*rows..
        host_idx = ttnn.to_torch(indices, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=2))
        host_idx = host_idx.to(torch.int64).numpy().astype(np.int64) & 0xFFFFFFFF
        rows = host_idx.shape[2] // n_dev
        per_chip = host_idx.reshape(n_dev, rows, host_idx.shape[-1])
        streams, compact = build_plan_host(
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
        self._last = SparseKvPushPlan(indices=indices, streams=streams_tt, compact_indices=compact_tt)
        return self._last
