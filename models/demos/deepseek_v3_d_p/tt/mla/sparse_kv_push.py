# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Planner for the sparse KVPE push of GLM shared layers.

Instead of gathering the whole populated prefix onto every chip, each owner chip pushes only the KVPE rows the
requester's top-k indices reference, into the requester's compact buffer; sparse_sdpa then reads that buffer through
indices rewritten to compact slots (no block-cyclic remap).

The plan is built on device (ttnn.experimental.sparse_push_plan, then an all-gather of its tables), so it needs no host
round trip and is trace-safe. One plan serves every layer that reuses the same indices (GLM shared layers), since only
the KV slot differs between them. The indexer writes every chunk's indices into the same buffer, so a layer that
computes new indices must call invalidate().
"""

from dataclasses import dataclass

import torch

import ttnn


@dataclass
class SparseKvPushPlan:
    table: ttnn.Tensor  # every chip's plan table, all-gathered
    compact_indices: ttnn.Tensor  # indices rewritten to compact slots, same distribution as the top-k indices
    stripe_chunk: int
    max_rows: int


def sparse_kv_push_supported(mesh_device) -> bool:
    """The push routes over all four torus links of every chip: FABRIC_2D_TORUS_XY with both mesh sides >= 3."""
    rows, cols = tuple(mesh_device.shape)
    return ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_2D_TORUS_XY and min(rows, cols) >= 3


def _shard_over_mesh(tensor, mesh_device):
    """Declare `tensor` sharded on dim 2 over every chip in row-major order (what the full-mesh gather expects)."""
    dist = ttnn.MeshShape(mesh_device.shape[0], mesh_device.shape[1])
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(dist)]
    tensor.update_tensor_topology(ttnn.TensorTopology(dist, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords))


def _replicated(mesh_device, host, dtype, layout):
    return ttnn.from_torch(
        host,
        dtype=dtype,
        layout=layout,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


class SparseKvPushPlanner:
    """Builds and caches the push plan for one set of injected indices; shared by all MLA layers of a mesh.

    Every buffer is allocated on first use, which the warm pass does before a trace capture."""

    def __init__(self, mesh_device, links: int):
        self.mesh_device = mesh_device
        self.links = links
        self.banks = mesh_device.dram_grid_size().x
        self._last: SparseKvPushPlan | None = None
        self._key = None
        self._tables = None  # all-gathered plan tables, allocated once
        self._buffers = {}  # (parity, rows, width, dtype) -> compact buffer; consecutive layers alternate parity
        self._starts = {}  # chunk start -> 1-element device tensor (eager callers without metadata)
        # layer index -> plan whose prefix part an earlier layer already pushed into that layer's buffer
        self.prefetched = {}

    def push_buffer(self, parity, rows, like):
        """Compact buffer for layers of this parity; the previous layer's buffer stays readable while it fills."""
        key = (parity, rows, like.shape[-1], like.dtype)
        if key not in self._buffers:
            self._buffers[key] = _replicated(
                self.mesh_device, torch.zeros(1, 1, rows, like.shape[-1]), like.dtype, like.layout
            )
        return self._buffers[key]

    def chunk_start(self, start: int):
        if start not in self._starts:
            self._starts[start] = _replicated(
                self.mesh_device, torch.tensor([[[[start]]]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
        return self._starts[start]

    def invalidate(self):
        self._last = None
        self.prefetched.clear()

    def buffer_rows(self, rows: int, page_bytes: int) -> int:
        """Compact buffer rows for up to `rows` distinct rows: each owner's run pads to whole bank-lane units."""
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
        page_bytes: int,
        max_rows: int,
        chunk_start: int = 0,
        chunk_start_tensor: ttnn.Tensor | None = None,
    ) -> SparseKvPushPlan:
        """The plan for `indices`, cached on the indices buffer. Trace-safe with chunk_start_tensor (the metadata's
        kv_actual_global): the plan op reads the chunk start on device, so the cache key leaves its value out. Every
        layer that computes its own indices invalidate()s, so the key only has to tell calls within one forward apart.
        """
        start = chunk_start_tensor if chunk_start_tensor is not None else self.chunk_start(chunk_start)
        key = (indices.buffer_address(), stripe_chunk, page_bytes, start.buffer_address())
        if self._last is None or self._key != key:
            self._key, self._last = key, self._build(indices, stripe_chunk, page_bytes, max_rows, start)
        return self._last

    def _build(self, indices, stripe_chunk, page_bytes, max_rows, chunk_start):
        table, compact = ttnn.experimental.sparse_push_plan(
            indices,
            num_devices=self.mesh_device.get_num_devices(),
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            pages_per_packet=self._pages_per_packet(page_bytes),
            banks=self.banks,
            chunk_start=chunk_start,
        )
        _shard_over_mesh(table, self.mesh_device)
        rows = table.shape[2] * self.mesh_device.get_num_devices()
        if self._tables is None or self._tables.shape[2] != rows:
            self._tables = _replicated(
                self.mesh_device,
                torch.zeros(1, 1, rows, table.shape[3], dtype=torch.int32),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
        ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=self._tables, cluster_axis=None, num_links=self.links
        )
        ttnn.deallocate(table)
        return SparseKvPushPlan(
            table=self._tables, compact_indices=compact, stripe_chunk=stripe_chunk, max_rows=max_rows
        )
