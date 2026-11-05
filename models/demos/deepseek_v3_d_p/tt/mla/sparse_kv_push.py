# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse KVPE push for GLM shared layers.

Shared layers reuse the previous full layer's top-k, so instead of gathering the whole prefix onto every chip, each
owner chip pushes only the rows a requester's indices reference into that requester's compact buffer. sparse_sdpa then
reads the buffer through indices rewritten to compact slots. The plan is built on device, so it is trace-safe.
"""

from dataclasses import dataclass

import torch

import ttnn
from ttnn.core import divup

# Which of a requester's needed rows a push sends: ALL, PREFIX (rows earlier chunks wrote) or REST.
SparseKvPushPart = ttnn.experimental.SparseKvPushPart

# Cores the push needs per fabric link, for laying out its grid beside sparse_sdpa.
PUSH_CORES_PER_LINK = ttnn.experimental.SPARSE_KV_PUSH_CORES_PER_LINK


@dataclass
class SparseKvPushPlan:
    table: ttnn.Tensor  # every chip's plan table, all-gathered
    compact_indices: ttnn.Tensor
    stripe_chunk: int
    max_rows: int
    # Chunk starts below this take dense mode: compact_indices address a dense gather into the compact buffer and
    # the push sends nothing (0: off).
    dense_below: int = 0


def sparse_kv_push_supported(mesh_device, stripe_chunk: int, max_rows: int) -> bool:
    """Whether the push can serve a cache of `max_rows` rows in blocks of `stripe_chunk` rows on this mesh (the ops'
    own check); otherwise shared layers gather the whole prefix."""
    return ttnn.experimental.is_sparse_kv_push_supported(mesh_device, stripe_chunk, max_rows)


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
    """Plans, buffers, semaphores and sub-device managers of the push, shared by every MLA layer of a mesh.

    Trace-safe: on the metadata path (PREFILL_USE_TRACE=1) nothing depends on a host value that changes per chunk or
    request. The plan reads the chunk start from metadata[1] on device and it is not part of the plan's cache key (a
    layer that computes its own indices calls invalidate()); the push reads the cache slot from metadata[0]. Buffers,
    semaphores and sub-device managers are created on first use, which the warm pass does before any trace capture."""

    def __init__(self, mesh_device, links: int):
        self.mesh_device = mesh_device
        self.links = links
        grid = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        l1_small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank > 0
        buffer_type = ttnn.BufferType.L1_SMALL if l1_small else ttnn.BufferType.L1
        self.ready_semaphore, self.done_semaphore = (
            ttnn.create_global_semaphore(mesh_device, cores, 0, buffer_type) for _ in range(2)
        )
        ttnn.synchronize_device(mesh_device)
        self._plan: SparseKvPushPlan | None = None
        self._plan_key = None
        self._tables = None
        self._buffers = {}
        self._managers = {}  # query rows -> sub-device manager of grids(query rows)
        # layer index -> plan whose prefix rows the previous layer already pushed into its buffer, beside its attention
        self.prefetched = {}

    def grids(self, query_rows: int):
        """(attention, push) core grids for sparse_sdpa beside the next layer's prefix push.

        Cores past the fewest that keep attention's busiest core at its token count only add DRAM contention with the
        push, so attention takes that many whole columns. The push sits in the remaining columns' rows furthest from
        the Ethernet row, where it contends least with attention."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        push_cores = self.links * PUSH_CORES_PER_LINK
        # Attention's busiest core if it took every core the push leaves free...
        tokens_per_core = divup(query_rows, grid.x * grid.y - push_cores)
        # ...and the fewest whole columns that keep that load (GLM: 160 rows, 2 per core, 80 cores = 8 columns), but no
        # more than leave the push its cores (e.g. 96 rows would take 10 columns, the push needs 3 of 12).
        push_cols_needed = divup(push_cores, grid.y)
        attention_cols = min(divup(divup(query_rows, tokens_per_core), grid.y), grid.x - push_cols_needed)
        if attention_cols <= 0:
            raise ValueError(f"no room for {push_cores} push cores beside attention on a {grid.x}x{grid.y} grid")
        # The push takes the bottom rows of the remaining columns.
        push_cols = grid.x - attention_cols
        push_rows = divup(push_cores, push_cols)
        attention = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(attention_cols - 1, grid.y - 1))]
        )
        push = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(attention_cols, grid.y - push_rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
        )
        return attention, push

    def prefetch_manager(self, query_rows: int):
        """Sub-device manager that runs attention (sub-device 0, whose cores sparse_sdpa takes by default) beside the
        next layer's prefix push (sub-device 1), on grids(query_rows). Created at init, outside any trace capture."""
        if query_rows not in self._managers:
            attention, push = self.grids(query_rows)
            self._managers[query_rows] = self.mesh_device.create_sub_device_manager(
                [ttnn.SubDevice([attention]), ttnn.SubDevice([push])], 0
            )
        return self._managers[query_rows]

    def release(self):
        """Unregister the sub-device managers before the mesh closes."""
        if self._managers:
            self.mesh_device.clear_loaded_sub_device_manager()
            for manager in self._managers.values():
                self.mesh_device.remove_sub_device_manager(manager)
            self._managers.clear()

    def buffer_spec(self, like: ttnn.Tensor, *, distinct_rows: int, max_rows: int, stripe_chunk: int):
        """The compact buffer's spec for up to `distinct_rows` pushed rows of `like`'s rows. A chip's distinct rows are
        at most its query rows x top-k, so the buffer is sized from the chunk, not from the cache (about 220 MiB at a
        1M cache instead of the dense gather's 703 MiB)."""
        return ttnn.experimental.sparse_kv_push_output_spec(
            self.mesh_device,
            distinct_rows=distinct_rows,
            max_rows=max_rows,
            stripe_chunk=stripe_chunk,
            row_width=like.shape[-1],
            dtype=like.dtype,
        )

    def push_buffer(
        self, parity: int, like: ttnn.Tensor, *, distinct_rows: int, max_rows: int, stripe_chunk: int
    ) -> ttnn.Tensor:
        """The compact buffer for `parity`, sized as buffer_spec; created by the op once and reused by every chunk."""
        key = (parity, like.shape[-1], like.dtype, distinct_rows, max_rows, stripe_chunk)
        if key not in self._buffers:
            self._buffers[key] = ttnn.experimental.sparse_kv_push_create_output_buffer(
                self.mesh_device,
                distinct_rows=distinct_rows,
                max_rows=max_rows,
                stripe_chunk=stripe_chunk,
                row_width=like.shape[-1],
                dtype=like.dtype,
            )
        return self._buffers[key]

    def invalidate(self):
        """Drop the cached plan: new indices arrived in the same buffer (the indexer rewrote it, or the previous
        pipeline stage sent them)."""
        self._plan = None
        self.prefetched.clear()

    def plan(
        self,
        indices: ttnn.Tensor,
        *,
        stripe_chunk: int,
        page_bytes: int,
        max_rows: int,
        chunk_start: int = 0,
        chunk_start_tensor: ttnn.Tensor | None = None,
        dense_below: int = 0,
    ) -> SparseKvPushPlan:
        """The plan for `indices`, cached across the layers that reuse them. On a traced replay the plan op reads the
        chunk start from chunk_start_tensor on device, so one cached plan serves any chunk; eagerly it takes
        chunk_start as a runtime arg. Chunks starting below dense_below take dense mode (see SparseKvPushPlan)."""
        # Keyed by buffer address: full layers rewrite the shared indices buffer in place and call invalidate(), as
        # does a pipeline stage's first layer, so an unchanged address means the same indices.
        start_key = chunk_start_tensor.buffer_address() if chunk_start_tensor is not None else chunk_start
        key = (indices.buffer_address(), stripe_chunk, page_bytes, max_rows, start_key, dense_below)
        if self._plan is None or self._plan_key != key:
            self._plan_key = key
            self._plan = self._build(
                indices, stripe_chunk, page_bytes, max_rows, chunk_start, chunk_start_tensor, dense_below
            )
        return self._plan

    def _build(self, indices, stripe_chunk, page_bytes, max_rows, chunk_start, chunk_start_tensor, dense_below):
        n = self.mesh_device.get_num_devices()
        table, compact_indices = ttnn.experimental.sparse_push_plan(
            indices,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            page_bytes=page_bytes,
            chunk_start=None if chunk_start_tensor is not None else chunk_start,
            chunk_start_tensor=chunk_start_tensor,
            dense_below=dense_below,
        )
        gathered_rows = table.shape[2] * n
        if self._tables is None or self._tables.shape[2] != gathered_rows:
            self._tables = _replicated(
                self.mesh_device,
                torch.zeros(1, 1, gathered_rows, table.shape[3], dtype=torch.int32),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
        ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=self._tables, cluster_axis=None, num_links=self.links
        )
        ttnn.deallocate(table)
        return SparseKvPushPlan(
            table=self._tables,
            compact_indices=compact_indices,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            dense_below=dense_below,
        )
