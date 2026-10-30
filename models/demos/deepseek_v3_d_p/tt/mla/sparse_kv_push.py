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

# Which of a requester's needed rows a push sends: ALL, PREFIX (rows earlier chunks wrote) or REST.
SparseKvPushPart = ttnn.experimental.SparseKvPushPart

# Per link: one channel per torus direction, each with 2 senders and a fabric mux. Must match cores_needed() in
# ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/device/sparse_kv_push_common.hpp.
PUSH_CORES_PER_LINK = 4 * 3

# Must match kMaxStripeChunk and sparse_push_plan::kChunkRows in
# ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/device/kernels/sparse_push_constants.hpp.
MAX_STRIPE_CHUNK = 512
PLAN_CHUNK_ROWS = 512


@dataclass
class SparseKvPushPlan:
    table: ttnn.Tensor  # every chip's plan table, all-gathered
    compact_indices: ttnn.Tensor
    stripe_chunk: int
    max_rows: int


def sparse_kv_push_supported(mesh_device) -> bool:
    """The push sends over all four torus links of every chip."""
    rows, cols = tuple(mesh_device.shape)
    return ttnn.get_fabric_config() == ttnn.FabricConfig.FABRIC_2D_TORUS_XY and min(rows, cols) >= 3


def sparse_kv_push_fits(stripe_chunk: int, max_rows: int) -> bool:
    """Whether the push can serve a cache of `max_rows` rows in blocks of `stripe_chunk` rows: a block's bitmap is
    read as at most 64 B (512 rows, in whole 32-row words), and the plan reduces the bitmap in 512-row chunks."""
    return (
        0 < stripe_chunk <= MAX_STRIPE_CHUNK
        and stripe_chunk % 32 == 0
        and max_rows % stripe_chunk == 0
        and max_rows % PLAN_CHUNK_ROWS == 0
    )


def ceil_div(a: int, b: int) -> int:
    """a / b rounded up."""
    return -(-a // b)


def packet_shape(mesh_device, page_bytes: int) -> tuple[int, int]:
    """(pages_per_packet, banks) of the push's compact layout for cache rows of `page_bytes` bytes: the push sends
    packets of as many rows as fit one fabric payload, dealt round robin over the DRAM banks, so every owner's run in
    a compact buffer is padded to whole units of pages_per_packet * banks rows. Must match packet_shape() in
    ttnn/cpp/ttnn/operations/experimental/sparse_kv_push/device/sparse_kv_push_common.cpp."""
    pages_per_packet = ttnn.get_tt_fabric_max_payload_size_bytes() // page_bytes
    banks = ttnn.get_memory_view(mesh_device, ttnn.BufferType.DRAM).num_banks
    return pages_per_packet, banks


def _shard_over_mesh(tensor, mesh_device):
    """Mark `tensor` as sharded on dim 2 over every chip in row-major order, as the full-mesh gather expects."""
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
        self.ready_semaphore, self.data_valid_semaphore = (
            ttnn.create_global_semaphore(mesh_device, cores, 0, buffer_type) for _ in range(2)
        )
        ttnn.synchronize_device(mesh_device)
        self._plan: SparseKvPushPlan | None = None
        self._plan_key = None
        self._tables = None
        self._buffers = {}
        self._starts = {}
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
        tokens_per_core = ceil_div(query_rows, grid.x * grid.y - push_cores)
        # ...and the fewest whole columns that keep that load (GLM: 160 rows, 2 per core, 80 cores = 8 columns).
        attention_cols = ceil_div(ceil_div(query_rows, tokens_per_core), grid.y)
        # The push takes the bottom rows of the remaining columns.
        push_cols = grid.x - attention_cols
        assert (
            push_cols > 0 and ceil_div(push_cores, push_cols) <= grid.y
        ), f"no room for {push_cores} push cores beside attention on a {grid.x}x{grid.y} grid"
        push_rows = ceil_div(push_cores, push_cols)
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

    def buffer_rows(self, distinct_rows: int, max_rows: int, stripe_chunk: int, page_bytes: int) -> int:
        """Compact buffer rows for up to `distinct_rows` pushed rows. A chip's distinct rows are at most its query
        rows x top-k, so the buffer is sized from the chunk, not from the cache (about 220 MiB at a 1M cache instead of
        the dense gather's 703 MiB)."""
        num_devices = self.mesh_device.get_num_devices()
        pages_per_packet, banks = packet_shape(self.mesh_device, page_bytes)
        unit = pages_per_packet * banks  # each owner's run is padded to whole units
        # Two upper bounds: every distinct row plus at most one unit of padding per owner, or every owner's run full
        # (an owner holds at most its share of the cache's blocks).
        padded_distinct = min(distinct_rows, max_rows) + num_devices * unit
        owner_rows = ceil_div(ceil_div(max_rows, stripe_chunk), num_devices) * stripe_chunk
        all_full = num_devices * ceil_div(owner_rows, unit) * unit
        return min(padded_distinct, all_full)

    def push_buffer(self, parity: int, rows: int, like: ttnn.Tensor) -> ttnn.Tensor:
        """A zeroed, replicated compact buffer of `rows` rows shaped like `like`, one per parity; created once and
        reused by every chunk."""
        key = (parity, rows, like.shape[-1], like.dtype)
        if key not in self._buffers:
            self._buffers[key] = _replicated(
                self.mesh_device, torch.zeros(1, 1, rows, like.shape[-1]), like.dtype, like.layout
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
    ) -> SparseKvPushPlan:
        """The plan for `indices`, cached across the layers that reuse them. The plan op reads the chunk start on
        device, so with chunk_start_tensor one cached plan serves any chunk of a traced replay."""
        start = chunk_start_tensor if chunk_start_tensor is not None else self._chunk_start(chunk_start)
        # Keyed by buffer address: full layers rewrite the shared indices buffer in place and call invalidate(), as
        # does a pipeline stage's first layer, so an unchanged address means the same indices.
        key = (indices.buffer_address(), stripe_chunk, page_bytes, max_rows, start.buffer_address())
        if self._plan is None or self._plan_key != key:
            self._plan_key, self._plan = key, self._build(indices, stripe_chunk, page_bytes, max_rows, start)
        return self._plan

    def _chunk_start(self, start: int) -> ttnn.Tensor:
        if start not in self._starts:
            self._starts[start] = _replicated(
                self.mesh_device, torch.tensor([[[[start]]]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
            )
        return self._starts[start]

    def _build(self, indices, stripe_chunk, page_bytes, max_rows, chunk_start):
        n = self.mesh_device.get_num_devices()
        table, compact_indices = ttnn.experimental.sparse_push_plan(
            indices,
            num_devices=n,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            page_bytes=page_bytes,
            chunk_start=chunk_start,
        )
        _shard_over_mesh(table, self.mesh_device)
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
            table=self._tables, compact_indices=compact_indices, stripe_chunk=stripe_chunk, max_rows=max_rows
        )
