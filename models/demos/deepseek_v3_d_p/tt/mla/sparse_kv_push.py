# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Planner for the sparse KVPE push of GLM shared layers.

Instead of gathering the whole populated prefix onto every chip, each owner chip pushes only the KVPE rows the
requester's top-k indices reference, into the requester's compact buffer; sparse_sdpa then reads that buffer through
indices rewritten to compact slots (no block-cyclic remap).

The plan is built on device (ttnn.experimental.sparse_push_plan, then an all-gather of its tables), so it needs no host
round trip and is trace-safe. Rows at least HOT_THRESHOLD chips need are "hot" (ttnn.experimental.sparse_push_hot, on
the gathered tables): their owner multicasts them once into a hot region at the front of every chip's compact buffer
instead of pushing them to each requester, which cuts the load on the fabric's busiest links. One plan serves every
layer that reuses the same indices (GLM shared layers), since only the KV slot differs between them. The indexer writes every chunk's indices into the same buffer, so a layer that
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
    hot_table: ttnn.Tensor | None = None  # this chip's sparse_push_hot table (push_hot_table of the pushes)


# A row is hot when at least this many chips need it (GLM-5.3 at 1M: the busiest link carries ~36% less at L42/L62
# and less at every full layer). Per owner at most HOT_PREFIX_CAP hot rows of the prefix and HOT_CURRENT_CAP of the
# current chunk (2 blocks always fit: a chunk start need not be block aligned), which bounds the hot region.
HOT_THRESHOLD = 12
HOT_PREFIX_CAP = 2048
HOT_CURRENT_CAP = 512


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


# L1 front of each compact buffer, per core: sparse_sdpa, bound by its row reads, reads these rows from L1 instead of
# DRAM. Both parities stay allocated for the whole forward, so 2 x this must leave L1 for the largest GLM program's
# circular buffers (on the Galaxy about 370 KB per core are free; 2 x 192 KB clashes).
L1_FRONT_BYTES_PER_CORE = 160 * 1024


class SparseKvPushPlanner:
    """Builds and caches the push plan for one set of injected indices; shared by all MLA layers of a mesh.

    Every buffer is allocated on first use, which the warm pass does before a trace capture."""

    def __init__(self, mesh_device, links: int, l1_front_bytes_per_core: int | None = None, hot: bool = True):
        self.mesh_device = mesh_device
        self.links = links
        self.hot = hot
        self.l1_front_bytes_per_core = (
            L1_FRONT_BYTES_PER_CORE if l1_front_bytes_per_core is None else l1_front_bytes_per_core
        )
        # With an L1 front, each packet's rows are consecutive compact slots (one front shard or spill page); without
        # one they are spread over the DRAM banks so a packet lands in one bank.
        self.banks = 1 if self.l1_front_bytes_per_core else mesh_device.dram_grid_size().x
        self._fronts = {}
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

    def spill_buffer(self, parity, rows, like, page_bytes):
        """DRAM spill behind an L1 front: pages of one push packet each (pages_per_packet rows of page_bytes), so a
        packet lands in one write; compact row r is at page r // pages_per_packet."""
        unit = self._pages_per_packet(page_bytes)
        width = unit * page_bytes // (1 if like.dtype == ttnn.fp8_e4m3 else 2)
        key = ("spill", parity, rows, width, like.dtype)
        if key not in self._buffers:
            self._buffers[key] = _replicated(
                self.mesh_device, torch.zeros(1, 1, -(-rows // unit), width), like.dtype, like.layout
            )
        return self._buffers[key]

    def front_buffer(self, parity, like, page_bytes):
        """L1 front of the compact buffer of this parity: its first rows, with rows padded to DRAM-aligned pages.
        Sharded one packet per shard, round-robin over every worker core, so consecutive packets (and the rows
        sparse_sdpa reads) spread over all cores' L1 instead of crowding a few. None when disabled."""
        if not self.l1_front_bytes_per_core:
            return None
        key = (parity, like.shape[-1], like.dtype)
        if key not in self._fronts:
            grid = self.mesh_device.compute_with_storage_grid_size()
            cores = grid.x * grid.y
            unit = self._pages_per_packet(page_bytes)
            rows_per_core = self.l1_front_bytes_per_core // page_bytes // unit * unit
            width = page_bytes // (1 if like.dtype == ttnn.fp8_e4m3 else 2)
            memory_config = ttnn.MemoryConfig(
                ttnn.BufferType.L1,
                ttnn.NdShardSpec(
                    ttnn.Shape([1, 1, unit, width]),
                    ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                    ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
                ),
            )
            # Uninitialized: the push writes every slot sparse_sdpa reads.
            self._fronts[key] = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, 1, rows_per_core * cores, width]),
                like.dtype,
                ttnn.ROW_MAJOR_LAYOUT,
                self.mesh_device,
                memory_config,
            )
        return self._fronts[key]

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
        """Compact buffer rows for up to `rows` distinct rows: each owner's run pads to whole bank-lane units, behind
        the hot region (each owner's two capped sub-runs, padded likewise)."""
        unit = self._pages_per_packet(page_bytes) * self.banks
        devices = self.mesh_device.get_num_devices()
        hot = devices * sum(-(-cap // unit) * unit for cap in (HOT_PREFIX_CAP, HOT_CURRENT_CAP)) if self.hot else 0
        return hot + rows + devices * unit

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
        run_gather=None,
    ) -> SparseKvPushPlan:
        """The plan for `indices`, cached on the indices buffer. Trace-safe with chunk_start_tensor (the metadata's
        kv_actual_global): the plan op reads the chunk start on device, so the cache key leaves its value out. Every
        layer that computes its own indices invalidate()s, so the key only has to tell calls within one forward apart.

        `run_gather(gather)`, when given, issues the plan tables' all-gather: it calls gather(subdevice_id) (the caller
        runs it on a sub-device next to other work; Q-stem || push overlap, GLM_QPUSH_FULL). Unused on a cache hit.
        """
        start = chunk_start_tensor if chunk_start_tensor is not None else self.chunk_start(chunk_start)
        key = (indices.buffer_address(), stripe_chunk, page_bytes, start.buffer_address())
        if self._last is None or self._key != key:
            self._key, self._last = key, self._build(indices, stripe_chunk, page_bytes, max_rows, start, run_gather)
        return self._last

    def _build(self, indices, stripe_chunk, page_bytes, max_rows, chunk_start, run_gather=None):
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
        gather = lambda subdevice_id=None: ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=self._tables, cluster_axis=None, subdevice_id=subdevice_id, num_links=self.links
        )
        if run_gather is not None:
            run_gather(gather)
        else:
            gather()
        hot_table = None
        if self.hot:
            assert 2 * stripe_chunk <= HOT_CURRENT_CAP, f"stripe_chunk {stripe_chunk} > HOT_CURRENT_CAP / 2"
            hot_table = ttnn.experimental.sparse_push_hot(
                indices,
                compact,
                table,
                self._tables,
                num_devices=self.mesh_device.get_num_devices(),
                stripe_chunk=stripe_chunk,
                max_rows=max_rows,
                pages_per_packet=self._pages_per_packet(page_bytes),
                banks=self.banks,
                threshold=HOT_THRESHOLD,
                prefix_cap=HOT_PREFIX_CAP,
                current_cap=2 * stripe_chunk,
                chunk_start=chunk_start,
            )
        ttnn.deallocate(table)
        return SparseKvPushPlan(
            table=self._tables,
            compact_indices=compact,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            hot_table=hot_table,
        )
