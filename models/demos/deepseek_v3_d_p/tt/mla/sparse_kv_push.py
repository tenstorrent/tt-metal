# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse KVPE push for GLM shared layers.

Shared layers reuse the previous full layer's top-k, so instead of gathering the whole prefix onto every chip, each
owner chip pushes only the rows a requester's indices reference into that requester's compact buffer. sparse_sdpa then
reads the buffer through indices rewritten to compact slots. The plan is built on device, so it is trace-safe.
"""

from collections import Counter
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
    # The tables' all-gather while deferred (plan(defer_gather=True)): run_gather() runs it, e.g. on a sub-device
    # beside attention, since attention needs only compact_indices. None once run.
    pending_gather: object = None

    def run_gather(self, **gather_kwargs):
        if self.pending_gather is not None:
            gather, self.pending_gather = self.pending_gather, None
            gather(**gather_kwargs)


@dataclass(frozen=True)
class PushStep:
    """The attention step a pre-send fills the buffer for, in a packed forward (several chunks per forward): the
    step that runs next. Within a layer that is the next chunk (layer_offset 0); after a layer's last chunk it is the
    next layer's first chunk (layer_offset 1). Its prefix (rows below its chunk start) is complete by then: a slot's
    earlier chunks in the same forward run their attention steps, and so write that layer's KV, before it.

    indices: that step's top-k, or None for the pre-sending step's own (the next layer reusing this chunk's).
    cache_user_id, kv_actual_isl: its chunk's slot and start (eager); metadata: its chunk's metadata (traced)."""

    layer_offset: int
    indices: ttnn.Tensor | None = None
    cache_user_id: int = 0
    kv_actual_isl: int = 0
    metadata: list | None = None


# One chunk per forward: the step after this one is the next layer's, on the same chunk.
SAME_CHUNK_NEXT_LAYER = "same_chunk_next_layer"


def packed_push_targets(nonempty: list[bool]) -> dict[int, tuple[int, int]]:
    """For each chunk position that runs its attention step, the step after it: (layer_offset, chunk position).

    Empty positions (False) run no attention step in an eager forward and are skipped; a traced forward runs every
    position, so passes all True. The last running position points at the next layer's first running position."""
    running = [k for k, real in enumerate(nonempty) if real]
    targets = {k: (0, nxt) for k, nxt in zip(running, running[1:])}
    if running:
        targets[running[-1]] = (1, running[0])
    return targets


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
    request. The plan reads the chunk start from metadata[1] on device, so the cache key holds that tensor's address,
    not its value (a layer that computes its own indices calls invalidate()); the push reads the cache slot from
    metadata[0]. Buffers, semaphores and sub-device managers are created on first use, which the warm pass does before
    any trace capture."""

    def __init__(self, mesh_device, links: int):
        self.mesh_device = mesh_device
        self.links = links
        # Links of the prefix push beside attention. It has slack there (it ends long before attention), while every
        # core it takes slows attention, so it runs on one link's cores and leaves the rest to attention.
        self.prefetch_links = min(links, 1)
        grid = mesh_device.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        l1_small = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank > 0
        buffer_type = ttnn.BufferType.L1_SMALL if l1_small else ttnn.BufferType.L1
        self.ready_semaphore, self.done_semaphore = (
            ttnn.create_global_semaphore(mesh_device, cores, 0, buffer_type) for _ in range(2)
        )
        ttnn.synchronize_device(mesh_device)
        # Plan key -> plan, for every distinct (indices, chunk start) since the last invalidate(): one per chunk of a
        # packed forward, so each shared layer reuses all of them instead of rebuilding them chunk after chunk.
        self._plans: dict = {}
        # Gathered plan tables, one per cached plan (a live plan's table must not be overwritten by the next build).
        # Kept across invalidate() so the warm pass allocates them and traced replays reuse them.
        self._tables: list = []
        # Plans built (sparse_push_plan + all-gather), counted on the host.
        self.builds = 0
        self._buffers = {}
        self._managers = {}  # query rows -> sub-device manager of grids(query rows)
        # layer index -> (plan, buffer parity): the plan whose prefix rows the previous attention step already pushed,
        # beside its attention, into that buffer (None: the dense gather buffer)
        self.prefetched = {}
        # Pre-sends on/off, for comparing against pushing every row in line (tests).
        self.presend = True
        # Pushes enqueued by part name (ALL / PREFIX / REST), counted on the host as they are enqueued.
        self.pushes = Counter()

    def grids(self, query_rows: int):
        """(attention, push) core grids for sparse_sdpa beside the next layer's prefix push.

        The push takes the fewest whole columns that hold its cores (prefetch_links links), in their rows furthest from
        the Ethernet row; attention takes every other column. More attention cores make sparse_sdpa faster even when
        its busiest core keeps the same token count (GLM, 160 rows: 100 cores beat 80), and the push still finishes
        well before attention on its one link."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        push_cores = self.prefetch_links * PUSH_CORES_PER_LINK
        push_cols = divup(push_cores, grid.y)
        attention_cols = grid.x - push_cols
        if attention_cols <= 0 or query_rows <= 0:
            raise ValueError(f"no room for {push_cores} push cores beside attention on a {grid.x}x{grid.y} grid")
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

    def q_stem_grids(self):
        """(q stem, push) core grids for a shared layer's rest push beside its q stem: the push on one link's cores in
        the bottom rows of the last columns (where the prefix push runs beside attention), the q stem on every other
        core. q_stem_columns is how many whole columns the q stem keeps, for rectangular matmul grids."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        push_cores = self.prefetch_links * PUSH_CORES_PER_LINK
        push_cols = divup(push_cores, grid.y)
        push_rows = divup(push_cores, push_cols)
        x0, y0 = grid.x - push_cols, grid.y - push_rows
        push = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
        q_stem = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x0 - 1, grid.y - 1)),
                ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(grid.x - 1, y0 - 1)),
            ]
        )
        self.q_stem_columns = x0
        return q_stem, push

    def q_stem_manager(self):
        """Sub-device manager of q_stem_grids (q stem on sub-device 0, push on 1), created on first use."""
        if "q_stem" not in self._managers:
            q_stem, push = self.q_stem_grids()
            self._managers["q_stem"] = self.mesh_device.create_sub_device_manager(
                [ttnn.SubDevice([q_stem]), ttnn.SubDevice([push])], 0
            )
        return self._managers["q_stem"]

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
        """Drop the cached plans: new indices arrived in the same buffers (the indexer rewrote them, or the previous
        pipeline stage sent them). The gathered-table pool is kept."""
        self._plans.clear()
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
        defer_gather: bool = False,
        core_grid: ttnn.CoreRangeSet | None = None,
    ) -> SparseKvPushPlan:
        """The plan for `indices`, cached across the layers that reuse them, one per chunk of a packed forward. On a
        traced replay the plan op reads the chunk start from chunk_start_tensor on device, so one cached plan serves
        any chunk at that metadata slot; eagerly it takes chunk_start as a runtime arg.

        core_grid: the cores (one rectangle) to build a new plan on, e.g. a loaded sub-device's; the whole grid by
        default. It does not change the plan, so it is not part of the cache key.

        defer_gather leaves a new plan's table gather pending (see SparseKvPushPlan.run_gather); without it, a pending
        gather runs now."""
        # Keyed by buffer address: full layers rewrite the shared indices buffer in place and call invalidate(), as
        # does a pipeline stage's first layer, so an unchanged address means the same indices.
        start_key = chunk_start_tensor.buffer_address() if chunk_start_tensor is not None else chunk_start
        key = (indices.buffer_address(), stripe_chunk, page_bytes, max_rows, start_key)
        plan = self._plans.get(key)
        if plan is None:
            plan = self._build(
                len(self._plans),
                indices,
                stripe_chunk,
                page_bytes,
                max_rows,
                chunk_start,
                chunk_start_tensor,
                core_grid,
            )
            self._plans[key] = plan
        if not defer_gather:
            plan.run_gather()
        return plan

    def _build(self, slot, indices, stripe_chunk, page_bytes, max_rows, chunk_start, chunk_start_tensor, core_grid):
        n = self.mesh_device.get_num_devices()
        table, compact_indices = ttnn.experimental.sparse_push_plan(
            indices,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            page_bytes=page_bytes,
            chunk_start=None if chunk_start_tensor is not None else chunk_start,
            chunk_start_tensor=chunk_start_tensor,
            core_range=core_grid.ranges()[0] if core_grid is not None else None,
        )
        self.builds += 1
        gathered_rows = table.shape[2] * n
        if slot == len(self._tables):
            self._tables.append(None)
        tables = self._tables[slot]
        if tables is None or tables.shape[2] != gathered_rows:
            tables = self._tables[slot] = _replicated(
                self.mesh_device,
                torch.zeros(1, 1, gathered_rows, table.shape[3], dtype=torch.int32),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            )

        def gather(**gather_kwargs):
            ttnn.experimental.high_bw_all_gather(
                table, dim=2, output_tensor=tables, cluster_axis=None, num_links=self.links, **gather_kwargs
            )
            ttnn.deallocate(table)

        return SparseKvPushPlan(
            table=tables,
            compact_indices=compact_indices,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            pending_gather=gather,
        )
