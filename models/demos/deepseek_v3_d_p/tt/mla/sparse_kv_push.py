# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sparse KVPE push for GLM shared layers.

Shared layers reuse the previous full layer's top-k, so instead of gathering the whole prefix onto every chip, each
owner chip pushes only the rows a requester's indices reference into that requester's compact buffer. sparse_sdpa then
reads the buffer through indices rewritten to compact slots. The plan is built on device, so it is trace-safe.
"""

import itertools
import os
from collections import Counter
from dataclasses import dataclass, field

import torch

import ttnn
from ttnn.core import divup

# Which of a requester's needed rows a push sends: ALL, PREFIX (rows earlier chunks wrote) or REST.
SparseKvPushPart = ttnn.experimental.SparseKvPushPart

# sparse_kv_dispatch geometry (KV_DISPATCH_2D_PLAN.md): planner cores and lanes a link.
KVD_PLANNERS = int(os.environ.get("SPARSE_MLA_KV_DISPATCH_PLANNERS", "48"))
KVD_LANES_PER_LINK = int(os.environ.get("SPARSE_MLA_KV_DISPATCH_LANES", "2"))


def _spike() -> bool:
    """SPIKE (GLM_KV_DISPATCH_PLAN.md Phase 0, not for merge): see mla.py _kvd_spike."""
    return os.environ.get("SPARSE_MLA_KVD_SPIKE", "0") == "1"


def _spike_compact() -> bool:
    """SPIKE: the dispatch lands rows in the push's compact buffers (plan rank table), not at global pages."""
    return _spike() and os.environ.get("SPARSE_MLA_KVD_COMPACT", "1") == "1"


# Cores the push needs per fabric link, for laying out its grid beside sparse_sdpa.
PUSH_CORES_PER_LINK = ttnn.experimental.SPARSE_KV_PUSH_CORES_PER_LINK


_PLAN_GENERATIONS = itertools.count(1)  # (32-bit on device: wraps after 4G plan builds)


@dataclass
class SparseKvPushPlan:
    table: ttnn.Tensor  # every chip's plan table, all-gathered
    compact_indices: ttnn.Tensor
    stripe_chunk: int
    max_rows: int
    # Inbox mode: `table` is this chip's own plan table and the push exchanges the tables through this inbox.
    inbox: ttnn.Tensor | None = None
    # The planner's cache slot of this plan (its gathered-table pool entry).
    slot: int = 0
    # SPIKE: a dispatch on this plan build already exchanged its slices into `inbox` (later calls skip the exchange).
    exchanged: bool = False
    # SPIKE (compact landing): the plan's rank table, for sparse_kv_dispatch to land rows at compact pages.
    rank_table: ttnn.Tensor | None = None
    # SPIKE (plan once, dispatch many): this plan slot's routing record; `recorded` once a dispatch on this plan build
    # recorded its planners' routing there (later calls replay it).
    replay_buffer: ttnn.Tensor | None = None
    recorded: bool = False
    # this plan build's number: the record holds it, a replay of another build routes instead (exact)
    generation: int = field(default_factory=lambda: next(_PLAN_GENERATIONS))


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


def sparse_kv_push_supported(mesh_device, stripe_chunk: int, max_rows: int, links: int) -> bool:
    """Whether the push can serve a cache of `max_rows` rows in blocks of `stripe_chunk` rows on this mesh with `links`
    fabric links (the ops' own check); otherwise shared layers gather the whole prefix."""
    return ttnn.experimental.is_sparse_kv_push_supported(mesh_device, stripe_chunk, max_rows, links)


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
        # Inbox mode (owner-major plans): the dispatch exchanges the plan tables itself, so a plan is not followed by an
        # all-gather of every chip's table. Scratch of each dispatch call, shared by every plan.
        self._inbox = None
        self._buffers = {}
        self._managers = {}  # query rows -> sub-device manager of grids(query rows)
        # layer index -> (plan, buffer parity): the plan whose prefix rows the previous attention step already pushed,
        # beside its attention, into that buffer (None: the dense gather buffer)
        self.prefetched = {}
        # Pre-sends on/off, for comparing against pushing every row in line (tests; SCRATCH_NO_PRESEND=1 off).
        self.presend = not os.environ.get("SCRATCH_NO_PRESEND")
        # Pushes enqueued by part name (ALL / PREFIX / REST), counted on the host as they are enqueued.
        self.pushes = Counter()
        # sparse_kv_dispatch: its four persistent words and its scratch, made on first use
        self._kvd_sems = None
        self._kvd_scratch = {}
        # SPIKE: dispatch calls by part name
        self.kvd_calls = Counter()

    def dispatch_resources(self, max_rows: int, stripe_chunk: int, planners: int = KVD_PLANNERS, grid_key=None):
        """(semaphores, scratch) of sparse_kv_dispatch for a cache of max_rows rows: made on first use (the warm pass,
        before any trace capture) and shared by every layer."""
        # One set of persistent words per program variant (planner count, grid and placement: grid_key, None for the
        # whole chip): the fence pairs each TX with its neighbour's TX of the same program, whose cores differ between
        # variants. Shared words let one chip's next call (the other variant) release a neighbour's TX still in the
        # previous call, which then sends into landing pools not yet set up (non-deterministic rows).
        if self._kvd_sems is None:
            self._kvd_sems = {}
        sem_key = (planners, grid_key)
        if sem_key not in self._kvd_sems:
            grid = self.mesh_device.compute_with_storage_grid_size()
            cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))])
            l1_small = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.L1_SMALL).total_bytes_per_bank > 0
            buffer_type = ttnn.BufferType.L1_SMALL if l1_small else ttnn.BufferType.L1
            self._kvd_sems[sem_key] = [
                ttnn.create_global_semaphore(self.mesh_device, cores, 0, buffer_type) for _ in range(4)
            ]
        key = (max_rows, stripe_chunk, planners, grid_key)  # (scratch too: one a program variant)
        if key not in self._kvd_scratch:
            spec = ttnn.experimental.sparse_kv_dispatch_scratch_spec(
                self.mesh_device, max_rows, stripe_chunk, planners, self.links, KVD_LANES_PER_LINK
            )
            self._kvd_scratch[key] = ttnn.allocate_tensor_on_device(spec, self.mesh_device)
        return self._kvd_sems[sem_key], self._kvd_scratch[key]

    def _replay_buffer(self, slot: int, max_rows: int, stripe_chunk: int):
        """SPIKE: plan slot `slot`'s sparse_kv_dispatch routing record (made on first use, before any capture)."""
        if not hasattr(self, "_replay"):
            self._replay = {}
        key = (slot, max_rows, stripe_chunk)
        if key not in self._replay:
            spec = ttnn.experimental.sparse_kv_dispatch_replay_spec(self.mesh_device, max_rows, stripe_chunk)
            self._replay[key] = ttnn.allocate_tensor_on_device(spec, self.mesh_device)
        return self._replay[key]

    def grids(self, query_rows: int):
        """(attention, push) core grids for sparse_sdpa beside the next layer's prefix push.

        Cores past the fewest that keep attention's busiest core at its token count only add DRAM contention with the
        push, so attention takes that many whole columns. The push sits in the remaining columns' rows furthest from
        the Ethernet row, where it contends least with attention."""
        grid = self.mesh_device.compute_with_storage_grid_size()
        if _spike():
            # SPIKE: the dispatch (pre-send and rest) takes the 4 columns right of attention's 8, their bottom rows (8 of
            # 10: 32 cores; 24 port cores and 4 planners). Every row costs attention up to 3% (the dispatch's ports and
            # planners beside SDPA's DRAM readers); SPARSE_MLA_KVD_SPIKE_ROWS overrides.
            attention_cols = grid.x - 4
            attention = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(attention_cols - 1, grid.y - 1))]
            )
            rows = max(1, min(int(os.environ.get("SPARSE_MLA_KVD_SPIKE_ROWS", "8")), grid.y))
            push = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(attention_cols, grid.y - rows), ttnn.CoreCoord(grid.x - 1, grid.y - 1))]
            )
            return attention, push
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
        if key not in self._buffers and _spike() and not _spike_compact():  # SPIKE: global landing: a full-size buffer
            self._buffers[key] = ttnn.allocate_tensor_on_device(like.spec, self.mesh_device)
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
        owner_major: bool = False,
    ) -> SparseKvPushPlan:
        """The plan for `indices`, cached across the layers that reuse them, one per chunk of a packed forward. On a
        traced replay the plan op reads the chunk start from chunk_start_tensor on device, so one cached plan serves
        any chunk at that metadata slot; eagerly it takes chunk_start as a runtime arg. owner_major: the owner-major
        table and inbox sparse_kv_dispatch needs (if a slot fits a packet); otherwise the all-gathered tables
        sparse_kv_push takes."""
        # Keyed by buffer address: full layers rewrite the shared indices buffer in place and call invalidate(), as
        # does a pipeline stage's first layer, so an unchanged address means the same indices.
        start_key = chunk_start_tensor.buffer_address() if chunk_start_tensor is not None else chunk_start
        key = (indices.buffer_address(), stripe_chunk, page_bytes, max_rows, start_key, owner_major)
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
                owner_major,
            )
            self._plans[key] = plan
        return plan

    def _build(
        self,
        slot,
        indices,
        stripe_chunk,
        page_bytes,
        max_rows,
        chunk_start,
        chunk_start_tensor,
        owner_major=False,
    ):
        n = self.mesh_device.get_num_devices()
        # Inbox mode needs a table slot to fit one fabric packet; larger caches on small meshes keep the all-gather.
        slot_bytes = ttnn._ttnn.operations.experimental.sparse_kv_push_inbox_slot_bytes(max_rows, stripe_chunk, n)
        pages_per_packet, _ = ttnn._ttnn.operations.experimental._sparse_kv_push_packet_shape(
            self.mesh_device, page_bytes
        )
        # (SPIKE: the dispatch fragments slots, so a dispatch plan is always owner-major)
        inbox_mode = owner_major and (slot_bytes <= pages_per_packet * page_bytes or _spike())
        outputs = ttnn.experimental.sparse_push_plan(
            indices,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            page_bytes=page_bytes,
            chunk_start=None if chunk_start_tensor is not None else chunk_start,
            chunk_start_tensor=chunk_start_tensor,
            owner_major=inbox_mode,
            rank_table=inbox_mode and _spike_compact(),
        )
        self.builds += 1
        table, compact_indices = outputs[0], outputs[1]
        if inbox_mode:
            slot_words = slot_bytes // 4
            # one inbox per cached plan slot: a later call on this plan reuses its exchanged slices
            if not isinstance(self._inbox, dict):
                self._inbox = {}
            if slot not in self._inbox or self._inbox[slot].shape[3] != slot_words:
                self._inbox[slot] = _replicated(
                    self.mesh_device,
                    torch.zeros(1, 1, n, slot_words, dtype=torch.int32),
                    ttnn.uint32,
                    ttnn.ROW_MAJOR_LAYOUT,
                )
            return SparseKvPushPlan(
                table=table,
                compact_indices=compact_indices,
                stripe_chunk=stripe_chunk,
                max_rows=max_rows,
                inbox=self._inbox[slot],
                slot=slot,
                rank_table=outputs[-1] if _spike_compact() else None,
                replay_buffer=(
                    self._replay_buffer(slot, max_rows, stripe_chunk)
                    if _spike() and os.environ.get("SPARSE_MLA_KVD_REPLAY", "1") == "1"
                    else None
                ),
            )
        gathered_rows = table.shape[2] * n
        # Inbox-mode plans hold no gathered table, so their slots leave gaps here.
        self._tables.extend([None] * (slot + 1 - len(self._tables)))
        tables = self._tables[slot]
        if tables is None or tables.shape[2] != gathered_rows:
            tables = self._tables[slot] = _replicated(
                self.mesh_device,
                torch.zeros(1, 1, gathered_rows, table.shape[3], dtype=torch.int32),
                ttnn.uint32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
        ttnn.experimental.high_bw_all_gather(
            table, dim=2, output_tensor=tables, cluster_axis=None, num_links=self.links
        )
        ttnn.deallocate(table)
        return SparseKvPushPlan(
            table=tables,
            compact_indices=compact_indices,
            stripe_chunk=stripe_chunk,
            max_rows=max_rows,
            slot=slot,
        )
