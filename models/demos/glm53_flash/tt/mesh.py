# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mesh layout, weight placement and collectives for the multi-chip GLM-5.3-Flash decoder.

Axis convention (design section 4): mesh columns (axis 1) are the tensor-parallel group of a
row (heads / intermediate / vocab split ``cols``-way), mesh rows (axis 0) are the data-parallel
groups for attention and the expert-TP axis of the routed MoE. A ``(1, 1)`` mesh degrades every
collective to the identity so the single-chip stage-1 modules keep running unchanged.

Placement helpers return ``ttnn`` mesh mappers for ``ttnn.from_torch``; ``upload`` wraps the
common case. Collectives wrap the generic ``ttnn.all_reduce`` / ``all_gather`` /
``reduce_scatter`` composites with ``cluster_axis`` (the ones gpt_oss uses on the Blackhole
Galaxy); every collective must run once eagerly before a trace is captured. The TP all-reduce has a
second form (``GLM53_TP_ALLREDUCE=gather``): one all-gather of the column partials + a local
``fast_reduce_nc`` sum (decode ledger B7, ``all_reduce_cols``).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import torch
from loguru import logger

import ttnn

TP_AXIS = 1  # columns
DP_AXIS = 0  # rows (and expert-TP)
TP_ALLREDUCE_GATHER_ROWS = 32  # gather-mode all-reduce (GLM53_TP_ALLREDUCE=gather): decode activations only
# Roofline C2 (2026-10-26): the attention TP all-reduce measures 51 us (RS 39 + AG 12, ep_b4 #269-270) where the MoE one
# on the same [1,1,32,4096] bf16 shape measures 30 (RS 12 + AG 18, #310-311). Code reading of every decode call site
# (kda.py ``_block_decode`` / ``_output``, dsa.py ``_o_proj`` / ``_decode_state_unified``, mlp.py, moe.py): all hand
# ``all_reduce_cols`` the output of a ``DecodeLinear`` (``ttnn.linear``, 1D mcast config) or of an ``add`` of two such
# outputs -- DRAM interleaved, bf16, TILE, [1,1,B,H] -- with the same link count (``ttnn.get_num_links``): the input
# placement and memory config are IDENTICAL, so nothing on the input side explains the 21 us. The extra time sits in the
# reduce-scatter phase (39 vs 12 us), which is where a chip waits for the row's other columns: column skew accumulated
# over the ~600 us of column-independent ops since the previous (synchronizing) MoE all-reduce (the MoE all-reduce follows
# the row collective of the experts, which lines the column up). ``GLM53_TP_ALLREDUCE_INPUT=l1`` is the only input-side
# experiment left (one ``to_memory_config`` into L1 interleaved before the composite on decode shapes; default ``dram`` =
# today's path, no extra op); the decisive measurement is a per-chip tracy of the RS row's start timestamps per column.
TP_ALLREDUCE_INPUT_ENV = "GLM53_TP_ALLREDUCE_INPUT"


def mesh_device_params(rows: int, cols: int, **overrides) -> dict:
    """``device_params`` for the pytest ``mesh_device`` fixture: fabric on for any multi-chip
    mesh (2D torus on the full Galaxy, plain 2D for a block), relaxed init, the architecture's
    maximum fabric payload, an L1-small region for the composite collectives."""
    params: dict = {"l1_small_size": 32768}
    if rows * cols > 1:
        import os

        from models.demos.deepseek_v3_d_p.tt.moe.init_helpers import create_fabric_router_config, get_max_payload_size

        # GLM53_FABRIC: torus (default on the 8x4) | 2d (default on a block) | 1d | 1d_ring. The EP expert layout
        # (GLM53_MOE_LAYOUT=ep) defaults to 1d: ``ttnn.all_to_all_dispatch`` / ``all_to_all_combine`` send 1D hop
        # packets under a Linear topology and are exercised under FABRIC_1D only (Blackhole CI, the DeepSeek-V3
        # Galaxy model); the first block run under FABRIC_2D hung. Validated here under FABRIC_1D on a (2,4) block
        # (tests/test_a2a_perf.py::test_dispatch_combine_golden, tests/test_mesh_moe.py with the ep layout).
        default = (
            "1d"
            if os.environ.get("GLM53_MOE_LAYOUT", "hybrid") == "ep"
            else ("torus" if (rows, cols) == (8, 4) else "2d")
        )
        fabric = os.environ.get("GLM53_FABRIC", default)
        params["fabric_config"] = {
            "torus": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "2d": ttnn.FabricConfig.FABRIC_2D,
            "1d": ttnn.FabricConfig.FABRIC_1D,
            "1d_ring": ttnn.FabricConfig.FABRIC_1D_RING,
        }[fabric]
        params["reliability_mode"] = ttnn.FabricReliabilityMode.RELAXED_INIT
        params["fabric_router_config"] = create_fabric_router_config(get_max_payload_size())
    params.update(overrides)
    return params


# Per-device runtime state shared by every GLMMesh built over the same mesh device (``GLMMesh.__post_init__``):
# ``frozen`` = [bool] (a trace exists), ``transients`` = the lazily built device buffers made under that rule since
# the last ``GLMMesh.release_transients``, ``memo`` = those transients by key (one build per key per step: a chunk's
# geometry constants are asked for by every layer), ``logged`` = the keys whose post-capture build was reported once.
_RUNTIME: dict[tuple, dict] = {}


def _runtime_key(device) -> tuple:
    ident = getattr(device, "id", None)
    if callable(ident):
        try:
            return ("device", int(ident()))
        except Exception:  # pragma: no cover - a device without a usable id
            pass
    return ("object", id(device))


def _allocated(t) -> bool:
    """True when ``t`` (a tensor or a list of tensors from ``GLMMesh.lazy``) still holds its device buffers."""
    items = list(t) if isinstance(t, (list, tuple)) else [t]
    return all(getattr(i, "is_allocated", None) is not None and i.is_allocated() for i in items)


def _runtime_state(device) -> dict:
    return _RUNTIME.setdefault(_runtime_key(device), {"frozen": [False], "transients": [], "memo": {}, "logged": set()})


def lazy_const(mesh: "GLMMesh | None", cache: dict, key, make: Callable[[], object]):
    """``GLMMesh.lazy`` for modules that may run without a mesh (single-chip unit tests): plain caching then."""
    if mesh is not None:
        return mesh.lazy(cache, key, make)
    t = cache.get(key)
    if t is None:
        t = cache[key] = make()
    return t


def from_torch_host_layout(
    t: torch.Tensor, dtype, layout, device, *, memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=None
) -> ttnn.Tensor:
    """``ttnn.from_torch`` whose layout conversion runs on the HOST (``from_torch`` without a device, then
    ``to_device``: a copy, no program). With a device given, ``from_torch`` tilizes on the device — a Tilize
    program per tensor shape, which under a sealed program cache (after the trace captures) is a request-time
    compile. For the small per-request constants (masks, selectors, bounds) the host conversion is free."""
    kw = {"mesh_mapper": mesh_mapper} if mesh_mapper is not None else {}
    host = ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, **kw)
    return ttnn.to_device(host, device, memory_config=memory_config)


@dataclass(frozen=True)
class GLMMesh:
    """A mesh device plus the axis convention. ``tp`` = columns, ``dp`` = rows."""

    device: ttnn.MeshDevice
    rows: int
    cols: int
    topology: ttnn.Topology | None = None  # None = ttnn default for the fabric
    _ccl: object = field(default=None, repr=False, compare=False)  # lazy TT_CCL (async collectives)
    _reduce_ckc: object = field(default=None, repr=False, compare=False)  # lazy fp32 config (all-reduce gather mode)
    # device constants shared by every module of the model (``const``): one tensor per key instead of one per layer
    _consts: dict = field(default_factory=dict, repr=False, compare=False)
    _prefill_depth: list = field(default_factory=lambda: [0], repr=False, compare=False)  # prefill_scope nesting
    # the eager KDA prefill's persistent chronology scalars (tt/kda.py chrono_scalar: one start/end pair per mesh)
    chrono: dict = field(default_factory=dict, repr=False, compare=False)
    # [bool], set by the mesh model after a trace capture (``freeze_allocations``): a device buffer allocated while
    # traces exist may sit where a replay's transient buffers were and is overwritten by the next replay (the
    # 2026-10-27 served KDA hang), so no lazily built constant may outlive a replay after it. The list is the
    # device's (``_runtime_state``): every GLMMesh over the same mesh device shares it, including the per-layer
    # meshes ``FunctionalDecoder.from_state_dict`` builds, whose guards (tt/kda.py ``scalar`` / ``chrono_scalar``)
    # never saw the model mesh's flag before 2026-10-28.
    allocations_frozen: list = field(default=None, repr=False, compare=False)
    # the device buffers ``lazy`` built under live traces since the last ``release_transients`` (shared per device)
    _transients: list = field(default=None, repr=False, compare=False)
    # the live transients by key (shared per device): a key asked for again within the step is served, not rebuilt
    _transient_memo: dict = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        state = _runtime_state(self.device)
        object.__setattr__(self, "allocations_frozen", state["frozen"])  # frozen dataclass
        object.__setattr__(self, "_transients", state["transients"])
        object.__setattr__(self, "_transient_memo", state["memo"])

    @classmethod
    def from_device(cls, mesh_device, topology=None) -> "GLMMesh":
        shape = list(mesh_device.shape)
        return cls(mesh_device, int(shape[0]), int(shape[1]), topology)

    @property
    def tp(self) -> int:
        return self.cols

    @property
    def dp(self) -> int:
        return self.rows

    @property
    def is_single(self) -> bool:
        return self.rows == 1 and self.cols == 1

    @property
    def mesh_shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    # ------------------------------------------------------------------ placement
    def replicate(self):
        return ttnn.ReplicateTensorToMesh(self.device)

    def shard_cols(self, dim: int):
        """Split ``dim`` over the TP group (columns), replicate over rows."""
        return ttnn.ShardTensor2dMesh(self.device, mesh_shape=self.mesh_shape, dims=(None, dim))

    def shard_rows(self, dim: int):
        """Split ``dim`` over the rows, replicate over columns."""
        return ttnn.ShardTensor2dMesh(self.device, mesh_shape=self.mesh_shape, dims=(dim, None))

    def shard_2d(self, row_dim: int | None, col_dim: int | None):
        if row_dim is not None and row_dim == col_dim:
            # one host dim split over BOTH axes (chunk i of rows*cols -> chip (i // cols, i % cols)): the 2-D mapper
            # rejects a repeated dim ("dims must be unique"), the 1-D one shards over the row-major chip list
            return ttnn.ShardTensorToMesh(self.device, dim=row_dim)
        return ttnn.ShardTensor2dMesh(self.device, mesh_shape=self.mesh_shape, dims=(row_dim, col_dim))

    def upload(
        self,
        t: torch.Tensor,
        dtype,
        *,
        cols: int | None = None,
        rows: int | None = None,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    ) -> ttnn.Tensor:
        """``from_torch`` with the placement ``(rows, cols)`` = host dims split over the mesh
        rows / columns (``None`` = replicated on that axis)."""
        if cols is not None:
            assert t.shape[cols] % self.cols == 0, (tuple(t.shape), cols, self.cols)
        if rows is not None:
            assert t.shape[rows] % self.rows == 0, (tuple(t.shape), rows, self.rows)
        if self.allocations_frozen[0]:
            # after a capture no program may be built, and ``from_torch`` with a device tilizes ON the device
            # (a Tilize program per shape; the 2026-10-28 served APC seal hit on the ragged chunk's router
            # mask): convert on the host, then a plain copy
            return from_torch_host_layout(
                t, dtype, layout, self.device, memory_config=memory_config, mesh_mapper=self.shard_2d(rows, cols)
            )
        return ttnn.from_torch(
            t.contiguous(),
            dtype=dtype,
            layout=layout,
            device=self.device,
            memory_config=memory_config,
            mesh_mapper=self.shard_2d(rows, cols),
        )

    def _shard_shape(self, shape, rows: int | None, cols: int | None) -> tuple[int, ...]:
        """The per-device shard shape of a global ``shape`` sharded on dim ``rows`` over the mesh rows and on
        dim ``cols`` over the columns (``None`` = replicated on that axis; negative dims allowed)."""
        local = list(shape)
        for dim, n in ((rows, self.rows), (cols, self.cols)):
            if dim is not None and n > 1:
                d = dim % len(local)
                local[d] = -(-local[d] // n)
        return tuple(local)

    def upload_cached(
        self,
        make: Callable[[], torch.Tensor],
        dtype,
        cache_file: Path | str | None,
        *,
        cols: int | None = None,
        rows: int | None = None,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        expect_shape: tuple[int, ...] | None = None,
    ) -> ttnn.Tensor:
        """``upload`` through a tensorbin cache: when ``cache_file`` exists the device tensor is
        ``ttnn.load_tensor``-ed from it (the host tensor is never built: ``make`` is a callable so the
        caller can skip reading / dequantizing the checkpoint on a hit); otherwise ``make()`` is
        uploaded and dumped into the cache (written to a temporary file and renamed, so a reader never
        sees a partial file). A hit is checked against ``dtype`` / ``layout`` / ``expect_shape``: a stale
        or foreign file raises instead of silently feeding wrong weights. The caller puts the mesh
        shape and placement into the file name (a (1, 1) dump must not be loaded onto a (2, 4) mesh).
        Measured on one chip: the 7.7 GB bfp8 experts of a MoE layer load in ~2.6 s against ~50 s for
        the host bfp8 conversion."""
        if cache_file is not None:
            cache_file = Path(cache_file)
            if cache_file.is_file():
                t = ttnn.load_tensor(cache_file, device=self.device)
                # a loaded multi-device tensor reports its per-device shard shape, the uploaded one the global
                # (mesh-wide) shape: both describe the same shards, so either matches ``expect_shape``
                shape_ok = expect_shape is None or tuple(t.shape) in (
                    tuple(expect_shape),
                    self._shard_shape(expect_shape, rows, cols),
                )
                if t.dtype != dtype or t.layout != layout or not shape_ok:
                    raise RuntimeError(
                        f"weight cache {cache_file}: {t.dtype} {t.layout} {tuple(t.shape)} does not match the "
                        f"expected {dtype} {layout} {expect_shape} (per-device "
                        f"{self._shard_shape(expect_shape, rows, cols)}); delete the stale file"
                    )
                return t
        t = self.upload(make(), dtype, cols=cols, rows=rows, layout=layout, memory_config=memory_config)
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_name(f"{cache_file.stem}.tmp{os.getpid()}{cache_file.suffix}")
            try:
                ttnn.dump_tensor(tmp, t)
                os.replace(tmp, cache_file)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        return t

    def const(self, key: tuple, make: Callable[[], ttnn.Tensor]) -> ttnn.Tensor:
        """A device constant shared by every module built on this mesh: ``make()`` runs once per ``key`` (on the
        first, eager use), later callers get the same tensor. For constants whose only role is to be copied (the
        zeroed slot tensor of the EP combine): one DRAM tensor per shape for the model instead of one per layer.
        Persistent constants must stay out of L1: an interleaved L1 constant per layer (8 KB pages round-robin from
        bank 0) pins the first cores' L1, and after enough layers the static circular buffers of the next program
        clash with the L1 buffers (the EP32 8x4 e2e, doc/mesh/README.md). Built under live traces it is a chunk
        transient (``lazy``)."""
        return self.lazy(self._consts, key, make)

    # ------------------------------------------------------- allocation discipline under live traces
    @property
    def frozen(self) -> bool:
        """True once a trace was captured on this device (``freeze_allocations``)."""
        return bool(self.allocations_frozen[0])

    def freeze_allocations(self) -> None:
        """After a trace capture: from here on a lazily built device buffer must not outlive the next replay
        (``lazy`` makes it a chunk transient; ``tt/kda.py`` ``scalar`` / ``chrono_scalar`` refuse outright)."""
        self.allocations_frozen[0] = True

    def lazy(self, cache: dict, key, make: Callable[[], object]):
        """A lazily built device constant (``make()`` returns a tensor or a list of tensors) keyed by its geometry.
        Built before the first trace capture it is cached in ``cache`` for the model's life. Built under live
        traces it is a chunk transient instead: handed to the caller, never cached, and released by
        ``release_transients`` at the end of the enclosing eager chunk (``GLM53MeshModel.prefill_streams`` /
        ``decode_step``), so no buffer allocated after a capture outlives the next replay (the tt-metal allocator's
        "potentially unsafe" warning names exactly that: a buffer allocated under an active trace is corrupted by
        the trace's execution unless it is released before then; the 2026-10-27 served hang was ``scalar(49152)``
        allocated after the captures). Within a step a transient is built once per key and served to the later
        askers (a chunk's geometry constants are asked for by every layer: one upload instead of one per layer, as
        the cached path before the captures), unless a caller freed it. The first post-capture build of a key is
        logged once at INFO: a served log then names the chunk geometries the boot warm-up
        (``GLM53_VLLM_WARMUP_PREFILL`` + the buckets) left out."""
        t = cache.get(key)
        if t is not None:
            return t
        if self.allocations_frozen[0]:
            t = self._transient_memo.get(key)
            if t is not None and _allocated(t):
                return t
            t = make()
            self._note_transient(key, t)
            self._transient_memo[key] = t
            return t
        t = cache[key] = make()
        return t

    def _note_transient(self, key, t) -> None:
        self._transients.extend(list(t) if isinstance(t, (list, tuple)) else [t])
        logged = _runtime_state(self.device)["logged"]
        if key not in logged:
            logged.add(key)
            logger.info(
                f"mesh: device constant {key!r} first built after a trace capture: a chunk transient from now on "
                "(released at the chunk end, rebuilt per use); extend the boot warm-up to cover this geometry"
            )

    def release_transients(self) -> int:
        """Deallocate the chunk transients ``lazy`` built under live traces; returns how many were released."""
        n = 0
        for t in self._transients:
            if getattr(t, "is_allocated", None) is not None and t.is_allocated():
                ttnn.deallocate(t)
                n += 1
        self._transients.clear()
        self._transient_memo.clear()
        return n

    # ------------------------------------------------------------------ readback
    def device_tensors(self, x: ttnn.Tensor) -> list[ttnn.Tensor]:
        """Per-chip tensors in mesh (row-major) order."""
        return list(ttnn.get_device_tensors(x))

    def to_host(self, x: ttnn.Tensor, *, rows: int | None = None, cols: int | None = None) -> torch.Tensor:
        """Host tensor: chip shards concatenated on ``rows`` / ``cols`` (``None`` = take the first
        replica on that axis). Equal replicas are the caller's assertion, not checked here. Only the
        shards the result uses are read: the first column of every row when ``cols`` is None, row 0
        only when ``rows`` is None (a per-row result on the 8x4 is 8 chip reads, not 32)."""
        devs = self.device_tensors(x)
        rr = range(self.rows) if rows is not None else range(1)
        cc = range(self.cols) if cols is not None else range(1)
        read = {(r, c): ttnn.to_torch(devs[r * self.cols + c]) for r in rr for c in cc}
        row_parts = [torch.cat([read[(r, c)] for c in cc], dim=cols) if cols is not None else read[(r, 0)] for r in rr]
        return torch.cat(row_parts, dim=rows) if rows is not None else row_parts[0]

    # ------------------------------------------------------------------ collectives
    @property
    def ccl(self):
        """Persistent CCL semaphores (``models.common.tt_ccl.TT_CCL``) for the async collectives."""
        if self._ccl is None:
            try:  # tt-metal main keeps tt_ccl as models.common.tt_ccl (TTTv2 moved to tt-transformers)
                from models.common.tt_ccl import get_tt_ccl
            except ImportError:  # the 2026-10 base still has it under models.common.modules
                from models.common.modules.tt_ccl import get_tt_ccl

            object.__setattr__(self, "_ccl", get_tt_ccl(self.device))  # frozen dataclass
        return self._ccl

    @property
    def async_ccl(self) -> bool:
        """``GLM53_CCL=async`` (default): single-op ``*_async`` gathers / reduce-scatters with persistent
        semaphores; ``generic``: the composite ``ttnn.all_gather`` / ``ttnn.reduce_scatter`` (on 8 rows the
        composites expand to all_broadcast + 8 untilizes + concat + tilize / 8 fill_pads + concat + rs)."""
        return os.environ.get("GLM53_CCL", "async") == "async"

    @contextmanager
    def prefill_scope(self):
        """Marks a prefill in flight: ``all_reduce_cols`` keeps the composite all-reduce inside it whatever the
        shape (a 128-token prompt on 8 rows is 16-row slabs, one tile: the shape gate alone cannot tell a prefill
        slab from a decode batch, and the gather mode moves 3 partials per chip where the composite moves 1.5;
        measured 2026-10-26 on the 8x4: 128-token TTFT 266 -> 314 ms with the gather mode in the slabs). Host
        state read per call; a captured trace replays the mode it was captured in (decoder prefill_forward and the
        MTP prefill enter it, so every prefill path is covered)."""
        self._prefill_depth[0] += 1
        try:
            yield
        finally:
            self._prefill_depth[0] -= 1

    @property
    def in_prefill(self) -> bool:
        return self._prefill_depth[0] > 0

    @property
    def tp_allreduce_gather(self) -> bool:
        """``GLM53_TP_ALLREDUCE=ccl`` (default): ``all_reduce_cols`` is the generic ``ttnn.all_reduce`` composite
        (reduce-scatter + all-gather over the row's columns: 51 / 30 us per call on the 8x4 at 32 users, the
        decode ledger). ``gather``: one ``all_gather_cols`` of the partials on a singleton outer dim + one
        ``fast_reduce_nc`` over it (B7, ~23 us): the sum is taken locally in fp32 DST and rounded to bf16 once,
        and every column chip sums the same gathered tensor in the same order, so the row's replicas are
        bit-identical. Read per call like ``async_ccl``; the choice is a function of the env and the (static)
        shape, so a captured trace replays the mode it was captured in. Decode shapes only (<= 32 rows: at
        prefill widths the gather moves 3 partials per chip where the composite moves 1.5)."""
        return not self.in_prefill and os.environ.get("GLM53_TP_ALLREDUCE", "ccl") == "gather"

    @property
    def reduce_ckc(self):
        """Compute config of the gather-mode all-reduce's ``fast_reduce_nc``: fp32 DST accumulation (the bf16
        partials are added exactly, one rounding on pack; HiFi4 is the op's own default and irrelevant to adds)."""
        if self._reduce_ckc is None:
            object.__setattr__(  # frozen dataclass
                self,
                "_reduce_ckc",
                ttnn.init_device_compute_kernel_config(
                    self.device.arch(),
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                ),
            )
        return self._reduce_ckc

    def _async_topology(self):
        return self.topology if self.topology is not None else ttnn.Topology.Linear

    def _all_gather_async(self, x, dim, axis, memory_config):
        ccl = self.ccl
        return ttnn.experimental.all_gather_async(
            x,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=ccl.get_and_cycle_ag_semaphore_handles(axis),
            num_links=ccl.get_num_links(axis),
            cluster_axis=axis,
            topology=self._async_topology(),
            memory_config=memory_config,
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(axis),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def _reduce_scatter_async(self, x, dim, axis, memory_config):
        ccl = self.ccl
        return ttnn.experimental.reduce_scatter_minimal_async(
            x,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=ccl.get_and_cycle_rs_semaphore_handles(axis),
            barrier_semaphore=ccl.get_and_cycle_barrier_semaphore_handle(axis),
            num_links=ccl.get_num_links(axis),
            cluster_axis=axis,
            memory_config=memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=self._async_topology(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    def _ccl_kwargs(self, num_links=None):
        kw = {}
        if self.topology is not None:
            kw["topology"] = self.topology
        if num_links is not None:
            kw["num_links"] = num_links
        return kw

    def partition_cols(self, x: ttnn.Tensor, dim: int) -> ttnn.Tensor:
        """Each column chip keeps its own ``1/cols`` slice of ``dim`` of a column-replicated tensor (a per-device
        slice, no fabric traffic): the user-partitioned DSA cache's "my users" of a row-wide activation. A TILE
        tensor whose per-column part is not tile-aligned on a tile dim (4 users split 4 ways on dim 2) goes
        through ROW_MAJOR: ``mesh_partition`` refuses a tiled split that would create padding."""
        if self.cols == 1:
            return x
        d = dim % len(x.shape)
        if x.layout == ttnn.TILE_LAYOUT and d >= len(x.shape) - 2 and (x.shape[d] // self.cols) % ttnn.TILE_SIZE:
            rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            part = ttnn.mesh_partition(rm, dim, cluster_axis=TP_AXIS)
            ttnn.deallocate(rm)
            out = ttnn.to_layout(part, ttnn.TILE_LAYOUT)
            ttnn.deallocate(part)
            return out
        return ttnn.mesh_partition(x, dim, cluster_axis=TP_AXIS)

    def all_to_all_cols(self, x: ttnn.Tensor, in_dim: int, out_dim: int, *, memory_config=None) -> ttnn.Tensor:
        """Redistribute over the row's columns: ``in_dim`` is split into ``cols`` chunks (chunk c goes to column
        c) and the received chunks are concatenated along ``out_dim`` (heads <-> users before / after the
        user-partitioned attention, as DeepSeek-V3's decode does)."""
        if self.cols == 1:
            return x
        # the op's convention is the reverse of this helper's: it SPLITS ``out_dim`` over the devices and GROWS
        # ``in_dim`` (output[in_dim] *= cols, output[out_dim] /= cols), so the arguments are swapped here. The
        # grown dim may sit on the tile-row dim only if its per-device size is a multiple of 16 (16 heads per
        # column qualifies); the split dim must not be the tile-row dim unless its size is a multiple of 16.
        return ttnn.experimental.all_to_all_async_generic(
            x,
            in_dim=out_dim,
            out_dim=in_dim,
            num_links=self.ccl.get_num_links(TP_AXIS),
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=TP_AXIS,
        )

    def all_to_all_rows(self, x: ttnn.Tensor, in_dim: int, out_dim: int, *, memory_config=None) -> ttnn.Tensor:
        """``all_to_all_cols`` along the mesh rows (axis 0): ``in_dim`` is split into ``rows`` chunks (chunk r goes
        to row r) and the received chunks are stacked along ``out_dim`` in sender-row order. TILE tensors only; a
        split of the tile-row dim needs a per-row chunk that is a multiple of 16 rows (the op's padding rule). Used
        by the EP combine (``TtEPExperts._combine_rows_a2a``: every expert chip's output rows back to their rows)."""
        if self.rows == 1:
            return x
        return ttnn.experimental.all_to_all_async_generic(
            x,
            in_dim=out_dim,
            out_dim=in_dim,
            num_links=self.ccl.get_num_links(DP_AXIS),
            memory_config=memory_config or ttnn.DRAM_MEMORY_CONFIG,
            cluster_axis=DP_AXIS,
        )

    def all_reduce_cols(self, x: ttnn.Tensor, *, memory_config=None, num_links=None) -> ttnn.Tensor:
        """Sum the TP partials of a row; result replicated over the row's columns (same shape, dtype and memory
        config as ``x`` unless ``memory_config`` is given). ``GLM53_TP_ALLREDUCE=gather`` (``tp_allreduce_gather``)
        takes ``_all_reduce_cols_gather`` for the inputs it covers (``_gather_reduce_dim``); ``num_links`` then is
        the gather's own link count (``TT_CCL``)."""
        if self.cols == 1:
            return x
        memory_config = memory_config or x.memory_config()
        if self.tp_allreduce_gather:
            dim = self._gather_reduce_dim(x)
            if dim is not None:
                return self._all_reduce_cols_gather(x, dim, memory_config)
        if self.tp_allreduce_input_l1 and self._gather_reduce_dim(x) is not None:
            # C2 experiment: the composite reads its input from L1 interleaved instead of DRAM (decode shapes only)
            x_l1 = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            y = ttnn.all_reduce(x_l1, cluster_axis=TP_AXIS, memory_config=memory_config, **self._ccl_kwargs(num_links))
            ttnn.deallocate(x_l1)
            return y
        return ttnn.all_reduce(x, cluster_axis=TP_AXIS, memory_config=memory_config, **self._ccl_kwargs(num_links))

    @property
    def tp_allreduce_input_l1(self) -> bool:
        """``GLM53_TP_ALLREDUCE_INPUT=l1`` (C2 experiment; default ``dram`` = today's path): the composite TP all-reduce
        takes its decode input from L1 interleaved. Read per call like ``tp_allreduce_gather`` (env + static shape)."""
        return not self.in_prefill and os.environ.get(TP_ALLREDUCE_INPUT_ENV, "dram") == "l1"

    @staticmethod
    def _gather_reduce_dim(x: ttnn.Tensor) -> int | None:
        """The singleton outer dim the gather-mode all-reduce stacks the column partials on: dim 1 of a 4-D
        interleaved TILE tensor of at most ``TP_ALLREDUCE_GATHER_ROWS`` rows (``[1, 1, T, H]``, the decode
        activations), dim 0 when only that one is singleton; ``None`` for everything else (other ranks, sharded
        or ROW_MAJOR inputs, prefill widths): the caller keeps the composite. A pure function of the static shape
        and layout: trace-safe."""
        shape = x.shape
        if len(shape) != 4 or x.layout != ttnn.TILE_LAYOUT or x.is_sharded():
            return None
        if shape[2] > TP_ALLREDUCE_GATHER_ROWS:
            return None
        if shape[1] == 1:
            return 1
        if shape[0] == 1:
            return 0
        return None

    def _all_reduce_cols_gather(self, x: ttnn.Tensor, dim: int, memory_config) -> ttnn.Tensor:
        """``all_reduce_cols`` as gather + local sum: the ``cols`` partials stacked on the singleton ``dim`` by one
        all-gather (DRAM: ``[1, cols, T, H]`` for dim 1, chunk c = column c's partial), summed over ``dim`` by
        ``fast_reduce_nc`` (fp32 DST, ``reduce_ckc``) into a tensor of ``x``'s shape / dtype in ``memory_config``.
        ``fast_reduce_nc`` reports the tile-padded row count of a short batch (4 users/row -> 32 rows), so below a
        tile the sum is written into a preallocated output of the logical shape (a device-side allocation, no
        slice op, trace-safe: the op keeps a preallocated output's spec). Two device ops, no host ops."""
        g = self.all_gather_cols(x, dim, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        output = None
        if x.shape[2] % ttnn.TILE_SIZE:
            output = ttnn.allocate_tensor_on_device(
                ttnn.Shape(list(x.shape)), x.dtype, ttnn.TILE_LAYOUT, self.device, memory_config
            )
        y = ttnn.experimental.fast_reduce_nc(
            g, dims=[dim], output=output, memory_config=memory_config, compute_kernel_config=self.reduce_ckc
        )
        ttnn.deallocate(g)
        return y

    def all_gather_cols(self, x: ttnn.Tensor, dim: int, *, memory_config=None) -> ttnn.Tensor:
        if self.cols == 1:
            return x
        if self.async_ccl:
            return self._all_gather_async(x, dim, TP_AXIS, memory_config or x.memory_config())
        return ttnn.all_gather(x, dim=dim, cluster_axis=TP_AXIS, memory_config=memory_config or x.memory_config())

    def all_gather_rows(self, x: ttnn.Tensor, dim: int, *, memory_config=None) -> ttnn.Tensor:
        """Concatenate the rows' shards along ``dim`` (rows ordered by mesh row)."""
        if self.rows == 1:
            return x
        if self.async_ccl:
            return self._all_gather_async(x, dim, DP_AXIS, memory_config or x.memory_config())
        return ttnn.all_gather(x, dim=dim, cluster_axis=DP_AXIS, memory_config=memory_config or x.memory_config())

    def reduce_scatter_rows(self, x: ttnn.Tensor, dim: int, *, memory_config=None, num_links=None) -> ttnn.Tensor:
        """Sum over the rows and hand row ``r`` the ``r``-th slice of ``dim``."""
        if self.rows == 1:
            return x
        if self.async_ccl:
            return self._reduce_scatter_async(x, dim, DP_AXIS, memory_config or x.memory_config())
        return ttnn.reduce_scatter(
            x,
            dim=dim,
            cluster_axis=DP_AXIS,
            memory_config=memory_config or x.memory_config(),
            **self._ccl_kwargs(num_links),
        )

    def all_reduce_rows(self, x: ttnn.Tensor, *, memory_config=None, num_links=None) -> ttnn.Tensor:
        """Sum over the rows, result replicated over the rows: the generic ``ttnn.all_reduce`` composite (its own
        reduce-scatter + all-gather pair, semaphores allocated by the composite). ``all_reduce_rows_rs_ag`` is the
        same sum through the model's persistent-semaphore async pair."""
        if self.rows == 1:
            return x
        return ttnn.all_reduce(
            x, cluster_axis=DP_AXIS, memory_config=memory_config or x.memory_config(), **self._ccl_kwargs(num_links)
        )

    def all_reduce_rows_rs_ag(self, x: ttnn.Tensor, *, dim: int = 3, memory_config=None) -> ttnn.Tensor:
        """Sum over the rows as ``reduce_scatter_rows(x, dim)`` + ``all_gather_rows(., dim)`` (the async ops with the
        persistent TT_CCL semaphores, Linear topology, as every row collective of the decode step): ``dim`` must split
        tile-aligned over the rows (the hidden dim of a [1,1,T,4096] bf16 activation: 512 columns per row on the 8x4,
        2048 on a (2,4) block), so no composite expansion happens whatever T is. The hybrid MoE's compact combine
        (``TtRoutedExperts._combine_rows_allreduce``, ``GLM53_HYBRID_COMBINE=allreduce``) sums the [1,1,rows*T,H]
        expert output with it: 2 x (7/8 x 256 KB) on the links instead of the padded [1,1,256,H] reduce-scatter's
        1.75 MB. Rounding: the chunk of hidden columns owned by row r is reduced in that chunk's ring order (bf16 per
        hop), the all-gather copies exactly; a different order than the row-chunk reduce-scatter, same partials."""
        if self.rows == 1:
            return x
        d = dim % len(x.shape)
        assert d >= len(x.shape) - 2, (dim, "the rs + ag form is for the tile dims")
        tile = ttnn.TILE_SIZE
        assert (x.shape[d] // tile) % self.rows == 0, (tuple(x.shape), d, self.rows, "chunk must be tile-aligned")
        memory_config = memory_config or x.memory_config()
        part = self.reduce_scatter_rows(x, dim=d, memory_config=memory_config)  # [.., H / rows]
        out = self.all_gather_rows(part, dim=d, memory_config=memory_config)
        ttnn.deallocate(part)
        return out


def single_chip(device) -> GLMMesh:
    """The stage-1 layout: everything local, collectives are identities."""
    return GLMMesh(device, 1, 1)


def interleave_blocks(parts: list[torch.Tensor], n: int, dim: int = -1) -> torch.Tensor:
    """Concatenate ``parts`` so that an ``n``-way split of ``dim`` hands shard ``i`` the ``i``-th
    ``n``-th of every part, in order: ``[p0_0 | p1_0 | ... | p0_1 | p1_1 | ...]``. Used for the
    fused ``[gate | up]`` and ``[q | k | v]`` projections so each TP chip keeps a complete
    ``gate_i | up_i`` (``q_i | k_i | v_i``) block of its own slice."""
    chunks = [p.chunk(n, dim=dim) for p in parts]
    return torch.cat([c for i in range(n) for c in (cc[i] for cc in chunks)], dim=dim).contiguous()


def selector_tiles() -> torch.Tensor:
    """The [33*32, 32] constant of the fused decode programs (``kda_decode``, ``dsa_decode_state``): tile r < 32
    has a 1 at (0, r) (E_r lifts row r of a block into row 0; its transpose places row 0 at row r), tile 32 is all
    ones (row / column sums by matmul)."""
    sel = torch.zeros(33, 32, 32)
    sel[torch.arange(32), 0, torch.arange(32)] = 1.0
    sel[32] = 1.0
    return sel.reshape(33 * 32, 32)
