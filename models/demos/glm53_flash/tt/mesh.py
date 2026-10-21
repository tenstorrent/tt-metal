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
Galaxy); every collective must run once eagerly before a trace is captured.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch

import ttnn

TP_AXIS = 1  # columns
DP_AXIS = 0  # rows (and expert-TP)


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


@dataclass(frozen=True)
class GLMMesh:
    """A mesh device plus the axis convention. ``tp`` = columns, ``dp`` = rows."""

    device: ttnn.MeshDevice
    rows: int
    cols: int
    topology: ttnn.Topology | None = None  # None = ttnn default for the fabric
    _ccl: object = field(default=None, repr=False, compare=False)  # lazy TT_CCL (async collectives)
    # device constants shared by every module of the model (``const``): one tensor per key instead of one per layer
    _consts: dict = field(default_factory=dict, repr=False, compare=False)

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
        clash with the L1 buffers (the EP32 8x4 e2e, doc/mesh/README.md)."""
        if key not in self._consts:
            self._consts[key] = make()
        return self._consts[key]

    # ------------------------------------------------------------------ readback
    def device_tensors(self, x: ttnn.Tensor) -> list[ttnn.Tensor]:
        """Per-chip tensors in mesh (row-major) order."""
        return list(ttnn.get_device_tensors(x))

    def to_host(self, x: ttnn.Tensor, *, rows: int | None = None, cols: int | None = None) -> torch.Tensor:
        """Host tensor: chip shards concatenated on ``rows`` / ``cols`` (``None`` = take the first
        replica on that axis). Equal replicas are the caller's assertion, not checked here."""
        shards = [ttnn.to_torch(d) for d in self.device_tensors(x)]
        grid = [[shards[r * self.cols + c] for c in range(self.cols)] for r in range(self.rows)]
        row_parts = []
        for r in range(self.rows):
            row_parts.append(torch.cat(grid[r], dim=cols) if cols is not None else grid[r][0])
        return torch.cat(row_parts, dim=rows) if rows is not None else row_parts[0]

    # ------------------------------------------------------------------ collectives
    @property
    def ccl(self):
        """Persistent CCL semaphores (``models.common.modules.tt_ccl.TT_CCL``) for the async collectives."""
        if self._ccl is None:
            from models.common.modules.tt_ccl import get_tt_ccl

            object.__setattr__(self, "_ccl", get_tt_ccl(self.device))  # frozen dataclass
        return self._ccl

    @property
    def async_ccl(self) -> bool:
        """``GLM53_CCL=async`` (default): single-op ``*_async`` gathers / reduce-scatters with persistent
        semaphores; ``generic``: the composite ``ttnn.all_gather`` / ``ttnn.reduce_scatter`` (on 8 rows the
        composites expand to all_broadcast + 8 untilizes + concat + tilize / 8 fill_pads + concat + rs)."""
        return os.environ.get("GLM53_CCL", "async") == "async"

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

    def all_reduce_cols(self, x: ttnn.Tensor, *, memory_config=None, num_links=None) -> ttnn.Tensor:
        """Sum the TP partials of a row; result replicated over the row's columns."""
        if self.cols == 1:
            return x
        return ttnn.all_reduce(
            x, cluster_axis=TP_AXIS, memory_config=memory_config or x.memory_config(), **self._ccl_kwargs(num_links)
        )

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
        if self.rows == 1:
            return x
        return ttnn.all_reduce(
            x, cluster_axis=DP_AXIS, memory_config=memory_config or x.memory_config(), **self._ccl_kwargs(num_links)
        )


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
