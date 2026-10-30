# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""The served KV caches' storage (serving_contract.md "KV cache", "The record rule"): one tensor per table config,
[slots x layers, 1, local rows, width] bf16 ROW_MAJOR in DRAM, ND-sharded with shard [1, 1, 32 / r, width] (one
32-token record), ROUND_ROBIN_1D over every DRAM bank, as DeepSeek's ``init_kvpe_cache`` (which fixes the shard at 32
rows; a ratio-2 record is 16 compressed rows of 512). Batch row b = slot x layers + layer row (the order
``update_padded_kv_cache`` writes: users outer, layers inner).

``CacheSlab`` is one (slot, layer) of such a tensor: the window cache (tt/attention.TtWindowCache) and the compressed
caches (tt/compressor.TtServedRows) read and write their slab through it. The ladder's state allocates a tensor of
one slab per cache (slots = layers = 1); the serving runtime allocates the engine-owned tensors once
(tt/runners/kv_contract.py) and binds every slot's state to its slabs. Same layout, same code, one format.

``DeviceScalars``: load-time uint32 tables [n, 1] (one value per page) the compressed caches slice per chunk for
``update_padded_kv_cache``'s metadata path (slot, start, valid as 1-element device tensors), which works in page rows
(a ratio-2 start is a multiple of 16 rows; the scalar path wants 32).
"""

from __future__ import annotations

from collections import OrderedDict

import torch

import ttnn

from .ccl import dealloc
from .common import DRAM, SP_AXIS, TP_AXIS


def alloc_served(mesh, width: int, rows: int, shard_rows: int, batches: int, dedup: bool) -> ttnn.Tensor:
    """[batches, 1, rows, width] bf16 ROW_MAJOR per chip, zeroed, DRAM ND-sharded [1, 1, shard_rows, width]
    ROUND_ROBIN_1D over every DRAM bank. ``dedup``: TP-dedup (rows hold the chip's own sp x tp stripes; topology
    sharded on dim 2 over both axes, as init_kvpe_cache(tp_axis=1)); else SP-sharded and TP-replicated (replicated
    topology, as init_kvpe_cache(tp_axis=None))."""
    from models.demos.common.prefill.runners.migration import get_num_dram_banks
    from models.demos.deepseek_v3_b1.micro_ops.dram_zero_fill.op import DRAMZeroFill

    assert rows % shard_rows == 0, (rows, shard_rows)
    banks = get_num_dram_banks(mesh)
    grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(b, 0), ttnn.CoreCoord(b, 0)) for b in range(banks)])
    spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, shard_rows, width],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mc = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=spec)
    t = ttnn.allocate_tensor_on_device(
        ttnn.Shape([batches, 1, rows, width]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, mesh, mc
    )
    DRAMZeroFill.op(t)
    phys = ttnn.MeshShape(mesh.shape[0], mesh.shape[1])
    coords = [ttnn.MeshCoordinate([c[i] for i in range(c.dims())]) for c in ttnn.MeshCoordinateRange(phys)]
    if dedup and mesh.shape[TP_AXIS] > 1:
        topo = ttnn.TensorTopology(phys, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
    else:
        n = mesh.shape[0] * mesh.shape[1]
        topo = ttnn.TensorTopology(ttnn.MeshShape([n]), [ttnn.PlacementReplicate()], coords)
    t.update_tensor_topology(topo)
    return t


class CacheSlab:
    """One (slot, layer row) of a served-cache tensor [slots x layers, 1, rows, width]."""

    __slots__ = ("tensor", "slot", "row", "layers", "owned")

    def __init__(self, tensor: ttnn.Tensor, slot: int = 0, row: int = 0, layers: int = 1, owned: bool = False):
        """``owned``: the tensor is this cache's own (freed with it); else it belongs to the serving engine."""
        assert 0 <= row < layers and (slot + 1) * layers <= tensor.shape[0], (slot, row, layers, tensor.shape)
        self.tensor, self.slot, self.row, self.layers = tensor, int(slot), int(row), int(layers)
        self.owned = bool(owned)

    @property
    def batch(self) -> int:
        return self.slot * self.layers + self.row

    @property
    def rows(self) -> int:
        return int(self.tensor.shape[-2])

    @property
    def width(self) -> int:
        return int(self.tensor.shape[-1])

    def slice_rows(self, lo: int, hi: int, out: ttnn.Tensor | None = None) -> ttnn.Tensor:
        """Local rows [lo, hi) of this slab -> [1, 1, hi - lo, width] ROW_MAJOR, DRAM interleaved (a new tensor;
        ``out``: written into that tensor and returned, HOST_ALLOC=on, P.15)."""
        b = self.batch
        if out is not None:
            ttnn.slice(self.tensor, [b, 0, lo, 0], [b + 1, 1, hi, self.width], memory_config=DRAM, output_tensor=out)
            return out
        return ttnn.slice(self.tensor, [b, 0, lo, 0], [b + 1, 1, hi, self.width], memory_config=DRAM)

    def host_rows(self) -> list[torch.Tensor]:
        """This slab's local rows [rows, width] of every chip (chip-linear r TP + c), host."""
        sl = self.slice_rows(0, self.rows)
        out = [ttnn.to_torch(t).reshape(self.rows, self.width) for t in ttnn.get_device_tensors(sl)]
        dealloc(sl)
        return out

    @property
    def slots(self) -> int:
        return int(self.tensor.shape[0]) // self.layers

    def free(self) -> None:
        if self.owned and self.tensor.is_allocated():
            dealloc(self.tensor)


class DeviceScalars:
    """uint32 values as a load-time device table [n, 1] ROW_MAJOR (one value per DRAM page), replicated; ``at(k)``
    slices the k-th value as a 1-element device tensor of its own (a device op, no host transfer; the caller frees
    it). The table has at least 2 rows, so a slice is never full-range (DS41-52)."""

    MEMO = 8  # GLUE_FOLD=on (P.19): the latest values sliced, kept for the chunk's other writes

    def __init__(self, mesh, values, memo: bool = False):
        self._memo = OrderedDict() if memo else None
        v = torch.as_tensor(list(values), dtype=torch.int64).reshape(-1, 1)
        self.n = v.shape[0]
        if self.n < 2:  # a one-row slice of a one-row table is full-range: ttnn.slice returns the table's own buffer,
            v = torch.cat([v, torch.zeros(2 - self.n, 1, dtype=torch.int64)])  # which the caller frees (DS41-52)
        self.t = ttnn.from_torch(
            v.to(torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    def at(self, k: int) -> ttnn.Tensor:
        """The k-th value; release it with ``release`` (memo: kept while it is among the latest MEMO)."""
        assert 0 <= k < self.n, (k, self.n)
        if self._memo is None:
            return ttnn.slice(self.t, [k, 0], [k + 1, 1], memory_config=DRAM)
        t = self._memo.get(k)
        if t is not None:
            self._memo.move_to_end(k)
            return t
        t = self._memo[k] = ttnn.slice(self.t, [k, 0], [k + 1, 1], memory_config=DRAM)
        while len(self._memo) > self.MEMO:
            dealloc(self._memo.popitem(last=False)[1])
        return t

    def release(self, t: ttnn.Tensor) -> None:
        if self._memo is None or not any(t is u for u in self._memo.values()):
            dealloc(t)


def sp_tp(mesh) -> tuple[int, int]:
    return mesh.shape[SP_AXIS], mesh.shape[TP_AXIS]
