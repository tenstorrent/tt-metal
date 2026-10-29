# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Resident decode (``GLM53_RESIDENT_DECODE=1``, ``doc/mesh/resident_decode.md``): the traced decode step feeds its
own sampled ids back and advances the live slots' positions ON DEVICE, so a steady-state step is a replay with no
host upload (the plugin's ``reload_inputs=False`` chain under ``supports_async_decode``). The host writes the inputs
only on a transition (a chain start, a changed decode layout): ``host_load``.

The captured step ends with two in-graph updates after the head (``GLM53MeshModel.decode_step``):

* ``feed_ids``: the head's greedy ids ``[1, 1, B, 1]`` uint32 (one per user of the row, replicated over the columns)
  -> the persistent ``ids_dev`` ``[1, 1, 1, B]`` uint32 ROW_MAJOR the next replay embeds;
* ``advance``: ``pos_f += live_f`` on a persistent fp32 TILE shadow of the positions (exact: positions < 2^24), then
  every position-derived field of ``DecodePositions`` / ``SparseDecodeMeta`` recomputed from it with the host
  formulas of ``tt/packed_inputs.py`` (``position_fields`` / ``sparse_host_values``) and copied into the EXISTING
  persistent int32 / uint32 ROW_MAJOR field tensors (the consumers never change). A parked slot (``live`` 0) keeps
  its position; its fed-back id is garbage the host never reads, as today.

Every value stays an exact integer in fp32 (floor / relu / where on integer-valued operands; ``index_kpool`` is a
power of two so the divisions are exact; fp32 operands keep the fp32 destination on Blackhole), every transient
lives in DRAM (the ``PackedDecodeInputs.unpack`` note). The per-user ``valid_len`` list
(``GLM53_TOPK_BATCHED=0``) and the live pool bound (``GLM53_SPARSE_DECODE_BOUND=1``) are not derived here: resident
decode refuses those knobs.
"""
from __future__ import annotations

import os

import torch

import ttnn
from models.demos.glm53_flash.tt.dsa import SPARSE_BOUND_ENV, SparseDecodeMeta, dense_exact_max_pos
from models.demos.glm53_flash.tt.functional_decoder import DecodePositions
from models.demos.glm53_flash.tt.mesh import GLMMesh
from models.demos.glm53_flash.tt.packed_inputs import RESIDENT_ENV, resident_decode_enabled  # noqa: F401 (re-export)


class ResidentDecodeFeedback:
    """The persistent shadow inputs and the in-graph feedback of the resident decode step (module docstring)."""

    def __init__(self, mesh: GLMMesh, cfg, positions: DecodePositions, ids_dev: ttnn.Tensor, *, partition: bool):
        if positions.sparse is None:
            raise ValueError("resident decode needs the sparse-decode metadata (SparseDecodeMeta) on the positions")
        if os.environ.get("GLM53_TOPK_BATCHED", "1") != "1":
            raise NotImplementedError(
                "resident decode does not derive the per-user valid_len list (GLM53_TOPK_BATCHED=0)"
            )
        if os.environ.get(SPARSE_BOUND_ENV, "0") != "0":
            raise NotImplementedError(f"resident decode does not derive the live pool bound ({SPARSE_BOUND_ENV}=1)")
        self.mesh = mesh
        self.cfg = cfg
        self.partition = bool(partition)
        self.P = int(cfg.index_kpool)
        self.K = int(cfg.index_select_pools)
        self.dense_max = int(dense_exact_max_pos(cfg))
        if self.P & (self.P - 1):
            raise ValueError(f"index_kpool {self.P} is not a power of two: the fp32 divisions would not be exact")
        R, C = SparseDecodeMeta._chips(mesh, self.partition)
        self.total = len(positions.host)
        if self.total % (R * C):
            raise ValueError(f"{self.total} user slots over {R} x {C} chip groups")
        self.users_per_chip = self.total // (R * C)
        self.users_per_row = self.total // mesh.rows
        self.positions = positions
        self.ids_dev = ids_dev
        self.live_host = [True] * self.total
        # the fp32 TILE shadows: [1, 1, 1, users_per_chip] per chip, placed like the position fields
        self.pos_f = self._upload([float(p) for p in positions.host])
        self.live_f = self._upload([1.0] * self.total)

    # ------------------------------------------------------------------------------------------ host side
    def _mapper(self):
        cols = 3 if self.partition and self.mesh.cols > 1 else None
        return self.mesh.shard_2d(3, cols)

    def _upload(self, values: list[float]) -> ttnn.Tensor:
        t = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)
        return self.mesh.upload(
            t, ttnn.float32, rows=3, cols=3 if self.partition and self.mesh.cols > 1 else None, layout=ttnn.TILE_LAYOUT
        )

    def _refresh(self, dst: ttnn.Tensor, values: list[float]) -> None:
        t = torch.tensor(values, dtype=torch.float32).reshape(1, 1, 1, -1)
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._mapper()), dst
        )

    def host_load(self, host_positions: list[int], live: list[bool] | None) -> None:
        """A transition: the shadows take the host's positions and live mask (the persistent field tensors and
        ``ids_dev`` are written by the caller's reload: ``GLM53MeshModel.load_decode_inputs``)."""
        if len(host_positions) != self.total:
            raise ValueError(f"{len(host_positions)} positions for {self.total} user slots")
        live = [True] * self.total if live is None else [bool(v) for v in live]
        if len(live) != self.total:
            raise ValueError(f"{len(live)} live flags for {self.total} user slots")
        self._refresh(self.pos_f, [float(p) for p in host_positions])
        self._refresh(self.live_f, [1.0 if v else 0.0 for v in live])
        self.live_host = live

    # ---------------------------------------------------------------------------------------- device side
    def _put(self, v: ttnn.Tensor, dst: ttnn.Tensor, dtype, *, column: bool) -> None:
        """``v`` [1,1,1,Bc] fp32 TILE (integer-valued) -> the persistent ``dst`` (ROW_MAJOR, ``dtype``): a row
        field keeps the shape, a column field ([Bc,1,...]) goes through a transpose; the final reshape is a view
        (the last dim is unchanged)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        temps = []
        if column:
            v = ttnn.transpose(v, -2, -1, memory_config=dram)
            temps.append(v)
        vi = ttnn.typecast(v, dtype, memory_config=dram)
        vr = ttnn.to_layout(vi, ttnn.ROW_MAJOR_LAYOUT, memory_config=dram)
        temps += [vi, vr]
        shape = list(dst.shape)
        view = vr if list(vr.shape) == shape else ttnn.reshape(vr, shape)
        ttnn.copy(view, dst)
        for t in temps:
            ttnn.deallocate(t)

    def feed_ids(self, ids: ttnn.Tensor) -> None:
        """The head's greedy ids ``[1, 1, B, 1]`` uint32 ROW_MAJOR -> ``ids_dev`` ``[1, 1, 1, B]`` (through fp32 for
        the transpose: ids < 2^24)."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        t = ttnn.to_layout(ids, ttnn.TILE_LAYOUT, memory_config=dram)
        f = ttnn.typecast(t, ttnn.float32, memory_config=dram)
        ttnn.deallocate(t)
        self._put(f, self.ids_dev, ttnn.uint32, column=True)  # [1,1,B,1] -> [1,1,1,B]
        ttnn.deallocate(f)

    def advance(self) -> None:
        """``pos_f += live_f``; every derived field from the new positions into the persistent tensors."""
        dram = ttnn.DRAM_MEMORY_CONFIG
        P, K, Dm = float(self.P), float(self.K), float(self.dense_max)
        ttnn.add(self.pos_f, self.live_f, output_tensor=self.pos_f)
        pos = self.pos_f
        a1 = ttnn.add(pos, 1.0, memory_config=dram)
        n = ttnn.floor(ttnn.multiply(a1, 1.0 / P, memory_config=dram), memory_config=dram)  # (a + 1) // P
        t = ttnn.subtract(a1, ttnn.multiply(n, P, memory_config=dram), memory_config=dram)  # (a + 1) % P
        pool = ttnn.floor(ttnn.multiply(pos, 1.0 / P, memory_config=dram), memory_config=dram)  # a // P
        ring = ttnn.subtract(pos, ttnn.multiply(pool, P, memory_config=dram), memory_config=dram)  # a % P
        # min(a, Dm) = Dm - relu(Dm - a)
        dense = ttnn.rsub(ttnn.relu(ttnn.rsub(pos, Dm, memory_config=dram), memory_config=dram), Dm, memory_config=dram)
        is_dense = ttnn.lt(n, K, memory_config=dram)  # n < K: the dense user
        cur_sparse = ttnn.add(t, P * K - 1.0, memory_config=dram)
        cur = ttnn.where(is_dense, pos, cur_sparse, memory_config=dram)
        valid = ttnn.add(ttnn.relu(ttnn.subtract(n, K, memory_config=dram), memory_config=dram), K, memory_config=dram)
        dp, sp = self.positions, self.positions.sparse
        self._put(pos, dp.positions, ttnn.int32, column=False)
        self._put(pool, dp.pool_ids, ttnn.int32, column=False)
        self._put(ring, dp.ring_slots, ttnn.int32, column=False)
        if dp.dense_positions is not None:
            self._put(dense, dp.dense_positions, ttnn.int32, column=False)
        self._put(cur, sp.cur_pos, ttnn.int32, column=False)
        self._put(n, sp.tail_pool, ttnn.uint32, column=True)
        if sp.valid_len_rows is not None:
            self._put(valid, sp.valid_len_rows, ttnn.uint32, column=True)
        for x in (a1, n, t, pool, ring, dense, is_dense, cur_sparse, cur, valid):
            ttnn.deallocate(x)

    def advance_host(self, live: list[bool] | None = None) -> list[int]:
        """The host mirror of ``advance`` (the caller writes it into ``DecodePositions.host`` through
        ``update_decode_positions(upload=False)``): the new positions."""
        live = self.live_host if live is None else live
        return [p + (1 if v else 0) for p, v in zip(self.positions.host, live)]
