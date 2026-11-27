# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 image-embedding splice on the device: the image rows plan_media_splice (media_embedding.py) lays out per
SP row, and the lookup and where that put them over the token embeddings at a chunk's image positions."""

import itertools

import numpy as np
import torch

import ttnn
from models.demos.common.prefill.media_embedding import SplicePlan


class MediaSpliceBuffers:
    """Per-SP-row image rows, on device, for splice_media.

    Per chip: a ROW_MAJOR table [h, hidden/tp] (ttnn.embedding's weight layout) with row 0 zeros, and a lookup
    (0 = not this SP row's). With the vocab sharded over SP, lookup [1,1,s_total] and is_img [1,1,s_total,1] cover
    every gathered position and are replicated; by_position, they cover the chip's own s_local and are SP-sharded.

    lookup and is_img are allocated once. Each table height's device table and host table are allocated on first
    use and kept; refill writes the rows into the host table's shards in place, which is safe because
    copy_host_to_device_tensor is done with them when it returns. is_img is tilized on device, as ttnn.where needs
    a TILE condition; refilling frees the previous one, which relies on the queue completing in order. Keep the
    uploads on one queue.
    """

    def __init__(
        self, mesh_device, *, sp: int, hidden: int, s_total: int, mesh_shape, sp_axis, tp_axis, by_position: bool
    ):
        self.sp, self.hidden = sp, hidden
        self._mesh_device = mesh_device
        sp_tp, sp_only, seq_on_sp = [None, None], [None, None], [None, None]
        sp_tp[sp_axis], sp_tp[tp_axis] = 0, 1
        sp_only[sp_axis] = 0
        seq_on_sp[sp_axis] = 2
        self._sp_tp, self._sp_only = tuple(sp_tp), tuple(sp_only)
        self._is_img_dims = tuple(seq_on_sp) if by_position else (None, None)
        self._mapper = lambda dims: ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape)
        self._mesh_shape, self._axes, self.tp = tuple(mesh_shape), (sp_axis, tp_axis), mesh_shape[tp_axis]

        self._tables = {}
        self.table = None
        looked_up = s_total // sp if by_position else s_total
        self.lookup = self._to_device(
            torch.zeros(sp, 1, looked_up, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self._sp_only
        )
        self._is_img_rows = self._to_device(
            torch.zeros(1, 1, s_total, 1, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self._is_img_dims
        )
        self.is_img = ttnn.to_layout(self._is_img_rows, ttnn.TILE_LAYOUT)

    def _to_device(self, t, dtype, layout, dims):
        return ttnn.from_torch(
            t,
            device=self._mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mapper(dims),
        )

    def _host(self, t, dtype, layout, dims):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self._mapper(dims))

    def table_for(self, height: int) -> tuple:
        """(device table, host table, {(sp_row, tp_col): [h, hidden/tp] view of that chip's host shard}) of
        this height, allocated on first use. Only row 0 of each shard is zeroed: runs start at slot 1, and
        lookup never points above the slots a forward fills. The device table is allocated without data; refill's
        upload writes all of it, row 0 included."""
        entry = self._tables.get(height)
        if entry is None:
            empty = torch.empty(self.sp, self.tp * height, self.hidden // self.tp, dtype=torch.bfloat16)
            host = self._host(empty, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, self._sp_tp)
            shards = host.host_buffer()
            views = {}
            for coord in itertools.product(*map(range, self._mesh_shape)):
                shard = torch.from_dlpack(shards.get_shard(ttnn.MeshCoordinate(*coord)))
                view = shard.view(torch.int16).view(height, -1).numpy()
                view[0] = 0
                views[coord[self._axes[0]], coord[self._axes[1]]] = view
            device = ttnn.allocate_tensor_on_device(
                ttnn.Shape([1, height, self.hidden // self.tp]),
                ttnn.bfloat16,
                ttnn.ROW_MAJOR_LAYOUT,
                self._mesh_device,
                ttnn.DRAM_MEMORY_CONFIG,
            )
            entry = self._tables[height] = (device, host, views)
        return entry

    def release_tables(self) -> None:
        """Free every height's tables, and with each host table its pin."""
        for device, _, _ in self._tables.values():
            ttnn.deallocate(device)
        self._tables.clear()
        self.table = None

    # TODO(#66176): write each forward's image rows straight to each chip's shard, without the host table.
    def refill(self, plan: SplicePlan) -> None:
        self.table, host, views = self.table_for(plan.height)
        cols = self.hidden // self.tp
        for sp_row, slot, rows in plan.runs:
            for t in range(self.tp):
                views[sp_row, t][slot : slot + len(rows)] = rows[:, t * cols : (t + 1) * cols].view(np.int16)
        ttnn.copy_host_to_device_tensor(host, self.table)
        ttnn.copy_host_to_device_tensor(
            self._host(plan.lookup.reshape(self.sp, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self._sp_only),
            self.lookup,
        )
        ttnn.copy_host_to_device_tensor(
            self._host(
                plan.is_img.reshape(1, 1, -1, 1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self._is_img_dims
            ),
            self._is_img_rows,
        )
        ttnn.deallocate(self.is_img)
        self.is_img = ttnn.to_layout(self._is_img_rows, ttnn.TILE_LAYOUT)


def splice_media(emb: ttnn.Tensor, media: MediaSpliceBuffers) -> ttnn.Tensor:
    """emb [1, 1, s, hidden/tp] with media's image rows at its image positions. ttnn.embedding returns zeros where
    lookup is 0 (row 0 of the table), so one where() both supplies this SP row's image rows and, before a
    reduce-scatter, suppresses the token embedding where another row supplies them."""
    gathered = ttnn.embedding(media.lookup, media.table, layout=ttnn.TILE_LAYOUT, dtype=emb.dtype)
    if len(gathered.shape) == 3:
        gathered = ttnn.unsqueeze_to_4D(gathered)
    return ttnn.where(media.is_img, gathered, emb)
