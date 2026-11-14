# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 image-embedding injection: overwrite the token embeddings at a chunk's image positions."""

import torch

import ttnn
from models.demos.minimax_m3.tt.residual import activation_mapper


class MediaInjectionBuffers:
    """The splice's device tensors, allocated once and refilled each forward.

    Shapes are fixed for a deployment (``forward_tokens_budget`` rows, ``hidden`` columns), so the
    forward pays no host tilize. The mask is replicated over TP: it has no hidden dimension to shard.
    It goes up row-major and is tilized on device, for the reason MediaPartials.is_img does.
    """

    def __init__(self, mesh_device, *, rows_shape, mesh_shape, sp_axis: int, tp_axis: int, emb_tp_sharded: bool):
        sp, local, hidden = rows_shape
        assert sp == mesh_shape[sp_axis], f"rows for sp={sp} on mesh {mesh_shape}"
        self.shape = rows_shape
        self._mapper = lambda tp_sharded: activation_mapper(
            mesh_device, mesh_shape=mesh_shape, sp_axis=sp_axis, tp_axis=tp_axis, emb_tp_sharded=tp_sharded
        )
        self._cond_rows = self._alloc(
            mesh_device, torch.zeros(1, 1, sp * local, 1, dtype=torch.int32), ttnn.uint32, False, ttnn.ROW_MAJOR_LAYOUT
        )
        self.cond = ttnn.to_layout(self._cond_rows, ttnn.TILE_LAYOUT)
        self.rows = self._alloc(
            mesh_device,
            torch.zeros(1, 1, sp * local, hidden, dtype=torch.bfloat16),
            ttnn.bfloat16,
            emb_tp_sharded,
            ttnn.TILE_LAYOUT,
        )

    def _alloc(self, mesh_device, t: torch.Tensor, dtype, tp_sharded: bool, layout) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mapper(tp_sharded),
        )

    def refill(self, rows: torch.Tensor, mask: torch.Tensor, *, emb_tp_sharded: bool) -> None:
        sp, local, hidden = self.shape
        host = lambda t, dtype, tp_sharded, layout: ttnn.from_torch(
            t.reshape(1, 1, sp * local, -1).contiguous(),
            dtype=dtype,
            layout=layout,
            mesh_mapper=self._mapper(tp_sharded),
        )
        ttnn.copy_host_to_device_tensor(
            host(mask.to(torch.int32), ttnn.uint32, False, ttnn.ROW_MAJOR_LAYOUT), self._cond_rows
        )
        ttnn.copy_host_to_device_tensor(
            host(rows.to(torch.bfloat16), ttnn.bfloat16, emb_tp_sharded, ttnn.TILE_LAYOUT), self.rows
        )
        ttnn.deallocate(self.cond)
        self.cond = ttnn.to_layout(self._cond_rows, ttnn.TILE_LAYOUT)


class MediaPartials:
    """D's per-SP-row image rows, on device, for the hook inside TtParallelEmbedding._forward_2d.

    table [h, hidden/tp] per chip (row 0 zeros), lookup [1,1,s_total] per chip (0 = not this SP row's),
    is_img [1,1,s_total,1] replicated. Allocated once per table height — build_media_partials snaps the
    height to a power of two, so a deployment uses a handful — and refilled with
    copy_host_to_device_tensor, so a forward pays no host tilize.

    is_img goes up row-major and is tilized on device: ttnn.where needs a TILE condition, but tilizing
    a width-1 tensor on the host pads that dimension to a full tile, so a tiled upload would send 32
    uint32 per position to carry one bit. That tilize is the one tensor a forward allocates. Refilling
    frees the previous tile, which is safe only because a command queue completes in order: the forward
    still reading it runs before anything writes the reused address. Keep the uploads on one queue.
    """

    def __init__(self, mesh_device, *, sp: int, height: int, hidden: int, s_total: int, mesh_shape, sp_axis, tp_axis):
        self.shape = (sp, height, hidden)
        self.s_total = s_total
        sp_tp, sp_only = [None, None], [None, None]
        sp_tp[sp_axis], sp_tp[tp_axis] = 0, 2
        sp_only[sp_axis] = 0
        self._sp_tp, self._sp_only, self._replicated = tuple(sp_tp), tuple(sp_only), (None, None)
        self._mapper = lambda dims: ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape)

        self.table = self._to_device(
            mesh_device,
            torch.zeros(sp, height, hidden, dtype=torch.bfloat16),
            ttnn.bfloat16,
            ttnn.TILE_LAYOUT,
            self._sp_tp,
        )
        self.lookup = self._to_device(
            mesh_device,
            torch.zeros(sp, 1, s_total, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self._sp_only,
        )
        self._is_img_rows = self._to_device(
            mesh_device,
            torch.zeros(1, 1, s_total, 1, dtype=torch.int32),
            ttnn.uint32,
            ttnn.ROW_MAJOR_LAYOUT,
            self._replicated,
        )
        self.is_img = ttnn.to_layout(self._is_img_rows, ttnn.TILE_LAYOUT)

    def _to_device(self, mesh_device, t, dtype, layout, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mapper(dims),
        )

    def _host(self, t, dtype, layout, dims):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, mesh_mapper=self._mapper(dims))

    def refill(self, table: torch.Tensor, lookup: torch.Tensor, is_img: torch.Tensor) -> None:
        sp, height, hidden = self.shape
        assert tuple(table.shape) == (sp, height, hidden), f"{tuple(table.shape)} against {self.shape}"
        ttnn.copy_host_to_device_tensor(self._host(table, ttnn.bfloat16, ttnn.TILE_LAYOUT, self._sp_tp), self.table)
        ttnn.copy_host_to_device_tensor(
            self._host(lookup.reshape(sp, 1, -1), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self._sp_only), self.lookup
        )
        ttnn.copy_host_to_device_tensor(
            self._host(
                is_img.reshape(1, 1, -1, 1).to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, self._replicated
            ),
            self._is_img_rows,
        )
        ttnn.deallocate(self.is_img)
        self.is_img = ttnn.to_layout(self._is_img_rows, ttnn.TILE_LAYOUT)


def inject_media_embeddings(
    mesh_device,
    x_embd: ttnn.Tensor,
    rows: torch.Tensor,
    mask: torch.Tensor,
    *,
    cache,
    mesh_shape,
    sp_axis: int,
    tp_axis: int,
    emb_tp_sharded: bool,
) -> ttnn.Tensor:
    """x_embd with its image rows replaced. Consumes x_embd.

    rows [sp, rows_per_chip, hidden] and mask [sp, rows_per_chip] come from build_injection
    (models/demos/common/prefill/media.py) in the forward's per-chip order, so they shard like x_embd.
    ttnn.where broadcasts the mask over hidden, so the mask goes up as one column rather than a full-size
    keep tensor, and rows outside the mask are never read. ``cache`` holds the device tensors across
    forwards: they are allocated on the first media forward and refilled after that.
    """
    if cache.media_buffers is None:
        cache.media_buffers = MediaInjectionBuffers(
            mesh_device,
            rows_shape=tuple(rows.shape),
            mesh_shape=mesh_shape,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            emb_tp_sharded=emb_tp_sharded,
        )
    buffers = cache.media_buffers
    assert tuple(rows.shape) == buffers.shape, f"rows {tuple(rows.shape)} against buffers {buffers.shape}"
    buffers.refill(rows, mask, emb_tp_sharded=emb_tp_sharded)
    out = ttnn.where(buffers.cond, buffers.rows, x_embd)
    ttnn.deallocate(x_embd)
    return out
