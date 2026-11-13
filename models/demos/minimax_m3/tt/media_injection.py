# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 image-embedding injection: overwrite the token embeddings at a chunk's image positions."""

import torch

import ttnn
from models.demos.minimax_m3.tt.residual import activation_mapper


class MediaInjectionBuffers:
    """The splice's device tensors, allocated once and refilled each forward.

    Shapes are fixed for a deployment (``forward_tokens_budget`` rows, ``hidden`` columns), so the
    forward pays no allocation and no host tilize. The mask is replicated over TP: it has no hidden
    dimension to shard.
    """

    def __init__(self, mesh_device, *, rows_shape, mesh_shape, sp_axis: int, tp_axis: int, emb_tp_sharded: bool):
        sp, local, hidden = rows_shape
        assert sp == mesh_shape[sp_axis], f"rows for sp={sp} on mesh {mesh_shape}"
        self.shape = rows_shape
        self._mapper = lambda tp_sharded: activation_mapper(
            mesh_device, mesh_shape=mesh_shape, sp_axis=sp_axis, tp_axis=tp_axis, emb_tp_sharded=tp_sharded
        )
        self.cond = self._alloc(mesh_device, torch.zeros(1, 1, sp * local, 1, dtype=torch.int32), ttnn.uint32, False)
        self.rows = self._alloc(
            mesh_device, torch.zeros(1, 1, sp * local, hidden, dtype=torch.bfloat16), ttnn.bfloat16, emb_tp_sharded
        )

    def _alloc(self, mesh_device, t: torch.Tensor, dtype, tp_sharded: bool) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mapper(tp_sharded),
        )

    def refill(self, rows: torch.Tensor, mask: torch.Tensor, *, emb_tp_sharded: bool) -> None:
        sp, local, hidden = self.shape
        host = lambda t, dtype, tp_sharded: ttnn.from_torch(
            t.reshape(1, 1, sp * local, -1).contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=self._mapper(tp_sharded),
        )
        ttnn.copy_host_to_device_tensor(host(mask.to(torch.int32), ttnn.uint32, False), self.cond)
        ttnn.copy_host_to_device_tensor(host(rows.to(torch.bfloat16), ttnn.bfloat16, emb_tp_sharded), self.rows)


class MediaPartials:
    """D's per-SP-row image rows, on device, for the hook inside TtParallelEmbedding._forward_2d.

    table [k+1, hidden/tp] per chip (row 0 zeros), lookup [1,1,s_total] per chip (0 = not this SP
    row's), is_img [1,1,s_total,1] replicated. Uploaded per forward: k varies with how many images the
    forward names, so unlike the A+B buffers these are not fixed-shape. The payload is small — the
    table is the image bytes divided by SP and TP — so the allocation is not what this is saving.
    """

    def __init__(self, mesh_device, table, lookup, is_img, *, mesh_shape, sp_axis: int, tp_axis: int):
        sp = mesh_shape[sp_axis]
        assert table.shape[0] == sp and lookup.shape[0] == sp, f"{tuple(table.shape)} {tuple(lookup.shape)}"

        def mapper(dims):
            return ttnn.ShardTensor2dMesh(mesh_device, dims=dims, mesh_shape=mesh_shape)

        sp_tp, sp_only, replicated = [None, None], [None, None], (None, None)
        sp_tp[sp_axis], sp_tp[tp_axis] = 0, 2
        sp_only[sp_axis] = 0
        self.table = ttnn.from_torch(
            table,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper(tuple(sp_tp)),
        )
        self.lookup = ttnn.from_torch(
            lookup.reshape(sp, 1, -1),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper(tuple(sp_only)),
        )
        self.is_img = ttnn.from_torch(
            is_img.reshape(1, 1, -1, 1).to(torch.int32),
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper(replicated),
        )

    def deallocate(self) -> None:
        for t in (self.table, self.lookup, self.is_img):
            ttnn.deallocate(t)


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
