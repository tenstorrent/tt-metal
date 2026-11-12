# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage-0 image-embedding injection: overwrite the token embeddings at a chunk's image positions."""

import torch

import ttnn
from models.demos.minimax_m3.tt.residual import activation_mapper


def inject_media_embeddings(
    mesh_device,
    x_embd: ttnn.Tensor,
    rows: torch.Tensor,
    mask: torch.Tensor,
    *,
    mesh_shape,
    sp_axis: int,
    tp_axis: int,
    emb_tp_sharded: bool,
) -> ttnn.Tensor:
    """x_embd with its image rows replaced. Consumes x_embd.

    rows [sp, rows_per_chip, hidden] and mask [sp, rows_per_chip] come from build_injection
    (models/demos/common/prefill/media.py) in the forward's per-chip order, so they shard like x_embd.
    ttnn.where broadcasts the mask over hidden, so the mask goes up as one column rather than a full-size
    keep tensor. The mask is replicated over TP: it has no hidden dimension to shard.
    """
    sp, local, hidden = rows.shape
    assert sp == mesh_shape[sp_axis], f"rows for sp={sp} on mesh {mesh_shape}"

    def to_device(t: torch.Tensor, dtype, tp_sharded: bool) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.reshape(1, 1, sp * local, -1).contiguous(),
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=activation_mapper(
                mesh_device, mesh_shape=mesh_shape, sp_axis=sp_axis, tp_axis=tp_axis, emb_tp_sharded=tp_sharded
            ),
        )

    cond = to_device(mask.to(torch.int32), ttnn.uint32, False)
    rows_t = to_device(rows.to(torch.bfloat16), ttnn.bfloat16, emb_tp_sharded)
    out = ttnn.where(cond, rows_t, x_embd)
    ttnn.deallocate(x_embd)
    ttnn.deallocate(cond)
    ttnn.deallocate(rows_t)
    return out
