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

    rows [sp, chunk_local, hidden] and mask [sp, chunk_local] come from build_injection
    (models/demos/common/prefill/media.py) in the chunk's rotated per-chip order, so they shard like x_embd.
    Computes x * keep + rows with keep = 1 off the mask: text rows stay bit-exact (x * 1 + 0) and image rows
    become the image's embeddings (x * 0 + row).
    """
    sp, chunk_local, hidden = rows.shape
    assert sp == mesh_shape[sp_axis], f"rows for sp={sp} on mesh {mesh_shape}"
    keep = (~mask).to(torch.bfloat16).unsqueeze(-1).expand(sp, chunk_local, hidden)
    mapper = activation_mapper(
        mesh_device, mesh_shape=mesh_shape, sp_axis=sp_axis, tp_axis=tp_axis, emb_tp_sharded=emb_tp_sharded
    )

    def to_device(t: torch.Tensor) -> ttnn.Tensor:
        return ttnn.from_torch(
            t.reshape(1, 1, sp * chunk_local, hidden).to(torch.bfloat16).contiguous(),
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    keep_t, rows_t = to_device(keep), to_device(rows)
    kept = ttnn.multiply(x_embd, keep_t)
    ttnn.deallocate(x_embd)
    ttnn.deallocate(keep_t)
    out = ttnn.add(kept, rows_t)
    ttnn.deallocate(kept)
    ttnn.deallocate(rows_t)
    return out
