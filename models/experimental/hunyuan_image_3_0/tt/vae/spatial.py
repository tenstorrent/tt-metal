# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Spatial (H/W) parallel helpers for the HunyuanImage-3.0 VAE decoder on a 2x2 mesh.
#
# The decoder shards feature-map height on `h_mesh_axis` and width on `w_mesh_axis`
# (BTHWC: H=dim2, W=dim3). Convs stay sharded and exchange a halo at the boundary
# (see conv3d._forward_sharded). Ops that need the FULL spatial context — GroupNorm
# (per-group stats) and attention (global over T*H*W) — are wrapped with:
#
#   gather_hw -> <op on full spatial> -> partition_hw
#
# After gather_hw on BOTH axes every device holds the identical full-spatial tensor,
# so the existing replicated op runs unchanged; partition_hw re-shards it. all_gather
# requires TILE layout, so we tilize around it and restore ROW_MAJOR (the conv I/O
# layout) on the way out.

import ttnn


def gather_hw(ccl, x_bthwc: ttnn.Tensor, *, h_mesh_axis=None, w_mesh_axis=None) -> ttnn.Tensor:
    """Sharded [B,T,H/h,W/w,C] -> full [B,T,H,W,C] (replicated on the h/w axes)."""
    if h_mesh_axis is None and w_mesh_axis is None:
        return x_bthwc
    x = ttnn.to_layout(x_bthwc, ttnn.TILE_LAYOUT)
    if h_mesh_axis is not None:
        x = ccl.all_gather(x, dim=2, mesh_axis=h_mesh_axis, use_hyperparams=False)
    if w_mesh_axis is not None:
        x = ccl.all_gather(x, dim=3, mesh_axis=w_mesh_axis, use_hyperparams=False)
    out = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    ttnn.deallocate(x)
    return out


def partition_hw(x_bthwc: ttnn.Tensor, *, h_mesh_axis=None, w_mesh_axis=None) -> ttnn.Tensor:
    """Full [B,T,H,W,C] -> sharded [B,T,H/h,W/w,C] (inverse of gather_hw)."""
    x = x_bthwc
    if h_mesh_axis is not None:
        x = ttnn.mesh_partition(x, dim=2, cluster_axis=h_mesh_axis)
    if w_mesh_axis is not None:
        x = ttnn.mesh_partition(x, dim=3, cluster_axis=w_mesh_axis)
    return x


def norm_sharded(norm, x_bthwc: ttnn.Tensor, ccl, *, h_mesh_axis=None, w_mesh_axis=None) -> ttnn.Tensor:
    """Run a full-spatial op `norm` on a spatially-sharded input: gather -> op -> re-shard."""
    full = gather_hw(ccl, x_bthwc, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
    normed = norm(full)
    ttnn.deallocate(full)
    return partition_hw(normed, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)


def enable_vae_spatial(module, ccl, *, h_mesh_axis, w_mesh_axis) -> None:
    """Recursively switch a (replicated) VAE decoder into H/W-spatial-parallel mode.

    - HunyuanSymmetricConv3d  -> runs `_forward_sharded` (neighbor-pad halo).
    - ResnetBlock/NormOut     -> wrap their GroupNorm with norm_sharded (gather→norm→shard).
    - AttnBlock               -> gather to full spatial, run, re-shard (global SDPA).

    Blocks read the injected `_sp_*` attrs in their forward; convs read their own
    `ccl/h_mesh_axis/w_mesh_axis`. No constructor signatures change.
    """
    from models.tt_dit.layers.module import Module, ModuleList
    from .conv3d import HunyuanSymmetricConv3d

    if isinstance(module, HunyuanSymmetricConv3d):
        module.ccl = ccl
        module.h_mesh_axis = h_mesh_axis
        module.w_mesh_axis = w_mesh_axis
        module.spatial_sharded = True
    if type(module).__name__ in ("ResnetBlockTTNN", "AttnBlockTTNN", "NormOutTTNN"):
        module._sp_ccl = ccl
        module._sp_h = h_mesh_axis
        module._sp_w = w_mesh_axis

    for v in vars(module).values():
        if isinstance(v, ModuleList):
            for it in v:
                enable_vae_spatial(it, ccl, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
        elif isinstance(v, Module):
            enable_vae_spatial(v, ccl, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
