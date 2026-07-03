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

import os

import ttnn

# GroupNorm mode for the spatially-sharded VAE decoder:
#   "dist"   -> distributed group_norm (per-shard fp32 stats + all-reduce, normalize local);
#               never gathers full spatial, so it fits L1 past the ~2100^2 gather ceiling.
#   "gather" -> legacy gather -> ttnn.group_norm -> re-shard (overflows L1 at large spatial).
# Default "dist"; override with HY_GN_MODE for A/B validation.
_GN_MODE = os.environ.get("HY_GN_MODE", "dist").lower()


def _all_reduce_sum(ccl, t: ttnn.Tensor, *, mesh_axis: int) -> ttnn.Tensor:
    """Sum a small (per-group) stat tensor across one mesh axis. all_gather concatenates
    each shard's value on dim0, then reduce — equal shard sizes make this an exact sum."""
    if mesh_axis is None or ccl.mesh_device.shape[mesh_axis] == 1:
        return t
    g = ccl.all_gather(t, dim=0, mesh_axis=mesh_axis, use_hyperparams=False)  # [nshard,1,G,1]
    s = ttnn.sum(g, dim=0, keepdim=True)  # [1,1,G,1]
    ttnn.deallocate(g)
    return s


def group_norm_distributed(norm, x_bthwc: ttnn.Tensor, ccl, *, h_mesh_axis=None, w_mesh_axis=None) -> ttnn.Tensor:
    """GroupNorm on an H/W-sharded input WITHOUT gathering full spatial.

    Per-group statistics pool over (channels-in-group x T x H x W). Each device computes
    its local sum / sum-of-squares (in the activation's native dtype, e.g. bf16 — the
    full-size sum/multiply ops dominate device time, so avoid the fp32 upcast) per group,
    the (tiny) [1,1,G,1] stat tensors are all-reduced across the h/w mesh axes to global
    stats, then the LOCAL shard is normalized and affine-transformed. Per-core work stays
    at the shard size, so L1 never overflows. Requires ``norm._raw_gamma`` / ``norm._raw_beta``
    ([1,1,1,C] fp32) attached at weight-load time (the packed ttnn.group_norm weights
    can't be reused here)."""
    G = norm.num_groups
    C = norm.num_channels
    Cg = C // G
    B, T, Hl, Wl, _ = x_bthwc.shape
    assert B == 1, "distributed group_norm assumes batch 1 (VAE decode)"
    n_local = T * Hl * Wl

    x = ttnn.to_layout(x_bthwc, ttnn.TILE_LAYOUT)
    x = ttnn.reshape(x, [1, 1, n_local, C])  # keep C last (multiple of 32) -> no TILE pad blowup

    # Reduce over SPATIAL first (dim2), keeping channels as the last dim. Reshaping to
    # per-group [.,G,Cg] with tiny Cg would pad Cg->32 in TILE and explode DRAM, so do
    # the (tiny) in-group channel reduction only after collapsing spatial to [1,1,1,C].
    csum = ttnn.sum(x, dim=2, keepdim=True)  # [1,1,1,C]
    xsq = ttnn.multiply(x, x)
    csumsq = ttnn.sum(xsq, dim=2, keepdim=True)  # [1,1,1,C]
    ttnn.deallocate(xsq)

    def _group_reduce(csum_1c):  # [1,1,1,C] -> per-group [1,1,G,1]
        g = ttnn.reshape(csum_1c, [1, 1, G, Cg])
        r = ttnn.sum(g, dim=3, keepdim=True)
        ttnn.deallocate(g)
        return r

    local_sum = _group_reduce(csum)
    local_sumsq = _group_reduce(csumsq)
    ttnn.deallocate(csum)
    ttnn.deallocate(csumsq)

    # all-reduce the tiny per-group stat tensors to global sums, then finalize mean/var
    for ax in (h_mesh_axis, w_mesh_axis):
        local_sum = _all_reduce_sum(ccl, local_sum, mesh_axis=ax)
        local_sumsq = _all_reduce_sum(ccl, local_sumsq, mesh_axis=ax)
    mesh_h = ccl.mesh_device.shape[h_mesh_axis] if h_mesh_axis is not None else 1
    mesh_w = ccl.mesh_device.shape[w_mesh_axis] if w_mesh_axis is not None else 1
    count = float(n_local * Cg * mesh_h * mesh_w)

    mean = ttnn.multiply(local_sum, 1.0 / count)  # [1,1,G,1]
    msq = ttnn.multiply(local_sumsq, 1.0 / count)
    ttnn.deallocate(local_sum)
    ttnn.deallocate(local_sumsq)
    var = ttnn.subtract(msq, ttnn.multiply(mean, mean))
    ttnn.deallocate(msq)
    inv = ttnn.rsqrt(ttnn.add(var, norm.eps))  # [1,1,G,1]
    ttnn.deallocate(var)

    # Expand per-group mean/inv to per-channel [1,1,1,C] (broadcast each group over its
    # Cg channels), on tiny tensors, then normalize the full local activation.
    ones_gcg = ttnn.ones([1, 1, G, Cg], dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=x.device())
    mean_c = ttnn.reshape(ttnn.multiply(mean, ones_gcg), [1, 1, 1, C])
    inv_c = ttnn.reshape(ttnn.multiply(inv, ones_gcg), [1, 1, 1, C])
    ttnn.deallocate(ones_gcg)
    ttnn.deallocate(mean)
    ttnn.deallocate(inv)

    xn = ttnn.multiply(ttnn.subtract(x, mean_c), inv_c)  # broadcast [1,1,1,C] over [1,1,n,C]
    ttnn.deallocate(x)
    ttnn.deallocate(mean_c)
    ttnn.deallocate(inv_c)

    out = ttnn.add(ttnn.multiply(xn, norm._raw_gamma), norm._raw_beta)  # per-channel affine
    ttnn.deallocate(xn)
    out = ttnn.typecast(out, norm.dtype)
    out = ttnn.reshape(out, [B, T, Hl, Wl, C])
    return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)


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
    """Run a full-spatial op `norm` on a spatially-sharded input.

    Default: distributed group_norm (no full-spatial gather; fits L1 at any resolution)
    when the norm carries raw affine (a GroupNorm3D loaded via the VAE weight loader).
    Falls back to gather -> op -> re-shard for other norms or when HY_GN_MODE=gather."""
    if _GN_MODE == "dist" and hasattr(norm, "_raw_gamma"):
        return group_norm_distributed(norm, x_bthwc, ccl, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
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
