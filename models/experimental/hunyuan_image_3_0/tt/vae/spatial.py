# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Spatial (H/W) parallel helpers for the HunyuanImage-3.0 VAE on a 2×2 mesh.
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


def mesh_mapper_hw_spatial(
    mesh_device: ttnn.MeshDevice,
    *,
    h_mesh_axis: int | None = None,
    w_mesh_axis: int | None = None,
) -> ttnn.ShardTensor2dMesh:
    """Build a 2D mesh mapper that shards H (dim 2) and/or W (dim 3)."""
    mesh_shape = tuple(mesh_device.shape)
    dims: list[int | None] = [None, None]
    if h_mesh_axis is not None:
        dims[h_mesh_axis] = 2
    if w_mesh_axis is not None:
        dims[w_mesh_axis] = 3
    # Unused mesh axis must still reference a distinct tensor dim (see pipeline decode).
    filled = [d if d is not None else (3 if 2 in dims else 2) for d in dims]
    return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_shape, dims=filled)


def encoder_w_spatial_enabled() -> bool:
    return os.environ.get("HY_ENCODER_W_SPATIAL", "0").strip().lower() in ("1", "true", "yes")


# GroupNorm mode for the spatially-sharded VAE decoder:
#   "dist"   -> distributed group_norm (per-shard fp32 stats + all-reduce, normalize local);
#               never gathers full spatial, so it fits L1 past the ~2100^2 gather ceiling.
#   "gather" -> legacy gather -> ttnn.group_norm -> re-shard (overflows L1 at large spatial).
# Default "dist"; override with HY_GN_MODE for A/B validation.
_GN_MODE = os.environ.get("HY_GN_MODE", "dist").lower()

# How distributed group_norm computes its two spatial sums (sum(x), sum(x^2)):
#   "split"  -> two independent reduces over [1,1,n,C] (default; no concat copy).
#   "concat" -> stack on dim1, one reduce over [1,2,n,C] (fewer dispatches, extra copy).
_GN_STATS = os.environ.get("HY_GN_STATS", "split").lower()

# Keep small activations resident in L1 so the GroupNorm normalize (mul + add) and the
# ROW_MAJOR hand-off to SiLU/conv skip DRAM round-trips. Gated by the normalized
# [B,T,H,W,C] element count: the mid block and early up-level res tensors fall below it,
# while the >=level-2 tail (tens of millions of elements — hundreds of MB) is far above and
# stays in DRAM (a single such tensor would not fit L1 across the worker grid anyway).
# 2M elems (~4 MB bf16) covers the mid block / early up-level res tensors; measured
# consistently faster than all-DRAM with identical PCC (0.999845). Always on (no env knob).
_L1_RESIDENT_MAX_ELEMENTS = 2 * 1024 * 1024


def _affine_mem_config(num_elements: int) -> ttnn.MemoryConfig:
    """L1-interleaved for activations small enough to stay resident, else DRAM (prevents L1
    OOM at high resolution). Used for the distributed GroupNorm normalize + SiLU chain."""
    return ttnn.L1_MEMORY_CONFIG if num_elements <= _L1_RESIDENT_MAX_ELEMENTS else ttnn.DRAM_MEMORY_CONFIG


def _spatial_sum(x: ttnn.Tensor, *, ncores: int = 110) -> ttnn.Tensor:
    """Sum over dim2 of a [1, D, n, C] TILE tensor -> [1, D, 1, C].

    A plain ``ttnn.sum(x, dim=2)`` parallelizes only over the D*ceil(C/32) output
    tiles, so a large-``n`` reduce with a small D*C (e.g. the C=128 full-spatial
    GroupNorm stat, 4 output tiles) pins to ~4 cores that each stream the whole
    ``n`` from DRAM — the dominant VAE-decode device cost. Split ``n`` into K
    chunks so the heavy reduce emits D*K*ceil(C/32) output tiles (spread across
    many cores), then fold the K partials with a second tiny reduce.

    K is the largest divisor of n/32 with D*K*ceil(C/32) <= ncores, so the split
    is a tile-aligned (free) reshape and the heavy reduce fits in one core wave;
    K==1 falls back to the plain single reduce (small tensors, C already wide)."""
    _, D, n, C = x.shape
    ct = max(1, (C + 31) // 32)
    target_k = max(1, ncores // (D * ct))
    K = 1
    if n % 32 == 0 and target_k > 1:
        m = n // 32  # tile-rows available to split
        for k in range(1, min(target_k, m) + 1):
            if m % k == 0:
                K = k
    if K <= 1:
        return ttnn.sum(x, dim=2, keepdim=True)
    xr = ttnn.reshape(x, [1, D * K, n // K, C])
    partial = ttnn.sum(xr, dim=2, keepdim=True)  # [1, D*K, 1, C], many output tiles
    partial = ttnn.reshape(partial, [1, D, K, C])
    out = ttnn.sum(partial, dim=2, keepdim=True)  # [1, D, 1, C], tiny fold
    ttnn.deallocate(partial)
    return out


def _gn_ones_gcg(norm, device, G: int, Cg: int) -> ttnn.Tensor:
    """Cached [1,1,G,Cg] ones — ``ttnn.ones`` H2D is illegal during trace capture."""
    cache = getattr(norm, "_gn_ones_gcg_cache", None)
    if cache is None:
        cache = {}
        norm._gn_ones_gcg_cache = cache
    key = (G, Cg)
    if key not in cache:
        cache[key] = ttnn.ones([1, 1, G, Cg], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    return cache[key]


def _all_reduce_sum(ccl, t: ttnn.Tensor, *, mesh_axis: int) -> ttnn.Tensor:
    """Sum a small per-group stat tensor (or several stacked on dim1) across one mesh
    axis. all_gather concatenates each shard's value on dim0, then reduce — equal shard
    sizes make this an exact sum."""
    if mesh_axis is None or ccl.mesh_device.shape[mesh_axis] == 1:
        return t
    g = ccl.all_gather(t, dim=0, mesh_axis=mesh_axis, use_hyperparams=False)  # [nshard,*,G,1]
    s = ttnn.sum(g, dim=0, keepdim=True)  # [1,*,G,1]
    ttnn.deallocate(g)
    return s


def group_norm_distributed(norm, x_bthwc: ttnn.Tensor, ccl, *, h_mesh_axis=None, w_mesh_axis=None) -> ttnn.Tensor:
    """GroupNorm on an H/W-sharded input WITHOUT gathering full spatial.

    Per-group statistics pool over (channels-in-group x T x H x W). Each device computes
    its local sum / sum-of-squares (in the activation's native dtype, e.g. bf16 — the
    full-size reduce ops dominate device time, so avoid the fp32 upcast) per group, the
    (tiny) [1,1,G,1] stat tensors are all-reduced across the h/w mesh axes to global
    stats, then the LOCAL shard is normalized and affine-transformed. Per-core work stays
    at the shard size, so L1 never overflows. Requires ``norm._raw_gamma`` / ``norm._raw_beta``
    ([1,1,1,C], stored in the norm's own dtype) attached at weight-load time (the packed
    ttnn.group_norm weights can't be reused here)."""
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
    # (ttnn.var's Welford reduce was tried here instead of this sum/sum-of-squares
    # approach, to skip materializing x^2 -- measured ~4x SLOWER in practice
    # (WelfordReduceDeviceOperation cost ~49ms/call vs sum's ~1-15us), so don't retry.)
    #
    # Two ways to compute the two spatial sums (sum(x), sum(x^2)):
    #   "concat" -> stack x and x^2 on dim1, ONE sum over [1,2,n,C]. One reduce dispatch,
    #               but the concat copies 2*n*C elements up front.
    #   "split"  -> two independent sums over [1,1,n,C]. No concat copy, two reduces.
    # At full spatial res the concat copy costs more than a second reduce dispatch, so
    # "split" is the default; HY_GN_STATS=concat restores the fused path for A/B.
    xsq = ttnn.multiply(x, x)
    if _GN_STATS == "concat":
        x_xsq = ttnn.concat([x, xsq], dim=1)  # [1,2,n_local,C]
        ttnn.deallocate(xsq)
        csum_both = _spatial_sum(x_xsq)  # [1,2,1,C]
        ttnn.deallocate(x_xsq)
        csum = ttnn.slice(csum_both, [0, 0, 0, 0], [1, 1, 1, C])
        csumsq = ttnn.slice(csum_both, [0, 1, 0, 0], [1, 2, 1, C])
        ttnn.deallocate(csum_both)
    else:
        csum = _spatial_sum(x)  # [1,1,1,C]
        csumsq = _spatial_sum(xsq)  # [1,1,1,C]
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

    # Stack sum/sumsq into one [1,2,G,1] tensor so each mesh axis needs only ONE
    # all_gather+sum round trip instead of two (sum and sumsq separately) — halves
    # the AllGatherAsync dispatch count for this stat all-reduce.
    stats = ttnn.concat([local_sum, local_sumsq], dim=1)  # [1,2,G,1]
    ttnn.deallocate(local_sum)
    ttnn.deallocate(local_sumsq)
    for ax in (h_mesh_axis, w_mesh_axis):
        stats = _all_reduce_sum(ccl, stats, mesh_axis=ax)
    local_sum = ttnn.slice(stats, [0, 0, 0, 0], [1, 1, G, 1])
    local_sumsq = ttnn.slice(stats, [0, 1, 0, 0], [1, 2, G, 1])
    ttnn.deallocate(stats)
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
    # Cg channels), on tiny tensors. The [1,1,G,Cg] ones broadcaster is a constant that
    # only depends on the channel config — cached per (G,Cg) and built before trace
    # capture (ttnn.ones H2D is illegal during trace).
    ones_gcg = _gn_ones_gcg(norm, x.device(), G, Cg)
    mean_c = ttnn.reshape(ttnn.multiply(mean, ones_gcg), [1, 1, 1, C])
    inv_c = ttnn.reshape(ttnn.multiply(inv, ones_gcg), [1, 1, 1, C])
    ttnn.deallocate(mean)
    ttnn.deallocate(inv)

    # Fold the affine into per-channel scale/shift so the full-spatial normalize is a
    # single mul+add (2 passes over [1,1,n,C]) instead of subtract+mul+mul+add (4):
    #   out = x*scale + shift,  scale = inv*gamma,  shift = beta - mean*scale.
    # scale/shift are computed on the tiny [1,1,1,C] tensors, off the hot path.
    scale_c = ttnn.multiply(inv_c, norm._raw_gamma)  # [1,1,1,C]
    shift_c = ttnn.subtract(norm._raw_beta, ttnn.multiply(mean_c, scale_c))
    ttnn.deallocate(mean_c)
    ttnn.deallocate(inv_c)

    # Normalize (mul + add) and the ROW_MAJOR hand-off to SiLU/conv stay in L1 for small
    # activations so these pointwise passes don't round-trip through DRAM; large ones fall
    # back to DRAM (see _affine_mem_config). conv3d accepts L1-interleaved ROW_MAJOR input.
    mem = _affine_mem_config(B * n_local * C)
    prod = ttnn.multiply(x, scale_c, memory_config=mem)  # broadcast [1,1,1,C] over [1,1,n,C]
    out = ttnn.add(prod, shift_c, memory_config=mem)
    ttnn.deallocate(prod)
    ttnn.deallocate(x)
    ttnn.deallocate(scale_c)
    ttnn.deallocate(shift_c)
    out = ttnn.reshape(out, [B, T, Hl, Wl, C], memory_config=mem)
    return ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT, memory_config=mem)


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
    from .resnet_conv import HunyuanResnetConvPair

    if isinstance(module, HunyuanSymmetricConv3d):
        module.ccl = ccl
        module.h_mesh_axis = h_mesh_axis
        module.w_mesh_axis = w_mesh_axis
        module.spatial_sharded = True
    if isinstance(module, HunyuanResnetConvPair):
        module.enable_spatial(ccl, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
    if type(module).__name__ in ("ResnetBlockTTNN", "AttnBlockTTNN", "NormOutTTNN", "EncoderHeadTTNN"):
        module._sp_ccl = ccl
        module._sp_h = h_mesh_axis
        module._sp_w = w_mesh_axis

    for v in vars(module).values():
        if isinstance(v, ModuleList):
            for it in v:
                enable_vae_spatial(it, ccl, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
        elif isinstance(v, Module):
            enable_vae_spatial(v, ccl, h_mesh_axis=h_mesh_axis, w_mesh_axis=w_mesh_axis)
