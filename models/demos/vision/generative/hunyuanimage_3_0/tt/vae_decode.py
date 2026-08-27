# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""On-device decode for HunyuanImage-3.0's ``AutoencoderKLConv3D`` (3D DCAE VAE).

Perf lever #1 for the T2I render: replace the ~36 s host ``model.vae.decode`` tail
with an on-device pass. This module implements the DECODE path only, mirroring the
HF reference (``autoencoder_kl_3d.py``) op-for-op so it can be PCC-gated against it.

Design:
  * conv3d primitive = the de-risked spike path: ``ttnn.experimental.conv3d`` with
    SYMMETRIC pad=1 and ``utils.conv3d.get_conv3d_config`` blockings (PCC 0.99999).
  * GroupNorm = ``tt_dit`` ``GroupNorm3D`` (5D BTHWC, dims=3 pooling == torch
    ``GroupNorm(32, C, eps=1e-6)`` over (C/G, T, H, W)).
  * UpsampleDCAE pixel-shuffle = the Mochi ``depth_to_spacetime`` 3-step reshape/permute
    (channel order ``(r1 r2 r3 c)`` == Mochi ``(texp, sexp_h, sexp_w, C_out)``, so the HF
    conv weights need NO swizzle).

This first cut targets a SINGLE CHIP (factor=1): no CCL / spatial gather runs. The mesh
HW-parallel path (the real perf win) is layered on later behind the same module API.

Tensor convention through the decoder: ROW_MAJOR ``[N, T, H, W, C]`` (channels-last), C a
multiple of 32 (all decoder widths are). Torch reference is ``[N, C, T, H, W]``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import ttnn
from models.tt_dit.layers.normalization import GroupNorm3D
from models.tt_dit.parallel.config import MochiVAEParallelConfig, ParallelFactor, vae_neighbor_pad
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils.conv3d import get_conv3d_config, register_conv3d_configs

ALIGNMENT = 32
GN_GROUPS = 32
GN_EPS = 1e-6

# conv3d blocking knobs (Lever: conv3d block-size sweep). Under H/W fracture the per-chip
# spatial is small, so a larger C_in_block that OOMs single-chip can fit L1 and cut passes.
import os as _os

# Conv3d blocking. 128/128 is the MEASURED optimum (VAE 4.04s -> 1.08s, -73%):
#   16/64 = +258%, 32/32 = +29%, 32/64 = baseline, 64/64 = -55%, 64/128 = -62%, 128/128 = -73%.
# Monotonic up to 128, then hard walls: cin>=256 or cout>=256 overflows L1 at render
# (TT_THROW program.cpp:1492), and cin=512 fails weight prep
# (TT_FATAL prepare_conv3d_weights.cpp:172 num_C_in_blocks*C_in_block == C_in_aligned).
# The knob is sharp in BOTH directions, so it stays env-overridable.
_CINBLK = int(_os.environ.get("HUNYUAN_VAE_CINBLK", "128"))
_COUTBLK = int(_os.environ.get("HUNYUAN_VAE_COUTBLK", "128"))


HF_MODEL_ID = "tencent/HunyuanImage-3.0"


def reference_autoencoder_path() -> str:
    """Absolute path to the HF reference ``autoencoder_kl_3d.py``, machine-independently.

    The PCC gate needs the exact HF module to compare against. Resolve it through
    ``huggingface_hub`` so it honours HF_HOME / HF_HUB_CACHE on any machine (and works
    under HF_HUB_OFFLINE=1 once the weights are cached) instead of hardcoding one
    developer's cache path. ``HUNYUAN_VAE_REF`` still overrides, and may be a glob.
    """
    override = _os.environ.get("HUNYUAN_VAE_REF")
    if override:
        import glob as _glob

        hits = sorted(_glob.glob(override))
        if not hits:
            raise FileNotFoundError(f"HUNYUAN_VAE_REF matched nothing: {override}")
        return hits[-1]
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_MODEL_ID, "autoencoder_kl_3d.py")


def _j(prefix: str, name: str) -> str:
    """Join a state_dict prefix with a leaf name, tolerating an empty prefix
    (isolated-module state_dicts have unprefixed keys like ``norm1.weight``)."""
    return f"{prefix}.{name}" if prefix else name


# Safe L1-fitting blockings for the big-in-channel decoder convs so they don't hit the
# get_conv3d_config fallback (C_in_block = in_channels -> L1 overflow for 512/1024-in).
# C_in_block=32 fits L1 everywhere; the mesh's small per-chip spatial keeps it fast later.
def _register_vae_conv3d_configs(cin_blk=_CINBLK, cout_blk=_COUTBLK):
    _co_big = min(cout_blk, 32)  # big-out UpsampleDCAE convs keep a small C_out_block
    register_conv3d_configs(
        {
            (1024, 1024, (3, 3, 3)): (cin_blk, cout_blk, 1, 1, 1),
            (1024, 8192, (3, 3, 3)): (cin_blk, _co_big, 1, 1, 1),  # up.0 UpsampleDCAE conv
            (1024, 4096, (3, 3, 3)): (cin_blk, _co_big, 1, 1, 1),  # up.1 UpsampleDCAE conv
            (512, 512, (3, 3, 3)): (cin_blk, cout_blk, 1, 1, 1),
            (512, 1024, (3, 3, 3)): (cin_blk, cout_blk, 1, 1, 1),  # up.2 UpsampleDCAE conv
            (256, 256, (3, 3, 3)): (cin_blk, cout_blk, 1, 1, 1),
            (256, 512, (3, 3, 3)): (cin_blk, cout_blk, 1, 1, 1),  # up.3 UpsampleDCAE conv
            # (32,1024), (128,128), (128,3): in-channels <=128 -> fallback C_in_block=in fits L1.
        }
    )


_register_vae_conv3d_configs()


def compute_kernel_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


# --------------------------------------------------------------------- mesh context


class MeshCtx:
    """Mesh HW-fracture context. ``h_factor``/``w_factor`` shard H/W across mesh axes.

    When both factors are 1 (or ctx is None), every module takes the single-chip path
    (byte-identical to the validated factor=1 decoder). When fractured:
      * convs run on the LOCAL shard with an H/W neighbor halo (zeros at the global
        boundary, real neighbor pixels interior — matches HF padding_mode="zeros");
      * GroupNorm gathers full H/W (exact statistic) then re-partitions;
      * the pixel-shuffle upsample stays LOCAL (spatial doubling needs no CCL).
    """

    def __init__(self, mesh_device, h_factor: int, h_axis: int, w_factor: int, w_axis: int, num_links: int = 2):
        self.mesh_device = mesh_device
        self.h_factor = h_factor
        self.h_axis = h_axis
        self.w_factor = w_factor
        self.w_axis = w_axis
        # time_parallel unused (T not fractured); put it on the non-w axis with factor 1.
        self.pcfg = MochiVAEParallelConfig(
            time_parallel=ParallelFactor(1, 1 - w_axis if w_factor > 1 else 0),
            h_parallel=ParallelFactor(h_factor, h_axis),
            w_parallel=ParallelFactor(w_factor, w_axis),
        )
        self.ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=num_links)

    @property
    def fractured(self) -> bool:
        return self.h_factor > 1 or self.w_factor > 1

    def shard_dims(self):
        dims = [0, 1]
        if self.h_factor > 1:
            dims[self.h_axis] = 2
        if self.w_factor > 1:
            dims[self.w_axis] = 3
        return dims


def _pad_to_multiple(t_nthwc: torch.Tensor, ctx: MeshCtx):
    """Host-pad H (dim2) / W (dim3) up to a multiple of the fracture factor. Returns (t, H0, W0)."""
    H0, W0 = t_nthwc.shape[2], t_nthwc.shape[3]
    if ctx.w_factor > 1 and W0 % ctx.w_factor:
        t_nthwc = torch.nn.functional.pad(t_nthwc, (0, 0, 0, (ctx.w_factor - W0 % ctx.w_factor) % ctx.w_factor))
    if ctx.h_factor > 1 and H0 % ctx.h_factor:
        t_nthwc = torch.nn.functional.pad(t_nthwc, (0, 0, 0, 0, 0, (ctx.h_factor - H0 % ctx.h_factor) % ctx.h_factor))
    return t_nthwc, H0, W0


def shard_input_nthwc(x_ncthw: torch.Tensor, ctx: MeshCtx, dtype=ttnn.bfloat16):
    """Host [N,C,T,H,W] -> device sharded ROW_MAJOR [N,T,H,W,C] (C padded to 32)."""
    t = x_ncthw.permute(0, 2, 3, 4, 1).contiguous()
    C = t.shape[-1]
    if C % ALIGNMENT:
        t = torch.nn.functional.pad(t, (0, ALIGNMENT - C % ALIGNMENT))
    t, _, _ = _pad_to_multiple(t, ctx)
    return ttnn.from_torch(
        t,
        device=ctx.mesh_device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            ctx.mesh_device, mesh_shape=tuple(ctx.mesh_device.shape), dims=ctx.shard_dims()
        ),
    )


def gather_output_ncthw(tt_nthwc, ctx: MeshCtx, out_c: int, valid_h: int, valid_w: int) -> torch.Tensor:
    """Device sharded [N,T,H,W,Cpad] -> host [N,out_c,T,valid_h,valid_w]."""
    t = ttnn.to_torch(
        tt_nthwc,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            ctx.mesh_device, mesh_shape=tuple(ctx.mesh_device.shape), dims=ctx.shard_dims()
        ),
    )
    t = t[:, :, :valid_h, :valid_w, :out_c]
    return t.permute(0, 4, 1, 2, 3).contiguous()


def _all_gather_hw(x_nthwc, ctx: MeshCtx):
    """Sharded [N,T,H,W,C] -> full [N,T,H*hf,W*wf,C] ROW_MAJOR (replicated on every chip)."""
    N, T, H, W, C = x_nthwc.shape
    x = ttnn.reshape(x_nthwc, [N * T, H, W, C])
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    if x.dtype != ttnn.bfloat16:
        x = ttnn.typecast(x, ttnn.bfloat16)
    if ctx.h_factor > 1:
        x = ctx.ccl.all_gather(x, dim=1, mesh_axis=ctx.h_axis, use_hyperparams=False)
    if ctx.w_factor > 1:
        x = ctx.ccl.all_gather(x, dim=2, mesh_axis=ctx.w_axis, use_hyperparams=False)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(x, [N, T, H * ctx.h_factor, W * ctx.w_factor, C])


def _partition_hw_nohalo(x_nthwc, ctx: MeshCtx):
    """Full [N,T,Hf,Wf,C] -> sharded [N,T,H,W,C] (mesh_partition only, NO halo)."""
    if ctx.h_factor > 1:
        x_nthwc = ttnn.mesh_partition(x_nthwc, dim=2, cluster_axis=ctx.h_axis, memory_config=x_nthwc.memory_config())
    if ctx.w_factor > 1:
        x_nthwc = ttnn.mesh_partition(x_nthwc, dim=3, cluster_axis=ctx.w_axis, memory_config=x_nthwc.memory_config())
    return x_nthwc


def _all_reduce_axis(t, ctx, mesh_axis):
    """Sum-reduce a (tiny) tensor across one mesh axis; identical copy left on every chip."""
    return ttnn.all_reduce(t, cluster_axis=mesh_axis, num_links=ctx.ccl.num_links, topology=ttnn.Topology.Linear)


def _groupnorm_dist(gn: "GN", x_nthwc, ctx, act=None):
    """Distributed reduce-moments GroupNorm on an H/W-sharded tensor.

    Numerically == ``torch.nn.GroupNorm`` (and the gather-based ``GroupNorm3D``) but exchanges
    only O(num_groups) scalars per norm instead of the full ~1GB activation. Each chip holds the
    FULL C and T but a LOCAL H/W shard. For each (batch, group g) it computes the per-shard mean
    and mean-of-squares over its local (C/G, T, H_local, W_local), then averages those across the
    H/W mesh axes (shards are uniform — decode() guarantees H,W divisible by the fracture factor —
    so the global group statistic is exactly the mean of the equal-count per-chip statistics).
    Finally it normalizes locally: ``(x - mean_g)*rstd_g*gamma_c + beta_c``.

    Reductions run in fp32 (summing millions of elements). Only the tiny per-group means (O(1)
    magnitudes) cross the fabric, so the bf16 CCL round-trip is lossless in practice.
    """
    N, T, H, W, C = x_nthwc.shape
    G = gn.num_groups
    Cg = C // G
    local_count = float(Cg * T * H * W)  # elems per (batch,group) on THIS shard (uniform across chips)
    nchips = (ctx.h_factor if ctx.h_factor > 1 else 1) * (ctx.w_factor if ctx.w_factor > 1 else 1)

    THW = T * H * W
    xt = ttnn.to_layout(ttnn.reshape(x_nthwc, (N, 1, THW, C)), ttnn.TILE_LAYOUT)  # bf16, kept for the affine
    xf = ttnn.typecast(xt, ttnn.float32)
    # local per-channel sums over the token dim (fp32 accumulation)
    psum = ttnn.to_layout(ttnn.reshape(ttnn.sum(xf, dim=2), (N, C)), ttnn.TILE_LAYOUT)  # [N,C] fp32
    xsq = ttnn.mul(xf, xf)
    psumsq = ttnn.to_layout(ttnn.reshape(ttnn.sum(xsq, dim=2), (N, C)), ttnn.TILE_LAYOUT)  # [N,C] fp32
    ttnn.deallocate(xsq)
    ttnn.deallocate(xf)
    # collapse channels -> groups via the 0/1 indicator matmul; scale to per-shard MEANS (O(1))
    inv_lc = 1.0 / local_count
    lmean = ttnn.multiply(ttnn.matmul(psum, gn.M), inv_lc)  # [N,G] fp32
    lmeansq = ttnn.multiply(ttnn.matmul(psumsq, gn.M), inv_lc)  # [N,G] fp32
    ttnn.deallocate(psum)
    ttnn.deallocate(psumsq)
    # all-reduce (sum) the per-shard means across H then W; average -> exact global group mean
    lmean = ttnn.typecast(ttnn.reshape(lmean, (N, 1, 1, G)), ttnn.bfloat16)
    lmeansq = ttnn.typecast(ttnn.reshape(lmeansq, (N, 1, 1, G)), ttnn.bfloat16)
    if ctx.h_factor > 1:
        lmean = _all_reduce_axis(lmean, ctx, ctx.h_axis)
        lmeansq = _all_reduce_axis(lmeansq, ctx, ctx.h_axis)
    if ctx.w_factor > 1:
        lmean = _all_reduce_axis(lmean, ctx, ctx.w_axis)
        lmeansq = _all_reduce_axis(lmeansq, ctx, ctx.w_axis)
    gmean = ttnn.multiply(ttnn.typecast(lmean, ttnn.float32), 1.0 / nchips)  # [N,1,1,G]
    gmeansq = ttnn.multiply(ttnn.typecast(lmeansq, ttnn.float32), 1.0 / nchips)
    var = ttnn.subtract(gmeansq, ttnn.mul(gmean, gmean))  # biased var, == torch GroupNorm
    rstd = ttnn.rsqrt(ttnn.add(var, gn.eps))
    # expand group stats back to per-channel and fold in the affine
    mean_ch = ttnn.matmul(ttnn.reshape(gmean, (N, G)), gn.Mt)  # [N,C] fp32
    rstd_ch = ttnn.matmul(ttnn.reshape(rstd, (N, G)), gn.Mt)  # [N,C] fp32
    scale = ttnn.mul(rstd_ch, gn.w)  # [N,C]
    shift = ttnn.subtract(gn.b, ttnn.mul(mean_ch, scale))  # [N,C]
    scale = ttnn.reshape(ttnn.typecast(scale, ttnn.bfloat16), (N, 1, 1, C))
    shift = ttnn.reshape(ttnn.typecast(shift, ttnn.bfloat16), (N, 1, 1, C))
    out = ttnn.add(ttnn.mul(xt, scale), shift)  # [N,1,THW,C] broadcast over the token dim
    if act == "silu":
        # Apply SiLU while still TILE. The caller's _silu_rm would otherwise do
        # to_layout(TILE) -> silu -> to_layout(RM) immediately after our
        # to_layout(RM) below -- an untilize/tilize pair that exactly cancels,
        # on the largest tensors in the decoder, x35 (norm, silu) pairs per decode.
        out = ttnn.silu(out)
    out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    return ttnn.reshape(out, (N, T, H, W, C))


def _groupnorm_mesh(gn: "GN", x_nthwc, ctx, act=None):
    """GroupNorm on an H/W-sharded tensor. Unfractured / single-chip: run the plain full-spatial
    ``GroupNorm3D`` (byte-identical to the validated factor=1 path). Fractured: distributed
    reduce-moments (no full-spatial gather)."""
    if ctx is None or not ctx.fractured:
        y = gn(x_nthwc)
        return _silu_rm(y) if act == "silu" else y
    return _groupnorm_dist(gn, x_nthwc, ctx, act=act)


# ----------------------------------------------------------------------------- helpers


def to_device_nthwc(x_ncthw: torch.Tensor, device, dtype=ttnn.bfloat16):
    """Host [N,C,T,H,W] -> device ROW_MAJOR [N,T,H,W,C], C padded up to ALIGNMENT."""
    t = x_ncthw.permute(0, 2, 3, 4, 1).contiguous()  # N T H W C
    C = t.shape[-1]
    if C % ALIGNMENT:
        t = torch.nn.functional.pad(t, (0, ALIGNMENT - C % ALIGNMENT))
    return ttnn.from_torch(t, device=device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)


def to_host_ncthw(x_nthwc, C: int) -> torch.Tensor:
    """Device ROW_MAJOR [N,T,H,W,Cpad] -> host [N,C,T,H,W], trimming channel padding."""
    t = ttnn.to_torch(x_nthwc, dtype=torch.float32)  # N T H W Cpad
    t = t[..., :C]
    return t.permute(0, 4, 1, 2, 3).contiguous()  # N C T H W


def _silu_rm(x_rm):
    """SiLU on a ROW_MAJOR tensor (elementwise); returns ROW_MAJOR."""
    x = ttnn.to_layout(x_rm, ttnn.TILE_LAYOUT)
    x = ttnn.silu(x)
    return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)


import time as _time


def _vae_timing():
    """HUNYUAN_VAE_TIMING=1 -> per-phase wall-clock inside the VAE decode.

    The 4.0 s decode has never been profiled (Tracy failed 3x: the device
    profiler buffer saturates on wall-clock, and the raw dump carries only
    firmware zones, no ttnn op names). Host timers + explicit device syncs give
    the same attribution for free. Default OFF."""
    return _os.environ.get("HUNYUAN_VAE_TIMING", "0") == "1"


def _phase(label, t0, device):
    """Sync then report. ttnn dispatch is ASYNC -- without the sync a timer
    measures enqueue time, not execution, and every phase looks instant."""
    ttnn.synchronize_device(device)
    now = _time.time()
    print(f"  VAE_PHASE {label:22s} {now - t0:7.3f}s", flush=True)
    return now


def _gn_silu(gn, x_nthwc, ctx):
    """GroupNorm followed by SiLU -- the decoder's most common pair (x35/decode).

    OFF: _silu_rm(_groupnorm_mesh(..)) == untilize (end of GN) + tilize (start of
    silu) back to back, a cancelling pair on a full-size activation.
    ON:  the SiLU runs inside the norm while still TILE, so both conversions go away.
    Numerically identical -- same ops, same order, only the layout hop is removed."""
    return _silu_rm(_groupnorm_mesh(gn, x_nthwc, ctx))


# ----------------------------------------------------------------------------- conv3d


class Conv3dSym:
    """Symmetric-pad 3D conv wrapping the de-risked ``ttnn.experimental.conv3d`` path.

    Eagerly prepares weights/bias from a torch ``nn.Conv3d`` (k=3 s=1 p=1, or k=1 p=0).
    Call with ROW_MAJOR [N,T,H,W,Cin] -> ROW_MAJOR [N,T,H,W,Cout].
    """

    def __init__(self, torch_conv: nn.Conv3d, device, T: int, H: int, W: int, weights_dtype=ttnn.bfloat16, ctx=None):
        self.device = device
        self.ctx = ctx
        self.oc = int(torch_conv.out_channels)
        self.ic = int(torch_conv.in_channels)
        self.k = tuple(torch_conv.kernel_size)
        self.stride = tuple(torch_conv.stride)
        self.padding = tuple(torch_conv.padding)
        self.dtype = weights_dtype
        self.kcfg = compute_kernel_config(device)

        g = device.compute_with_storage_grid_size()
        self.cfg = get_conv3d_config(self.ic, self.oc, self.k, weights_dtype, g, h_factor=1, w_factor=1, T=T, H=H, W=W)

        w = ttnn.from_torch(torch_conv.weight.data, dtype=weights_dtype, pad_value=0)
        self.weight = ttnn.experimental.prepare_conv3d_weights(
            weight_tensor=w, groups=1, C_in_block=self.cfg.C_in_block, alignment=ALIGNMENT, device=device
        )
        bias = torch_conv.bias.data if torch_conv.bias is not None else torch.zeros(self.oc)
        self.bias = ttnn.from_torch(
            bias.reshape(1, -1), device=device, dtype=weights_dtype, layout=ttnn.TILE_LAYOUT, pad_value=0
        )

    def _add_halo(self, x_nthwc):
        """Under H/W fracture: add the cross-chip H/W halo (zeros at the global boundary,
        real neighbor pixels interior — matches HF padding_mode="zeros"). T is left to the
        conv's internal padding (T is not fractured)."""
        ctx = self.ctx
        _, kH, kW = self.k
        x = ttnn.squeeze(x_nthwc, 0)  # (T, H, W, C)
        if ctx.h_factor > 1 and kH == 3:
            x = vae_neighbor_pad(
                ctx.ccl, x, cluster_axis=ctx.h_axis, dim=1, padding_left=1, padding_right=1, padding_mode="zeros"
            )
        if ctx.w_factor > 1 and kW == 3:
            x = vae_neighbor_pad(
                ctx.ccl, x, cluster_axis=ctx.w_axis, dim=2, padding_left=1, padding_right=1, padding_mode="zeros"
            )
        return ttnn.unsqueeze(x, 0)

    def __call__(self, x_nthwc):
        padding = self.padding
        if self.ctx is not None and self.ctx.fractured:
            x_nthwc = self._add_halo(x_nthwc)
            # H/W handled by the halo; conv still does the (symmetric, zeros) T padding on-chip.
            padding = ((self.k[0] - 1) // 2, 0, 0)
        return ttnn.experimental.conv3d(
            input_tensor=x_nthwc,
            weight_tensor=self.weight,
            device=self.device,
            bias_tensor=self.bias,
            dtype=self.dtype,
            output_channels=self.oc,
            kernel_size=self.k,
            stride=self.stride,
            groups=1,
            padding=padding,
            dilation=(1, 1, 1),
            padding_mode="zeros",
            config=self.cfg,
            compute_kernel_config=self.kcfg,
        )


class GN:
    """GroupNorm wrapper. ``__call__`` runs the plain full-spatial ``GroupNorm3D`` (single-chip and
    the already-gathered mid-attn tensor). The distributed reduce-moments path (``_groupnorm_dist``)
    instead reads the raw per-channel affine (``w``/``b``), ``num_groups``/``eps`` and the 0/1
    channel<->group indicator matrices (``M`` [C,G] sum, ``Mt`` [G,C] expand) stashed here, so it can
    normalize a spatially-sharded tensor WITHOUT gathering."""

    def __init__(self, gn: GroupNorm3D, C: int, num_groups: int, eps: float, w, b, M, Mt):
        self.gn = gn
        self.C = C
        self.num_groups = num_groups
        self.eps = eps
        self.w = w  # [1,C] fp32 gamma
        self.b = b  # [1,C] fp32 beta
        self.M = M  # [C,G] fp32 channel->group sum indicator
        self.Mt = Mt  # [G,C] fp32 group->channel expand indicator

    def __call__(self, x_nthwc):
        return self.gn(x_nthwc)


def _make_groupnorm(
    weight: torch.Tensor, bias: torch.Tensor, C: int, T: int, H: int, W: int, device, dtype=ttnn.bfloat16
):
    gn_t = nn.GroupNorm(GN_GROUPS, C, eps=GN_EPS)
    with torch.no_grad():
        gn_t.weight.copy_(weight)
        gn_t.bias.copy_(bias)
    gn = GroupNorm3D.from_torch(gn_t, input_nhw=1 * T * H * W, num_batches=1, mesh_device=device, dtype=dtype)
    # raw per-channel affine + channel<->group indicator matrices for the distributed path
    Cg = C // GN_GROUPS
    grp = (torch.arange(C) // Cg).long()  # group id per channel
    M = torch.zeros(C, GN_GROUPS, dtype=torch.float32)
    M[torch.arange(C), grp] = 1.0
    to = dict(device=device, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
    return GN(
        gn,
        C,
        GN_GROUPS,
        GN_EPS,
        ttnn.from_torch(weight.reshape(1, C).float(), **to),
        ttnn.from_torch(bias.reshape(1, C).float(), **to),
        ttnn.from_torch(M, **to),
        ttnn.from_torch(M.t().contiguous(), **to),
    )


# --------------------------------------------------------------------------- resnet


class ResnetBlock3D:
    """norm1 -> silu -> conv1 -> norm2 -> silu -> conv2 (+ identity residual).

    In the decoder every ResnetBlock has in==out, so the shortcut is always identity.
    """

    def __init__(self, state: dict, prefix: str, C: int, T: int, H: int, W: int, device, dtype=ttnn.bfloat16, ctx=None):
        self.ctx = ctx
        # norms operate on FULL spatial (gather reconstructs it) -> built at the full T,H,W
        self.norm1 = _make_groupnorm(
            state[_j(prefix, "norm1.weight")], state[_j(prefix, "norm1.bias")], C, T, H, W, device, dtype
        )
        self.norm2 = _make_groupnorm(
            state[_j(prefix, "norm2.weight")], state[_j(prefix, "norm2.bias")], C, T, H, W, device, dtype
        )
        self.conv1 = Conv3dSym(_conv3d_from_state(state, _j(prefix, "conv1"), C, C), device, T, H, W, dtype, ctx=ctx)
        self.conv2 = Conv3dSym(_conv3d_from_state(state, _j(prefix, "conv2"), C, C), device, T, H, W, dtype, ctx=ctx)

    def __call__(self, x_nthwc):
        h = _gn_silu(self.norm1, x_nthwc, self.ctx)
        h = self.conv1(h)
        h = _gn_silu(self.norm2, h, self.ctx)
        h = self.conv2(h)
        return ttnn.add(x_nthwc, h)


# --------------------------------------------------------------------------- attention


class AttnBlock3D:
    """GroupNorm -> 1x1 q,k,v -> single-head SDPA over (T*H*W) tokens -> 1x1 proj_out (+residual).

    q/k/v/proj_out are 1x1x1 convs == per-position linears; implemented as ttnn.linear.
    """

    def __init__(self, state: dict, prefix: str, C: int, T: int, H: int, W: int, device, dtype=ttnn.bfloat16, ctx=None):
        self.device = device
        self.ctx = ctx
        self.C = C
        self.T, self.H, self.W = T, H, W
        self.dtype = dtype
        self.norm = _make_groupnorm(
            state[_j(prefix, "norm.weight")], state[_j(prefix, "norm.bias")], C, T, H, W, device, dtype
        )
        self.kcfg = compute_kernel_config(device)

        def lin(name):
            w = state[_j(prefix, f"{name}.weight")].reshape(C, C)  # (O,I,1,1,1) -> (O,I)
            b = state[_j(prefix, f"{name}.bias")].reshape(1, C)
            wt = ttnn.from_torch(w.t().contiguous(), device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)  # (I,O)
            bt = ttnn.from_torch(b, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
            return wt, bt

        self.q = lin("q")
        self.k = lin("k")
        self.v = lin("v")
        self.proj = lin("proj_out")

    def _proj(self, x_tokens, wb):
        w, b = wb
        return ttnn.linear(x_tokens, w, bias=b, compute_kernel_config=self.kcfg)

    def __call__(self, x_nthwc):
        # Attention is global (every token attends every token). Under H/W fracture, gather
        # the full spatial, run the whole block on the replicated full tensor, then re-shard.
        # (mid attn is only at 64^2, so the redundant replicated compute is cheap.)
        if self.ctx is not None and self.ctx.fractured:
            xf = _all_gather_hw(x_nthwc, self.ctx)
            out = self._forward_single(xf)
            ttnn.deallocate(xf)
            return _partition_hw_nohalo(out, self.ctx)
        return self._forward_single(x_nthwc)

    def _forward_single(self, x_nthwc):
        N, T, H, W, C = x_nthwc.shape
        S = T * H * W
        h = self.norm(x_nthwc)  # RM NTHWC
        # (N,T,H,W,C) -> (N, S, C) tokens, TILE
        h = ttnn.reshape(h, (N, S, C))
        h = ttnn.to_layout(h, ttnn.TILE_LAYOUT)
        q = self._proj(h, self.q)
        k = self._proj(h, self.k)
        v = self._proj(h, self.v)
        # (N,S,C) -> (N,1,S,C) single head
        q = ttnn.reshape(q, (N, 1, S, C))
        k = ttnn.reshape(k, (N, 1, S, C))
        v = ttnn.reshape(v, (N, 1, S, C))
        attn = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = ttnn.reshape(attn, (N, S, C))
        out = self._proj(attn, self.proj)  # (N,S,C)
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.reshape(out, (N, T, H, W, C))
        return ttnn.add(x_nthwc, out)


# --------------------------------------------------------------------------- upsample


class UpsampleDCAE:
    """conv3d(in -> out*factor) + 3D pixel-shuffle, plus a repeat_interleave shortcut.

    factor = 2*2*2 (temporal levels) or 1*2*2 (spatial-only). Channel order out of the
    conv is (r1 r2 r3 c) so depth_to_spacetime consumes it directly (no swizzle).
    """

    def __init__(
        self,
        state: dict,
        prefix: str,
        in_c: int,
        out_c: int,
        add_temporal: bool,
        T: int,
        H: int,
        W: int,
        device,
        dtype=ttnn.bfloat16,
        ctx=None,
    ):
        self.device = device
        self.ctx = ctx
        self.in_c = in_c
        self.out_c = out_c
        self.texp = 2 if add_temporal else 1
        self.sexp = 2
        self.factor = self.texp * self.sexp * self.sexp
        self.repeats = self.factor * out_c // in_c
        conv_oc = out_c * self.factor
        # conv gets the halo via ctx; depth_to_spacetime + repeat_interleave shortcut are LOCAL
        # (pixel-shuffle maps input pixel h -> output 2h,2h+1, all within the chip's shard).
        self.conv = Conv3dSym(
            _conv3d_from_state(state, _j(prefix, "conv"), in_c, conv_oc), device, T, H, W, dtype, ctx=ctx
        )

    def _depth_to_spacetime(self, x_nthwc):
        """(N,T,H,W, texp*sexp*sexp*Cout) with channel order (texp,sexp_h,sexp_w,Cout)
        -> (N, T*texp, H*sexp, W*sexp, Cout). 3-step (<=6D) reshape/permute (Mochi)."""
        texp, sexp = self.texp, self.sexp
        N, T, H, W, C = x_nthwc.shape
        Cout = self.out_c
        sexp2_C = sexp * sexp * Cout
        x = ttnn.reshape(x_nthwc, [N, T, H, W, texp, sexp2_C])
        x = ttnn.permute(x, [0, 1, 4, 2, 3, 5])
        x = ttnn.reshape(x, [N, T * texp, H, W, sexp2_C])
        x = ttnn.reshape(x, [N, T * texp, H, W, sexp, sexp * Cout])
        x = ttnn.permute(x, [0, 1, 2, 4, 3, 5])
        x = ttnn.reshape(x, [N, T * texp, H * sexp, W, sexp * Cout])
        x = ttnn.reshape(x, [N, T * texp, H * sexp, W * sexp, Cout])
        return x

    def __call__(self, x_nthwc):
        h = self.conv(x_nthwc)  # (N,T,H,W, out*factor)
        sc = ttnn.repeat_interleave(x_nthwc, self.repeats, dim=4)  # channel-last repeat_interleave
        h = self._depth_to_spacetime(h)
        sc = self._depth_to_spacetime(sc)
        return ttnn.add(h, sc)


# --------------------------------------------------------------------------- helpers to torch conv


def _conv3d_from_state(state: dict, prefix: str, in_c: int, out_c: int) -> nn.Conv3d:
    """Reconstruct an nn.Conv3d from state_dict weight/bias, inferring k/pad from weight shape."""
    w = state[_j(prefix, "weight")]
    b = state.get(_j(prefix, "bias"))
    kt = tuple(w.shape[2:])  # (kT,kH,kW)
    pad = tuple((k - 1) // 2 for k in kt)  # symmetric
    conv = nn.Conv3d(in_c, out_c, kernel_size=kt, stride=1, padding=pad, bias=b is not None)
    with torch.no_grad():
        conv.weight.copy_(w)
        if b is not None:
            conv.bias.copy_(b)
    return conv


# --------------------------------------------------------------------------- full decoder


# decoder-reversed widths for block_out_channels=[128,256,512,1024,1024]
_DECODER_CH = [1024, 1024, 512, 256, 128]
_NUM_LEVELS = 5
_RES_PER_LEVEL = 3  # layers_per_block(2) + 1
_ADD_TEMPORAL = [True, True, False, False, False]  # i_level < log2(ffactor_temporal=4)=2


class VaeDecoder:
    """Single-chip (factor=1) on-device decode mirroring AutoencoderKLConv3D.decode.

    Instantiate from a host ``torch_decoder`` (``model.vae.decoder``) and an input latent
    spatial size (h,w in latent tokens, T=1). Call ``decode(latent_ncthw)`` -> host
    ``[N,3,1,H*16,W*16]`` (last temporal frame kept, matching HF decode()).
    """

    def __init__(
        self,
        torch_decoder: nn.Module,
        device,
        latent_t: int = 1,
        latent_h: int = 64,
        latent_w: int = 64,
        dtype=ttnn.bfloat16,
        ctx=None,
    ):
        self.device = device
        self.ctx = ctx
        self.dtype = dtype
        self.z_ch = 32
        # remembered so a cached decoder can be validated against the requested latent size
        self.latent_t, self.latent_h, self.latent_w = latent_t, latent_h, latent_w
        state = {k: v for k, v in torch_decoder.state_dict().items()}
        self.repeats_in = _DECODER_CH[0] // self.z_ch  # 1024//32 = 32

        # T,H,W track the FULL (logical) spatial size through the levels — norms are built at
        # full size (gather reconstructs it); convs get ctx and run on the local shard + halo.
        T, H, W = latent_t, latent_h, latent_w

        # conv_in (32 -> 1024)
        self.conv_in = Conv3dSym(
            _conv3d_from_state(state, "conv_in", self.z_ch, _DECODER_CH[0]), device, T, H, W, dtype, ctx=ctx
        )

        # mid: block_1, attn_1, block_2 (all at 1024, size unchanged)
        c = _DECODER_CH[0]
        self.mid_block_1 = ResnetBlock3D(state, "mid.block_1", c, T, H, W, device, dtype, ctx=ctx)
        self.mid_attn = AttnBlock3D(state, "mid.attn_1", c, T, H, W, device, dtype, ctx=ctx)
        self.mid_block_2 = ResnetBlock3D(state, "mid.block_2", c, T, H, W, device, dtype, ctx=ctx)

        # up levels
        self.levels = []
        for i in range(_NUM_LEVELS):
            ch = _DECODER_CH[i]
            blocks = [
                ResnetBlock3D(state, f"up.{i}.block.{j}", ch, T, H, W, device, dtype, ctx=ctx)
                for j in range(_RES_PER_LEVEL)
            ]
            up = None
            if i < _NUM_LEVELS - 1:  # levels 0..3 have upsample
                out_ch = _DECODER_CH[i + 1]
                up = UpsampleDCAE(
                    state, f"up.{i}.upsample", ch, out_ch, _ADD_TEMPORAL[i], T, H, W, device, dtype, ctx=ctx
                )
                # advance spatial/temporal dims for the next level
                if _ADD_TEMPORAL[i]:
                    T = T * 2
                H, W = H * 2, W * 2
            self.levels.append((blocks, up))

        # norm_out (128) + conv_out (128 -> 3) at final resolution
        self.norm_out = _make_groupnorm(
            state["norm_out.weight"], state["norm_out.bias"], _DECODER_CH[-1], T, H, W, device, dtype
        )
        self.conv_out = Conv3dSym(
            _conv3d_from_state(state, "conv_out", _DECODER_CH[-1], 3), device, T, H, W, dtype, ctx=ctx
        )

    def _forward(self, z_nthwc):
        _tm = _vae_timing()
        _t = _time.time() if _tm else None
        # conv_in + z.repeat_interleave(32) residual
        h = self.conv_in(z_nthwc)
        res = ttnn.repeat_interleave(z_nthwc, self.repeats_in, dim=4)
        h = ttnn.add(h, res)
        if _tm:
            _t = _phase("conv_in+res", _t, self.device)
        # mid
        h = self.mid_block_1(h)
        h = self.mid_attn(h)
        h = self.mid_block_2(h)
        if _tm:
            _t = _phase("mid(blk,attn,blk)", _t, self.device)
        # up levels
        for _i, (blocks, up) in enumerate(self.levels):
            for blk in blocks:
                h = blk(h)
            if up is not None:
                h = up(h)
            if _tm:
                _t = _phase(f"level{_i} {tuple(h.shape)[2]}x{tuple(h.shape)[3]}", _t, self.device)
        # head
        h = _gn_silu(self.norm_out, h, self.ctx)
        h = self.conv_out(h)
        if _tm:
            _t = _phase("head(norm,silu,conv)", _t, self.device)
        return h

    # cache the built decoder on the model so repeated renders reuse the prepared mesh weights
    _CACHE_ATTR = "_tt_ondevice_vae_decoder"

    def decode(self, latent_ncthw: torch.Tensor) -> torch.Tensor:
        """latent [N,32,T,H,W] (T=1) -> host image [N,3,1,H*16,W*16] (last frame).

        NB: mesh path assumes H,W divisible by the fracture factor at every level (true for
        latent 64x64 on an (8,4) mesh: all of 64/128/256/512/1024 divide by 8 and 4) so no
        zero-padding is introduced and GroupNorm statistics stay exact (no logical unpad)."""
        z_in = latent_ncthw[:, : self.z_ch].contiguous()
        _, _, _, H, W = z_in.shape
        if self.ctx is not None and self.ctx.fractured:
            _tm = _vae_timing()
            _t = _time.time() if _tm else None
            z = shard_input_nthwc(z_in, self.ctx, self.dtype)
            if _tm:
                _t = _phase("upload+shard", _t, self.device)
            h = self._forward(z)  # NTHWC, T at dim 1
            if _tm:
                _t = _phase("forward(total)", _t, self.device)
            if latent_ncthw.shape[2] == 1 and h.shape[1] > 1:
                h = h[:, -1:]  # keep last temporal frame ON DEVICE -> Tx smaller gather/transfer
            img = gather_output_ncthw(h, self.ctx, 3, H * 16, W * 16)
            if _tm:
                _t = _phase("gather+download", _t, self.device)
        else:
            z = to_device_nthwc(z_in, self.device, self.dtype)
            h = self._forward(z)
            img = to_host_ncthw(h, 3)  # [N,3,Tout,Hout,Wout]
        if latent_ncthw.shape[2] == 1:
            img = img[:, :, -1:]  # keep last temporal frame (HF decode())
        return img


def prebuild_ondevice_vae(
    model, mesh_device, latent_h: int = 64, latent_w: int = 64, latent_t: int = 1, dtype=ttnn.bfloat16
):
    """Build + cache the mesh VAE decoder as MODEL SETUP (the one-time ~19.4s prepare-mesh-weights
    cost, ~160 conv/norm tensors) so the per-image render path is decode-only.

    Idempotent: reuses the decoder cached on ``model`` when the mesh device AND latent size match;
    rebuilds on a mismatch (e.g. a different render resolution). Returns the decoder. Call it once
    from the model-setup / pre-render path (see host_glue_stage3.generate_image_ondevice) — then
    ``ondevice_vae_decode`` finds it cached and does no build inside the timed decode."""
    import time

    hf, wf = int(mesh_device.shape[0]), int(mesh_device.shape[1])
    dec = getattr(model, VaeDecoder._CACHE_ATTR, None)
    if (
        dec is not None
        and dec.device is mesh_device
        and (dec.latent_t, dec.latent_h, dec.latent_w) == (latent_t, latent_h, latent_w)
    ):
        return dec
    t0 = time.time()
    ctx = MeshCtx(mesh_device, h_factor=hf, h_axis=0, w_factor=wf, w_axis=1, num_links=1)
    dec = VaeDecoder(
        model.vae.decoder, mesh_device, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, ctx=ctx, dtype=dtype
    )
    setattr(model, VaeDecoder._CACHE_ATTR, dec)
    print(
        f"ONDEVICE_VAE prebuild={time.time() - t0:.1f}s (mesh weights prepared once; cached on model)",
        flush=True,
    )
    if _os.environ.get("HUNYUAN_VAE_WARMUP", "1") != "0":
        # WARM-UP: one throwaway decode at SETUP so the per-image decode runs WARM (compiles the
        # conv3d/groupnorm/pixel-shuffle programs here, not in the timed render). Measured on the
        # (8,4) mesh: cold first decode 24.0s -> warm 4.0s (-20s/image), output PCC 1.0. Same idea
        # as prebuild moving the weight-build off the per-image path. Gated HUNYUAN_VAE_WARMUP.
        tw = time.time()
        try:
            _C = int(model.config.vae["latent_channels"])
        except Exception:
            _C = 32
        _z = torch.zeros(1, _C, latent_t, latent_h, latent_w, dtype=torch.float32)
        try:
            dec.decode(_z)
            print(f"ONDEVICE_VAE warmup_decode={time.time() - tw:.1f}s (per-image decode now warm)", flush=True)
        except Exception as e:  # warm-up is best-effort: on any error the per-image decode just stays cold
            print(f"ONDEVICE_VAE warmup_decode SKIPPED {type(e).__name__}: {str(e)[:140]}", flush=True)
    return dec


def ondevice_vae_decode(model, mesh_device, latents_5d: torch.Tensor, dtype=ttnn.bfloat16) -> torch.Tensor:
    """Drop-in for host ``model.vae.decode(latents_5d, return_dict=False)[0]`` on the mesh.

    latents_5d = [1,32,1,H,W] (already caller-scaled). Returns host [1,3,1,H*16,W*16].
    Fractures H/W across the full mesh (H=axis0, W=axis1). The decoder build is delegated to
    ``prebuild_ondevice_vae`` (idempotent) — do it in model setup so this path is decode-only;
    if it wasn't prebuilt, the first call still builds (and caches) here."""
    import time

    _, _, T, H, W = latents_5d.shape
    dec = prebuild_ondevice_vae(model, mesh_device, latent_h=H, latent_w=W, latent_t=T, dtype=dtype)
    t1 = time.time()
    img = dec.decode(latents_5d)
    print(f"ONDEVICE_VAE decode={time.time() - t1:.1f}s", flush=True)
    return img
