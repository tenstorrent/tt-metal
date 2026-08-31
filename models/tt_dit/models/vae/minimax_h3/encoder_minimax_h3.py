# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 video VAE encoder: a causal 3D CNN, 6 levels, 16x spatial / 4x temporal.

The module tree **mirrors the diffusers key names one-for-one**
(``encoder.down_blocks.<L>.resnets.<i>.conv1``, ``...downsamplers.0.conv``,
``...resnets.0.conv_shortcut``), so almost nothing needs a ``_prepare_torch_state``
override. That is deliberate: the available checkpoint at
``/data/cglagovich/MiniMax-H3-diffusers/vae`` is diffusers-converted, not the raw
MiniMax layout.

Two facts drive the plumbing:

* ``ttnn.experimental.conv3d`` requires **ROW_MAJOR** input.
* Every norm is per *frame*: ``MiniMaxH3VideoGroupNorm`` in the reference folds T into
  the batch axis so no statistic ever mixes across frames.

The norm is :class:`MiniMaxH3DistributedFrameGroupNorm`, which computes the per-(frame,
group) statistics itself. Three alternatives were tried and all lose, so do not re-derive
them: ``ttnn.group_norm`` with T as batch is 2.7x slower and bf16-only, making every norm a
bf16 island in an otherwise fp32 encoder; a fused distributed GroupNorm device op is 1.6x
slower per frame; carrying the resnet chain in TILE to avoid the round trip is a wash.

Each norm is still specialised to its ``(T, H, W)`` at construction, because the divisor is
the *global* element count per (frame, group) and only the constructor knows the mesh factor.
That costs nothing: the encoder has exactly one activation shape per level, since the
reference always encodes a 256x256 tile over a 17-frame clip.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.module import Module, ModuleList, Parameter
from .conv_minimax_h3 import MiniMaxH3CausalConv3d

MINIMAX_H3_VAE_NUM_GROUPS = 32
MINIMAX_H3_VAE_NORM_EPS = 1e-6

# ``ttnn.add(..., activations=[...])`` takes EltwiseUnaryWithParam, not a string.
_SILU_ACTIVATION = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)

# Compute the group variance as E[x^2] - E[x]^2 rather than centring first. Saves one full
# pass over the activation per norm site (four -> three). The class docstring explains why
# the two-pass form is kept: the group means are not near zero, so this is exactly the
# cancellation Welford exists to avoid. Contained here by doing the subtraction on the
# per-(frame, group) stats in fp32 -- 32 scalars per frame -- so only the *sums* are bf16.
MINIMAX_H3_ONE_PASS_VARIANCE = True


class MiniMaxH3DistributedFrameGroupNorm(Module):
    """Per-frame GroupNorm whose spatial extent is sharded across a mesh axis.

    Takes and returns ``(1, T, H, W, C)`` ROW_MAJOR, with ``H`` the **local** height.

    No existing primitive does this job:

    * ``ttnn.group_norm`` has no notion of the mesh, so a sharded ``H`` gives each device a
      statistic over its own strip only.
    * A fused distributed GroupNorm device op does not fit: it hard-rejects ``N > 1``, and
      here ``N`` is ``T`` (17/9/5) because the statistic is per frame. It also takes a
      single ``cluster_axis``, so it cannot reduce over both mesh axes -- and it measures
      1.6x slower than this anyway.

    So the statistics are computed directly: per ``(frame, group)`` local sums, an
    all-reduce of **only those** -- ``T x 32`` scalars, against a full-activation gather --
    then an elementwise normalise. Channel-to-group contraction is a matmul with a 0/1
    matrix, and its transpose spreads the result back over channels.

    The default form is one pass on ``E[x^2] - E[x]^2``
    (:data:`MINIMAX_H3_ONE_PASS_VARIANCE`). The two-pass centred form is kept behind that
    flag because the group means here are *not* near zero -- exactly the cancellation
    ``GroupNorm3D`` uses Welford to avoid. What contains it is that the subtraction happens
    on the ``(T,1,1,G)`` stats tensor in fp32, never on the activation; PCC gates the
    difference at 0.02 pp.

    Runs in the activation dtype, so fp32 activations stay fp32 -- unlike
    ``ttnn.group_norm``, which is bf16-only and would floor the encoder's precision.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        num_frames: int,
        height: int,
        width: int,
        mesh_device: ttnn.MeshDevice,
        spatial_factor: int = 1,
        cluster_axis: int = 0,
        ccl_manager=None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        assert num_channels % MINIMAX_H3_VAE_NUM_GROUPS == 0
        self.num_channels = num_channels
        self.num_groups = MINIMAX_H3_VAE_NUM_GROUPS
        self.eps = MINIMAX_H3_VAE_NORM_EPS
        self.num_frames = num_frames
        self.local_height = height
        self.width = width
        self.spatial_factor = spatial_factor
        self.cluster_axis = cluster_axis
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager

        # The divisor is the GLOBAL element count per (frame, group), so a device normalises by
        # more elements than it holds.
        self.height = height
        self.elements_per_group = (num_channels // self.num_groups) * height * spatial_factor * width

        self.weight = Parameter(total_shape=[1, 1, 1, num_channels], device=mesh_device)
        self.bias = Parameter(total_shape=[1, 1, 1, num_channels], device=mesh_device)
        self.to_groups = Parameter(total_shape=[1, 1, num_channels, self.num_groups], device=mesh_device)
        self.from_groups = Parameter(total_shape=[1, 1, self.num_groups, num_channels], device=mesh_device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        state["weight"] = state["weight"].reshape(1, 1, 1, self.num_channels)
        state["bias"] = state["bias"].reshape(1, 1, 1, self.num_channels)
        per_group = self.num_channels // self.num_groups
        selector = torch.zeros(self.num_channels, self.num_groups)
        for group in range(self.num_groups):
            selector[group * per_group : (group + 1) * per_group, group] = 1.0
        state["to_groups"] = selector.reshape(1, 1, self.num_channels, self.num_groups)
        state["from_groups"] = selector.t().contiguous().reshape(1, 1, self.num_groups, self.num_channels)

    def _group_sums(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """``(T,1,HW_local,C)`` -> global per-``(frame, group)`` sums, shaped ``(T,1,1,G)``."""
        channel_sums = ttnn.sum(x_flat, dim=2, keepdim=True)
        group_sums = ttnn.matmul(channel_sums, self.to_groups.data)
        if self.spatial_factor <= 1:
            # Unsharded: the local sum is already the global one, and there is no ccl_manager.
            return group_sums
        # tt_dit exposes no all-reduce (not one call anywhere in the tree), so gather on the
        # singleton dim and sum locally. At T x 32 scalars the distinction is immaterial.
        gathered = self.ccl_manager.all_gather(group_sums, dim=1, mesh_axis=self.cluster_axis, use_hyperparams=False)
        return ttnn.sum(gathered, dim=1, keepdim=True)

    def _spread(self, per_group: ttnn.Tensor) -> ttnn.Tensor:
        """``(T,1,1,G)`` -> ``(T,1,1,C)``, each channel taking its own group's value."""
        return ttnn.matmul(per_group, self.from_groups.data)

    def forward(self, x_BTHWC: ttnn.Tensor, *, fuse_silu: bool = False) -> ttnn.Tensor:
        """``(1,T,H,W,C)`` ROW_MAJOR in and out.

        ``fuse_silu`` folds the SiLU into the final broadcast add. Run separately on the ROW_MAJOR
        output it is a 285 MB unary costing 3.7 ms; fused onto an op that already runs, it is free.
        """
        T, H, W, C = self.num_frames, self.local_height, self.width, self.num_channels
        B, xt, xh, xw, xc = x_BTHWC.shape
        assert B == 1, f"one unit per device, got B={B}"
        assert (xt, xh, xw) == (T, H, W), f"norm built for local (T,H,W)=({T},{H},{W}), got ({xt},{xh},{xw})"
        x = ttnn.reshape(x_BTHWC, (T, 1, H * W, C))
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        scale = 1.0 / self.elements_per_group
        if MINIMAX_H3_ONE_PASS_VARIANCE:
            # E[x^2] - E[x]^2, which never materialises ``x - mean``: three full passes over
            # the activation (x*x, x*gamma, +beta) instead of four. The cancellation this
            # form is known for is contained by doing the subtraction on the (T,1,1,G) stats
            # tensor in fp32 -- 32 scalars per frame, so the cast is free -- rather than on
            # the activation. Gated by PCC, not assumed: see the class docstring on the
            # two-pass form.
            sum_x = ttnn.typecast(self._group_sums(x), ttnn.float32)
            sum_xx = ttnn.typecast(self._group_sums(ttnn.multiply(x, x)), ttnn.float32)
            mean_g = ttnn.multiply(sum_x, scale)
            var_g = ttnn.subtract(ttnn.multiply(sum_xx, scale), ttnn.multiply(mean_g, mean_g))
            inv_g = ttnn.typecast(ttnn.rsqrt(ttnn.add(var_g, self.eps)), x.get_dtype())
            gamma = ttnn.multiply(self._spread(inv_g), self.weight.data)
            beta = ttnn.subtract(
                self.bias.data, ttnn.multiply(self._spread(ttnn.typecast(mean_g, x.get_dtype())), gamma)
            )
            scaled = ttnn.multiply(x, gamma)
        else:
            mean = self._spread(ttnn.multiply(self._group_sums(x), scale))
            centred = ttnn.subtract(x, mean)
            variance = self._spread(ttnn.multiply(self._group_sums(ttnn.multiply(centred, centred)), scale))
            # Fold ``weight`` into the inverse standard deviation on the (T,1,1,C) stats
            # tensor. ``normed * weight + bias`` costs three full passes over the activation;
            # ``centred * gamma + bias`` costs two, and the extra work lands on a tensor
            # 65536x smaller.
            gamma = ttnn.multiply(ttnn.rsqrt(ttnn.add(variance, self.eps)), self.weight.data)
            beta = self.bias.data
            scaled = ttnn.multiply(centred, gamma)
        out = ttnn.add(scaled, beta, activations=[_SILU_ACTIVATION]) if fuse_silu else ttnn.add(scaled, beta)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(out, (1, T, H, W, C))


def _gn_hw_sharded(
    norm: MiniMaxH3DistributedFrameGroupNorm, x_BTHWC: ttnn.Tensor, parallel_config, ccl_manager
) -> ttnn.Tensor:
    """Per-frame GroupNorm on an H/W-sharded tensor: gather the spatial extent, norm, re-shard.

    The shape of ``upsampler/latent_upsampler_ltx.py:_gn_hw_sharded``, which is the only
    in-tree solution to this problem, so the norms stay built at the **global** H/W while the
    convs are built at the local shard.

    Minus that function's crop/re-pad branch: it exists for mesh-factor
    padding, and H3 never has any. Its spatial extents are dyadic (256/128/64/32/16), so at
    factor 2/4/8 every one of the eight norm sites divides its mesh factor exactly *and*
    every local ``H*W`` is a multiple of 32. The assertion below is what keeps that true --
    if it ever fires, port the crop branch rather than relaxing it, because normalising over
    padding zeros is silently wrong (it is the live TODO at ``vae_mochi.py:270``).
    """
    if parallel_config is None:
        return norm(x_BTHWC)
    h_parallel, w_parallel = parallel_config.height_parallel, parallel_config.width_parallel
    if h_parallel.factor <= 1 and w_parallel.factor <= 1:
        return norm(x_BTHWC)

    if x_BTHWC.layout != ttnn.ROW_MAJOR_LAYOUT:
        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
    for dim, spatial in ((2, h_parallel), (3, w_parallel)):
        if spatial.factor > 1:
            assert x_BTHWC.shape[dim] * spatial.factor == (norm.height if dim == 2 else norm.width), (
                f"dim {dim} shard {x_BTHWC.shape[dim]} x factor {spatial.factor} != "
                f"global {norm.height if dim == 2 else norm.width}; mesh-factor padding is "
                "unsupported here and normalising over pad zeros is silently wrong"
            )
            x_BTHWC = ccl_manager.all_gather(x_BTHWC, dim=dim, mesh_axis=spatial.mesh_axis, use_hyperparams=False)

    x_BTHWC = norm(x_BTHWC)

    for dim, spatial in ((2, h_parallel), (3, w_parallel)):
        if spatial.factor > 1:
            # ROW_MAJOR is required: a sub-tile-wide shard cannot be sliced out of a tilized
            # tensor, so tilize-then-partition fails.
            x_BTHWC = ttnn.mesh_partition(x_BTHWC, dim=dim, cluster_axis=spatial.mesh_axis)
    return x_BTHWC


def _hw_sharded(parallel_config) -> bool:
    if parallel_config is None:
        return False
    return parallel_config.height_parallel.factor > 1 or parallel_config.width_parallel.factor > 1


def _norm_silu(
    norm: MiniMaxH3DistributedFrameGroupNorm,
    x_BTHWC: ttnn.Tensor,
    dtype: ttnn.DataType = ttnn.bfloat16,
    *,
    parallel_config=None,
    ccl_manager=None,
) -> ttnn.Tensor:
    """``silu(norm(x))``, returned ROW_MAJOR in ``dtype``, ready for the next conv.

    The norm runs in the activation dtype, so this is not a precision boundary; the cast is
    kept explicit because ``conv3d`` requires input and weight dtypes to match exactly.
    """
    if not _hw_sharded(parallel_config):
        # The norm folds the SiLU into its own final add. _gn_hw_sharded is bypassed only
        # when there is no sharding to gather for, so the sharded path is untouched.
        h = norm(x_BTHWC, fuse_silu=True)
        if h.get_dtype() != dtype:
            h = ttnn.typecast(h, dtype)
        return h
    h = ttnn.silu(_gn_hw_sharded(norm, x_BTHWC, parallel_config, ccl_manager))
    if h.get_dtype() != dtype:
        h = ttnn.typecast(h, dtype)
    if h.layout != ttnn.ROW_MAJOR_LAYOUT:
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
    return h


class MiniMaxH3ResnetBlock3d(Module):
    """``norm1 -> silu -> conv1 -> norm2 -> silu -> conv2``, plus a k1 ``conv_shortcut``.

    Only three blocks in the checkpoint carry ``conv_shortcut`` (``down_blocks.{1,3,5}
    .resnets.0``), so it is constructed conditionally to keep a strict load exact.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_frames: int,
        height: int,
        width: int,
        temporal_taps: int = 3,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        parallel_config=None,
        ccl_manager=None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.num_frames = num_frames
        self.height = height
        self.width = width
        conv_kwargs = dict(
            temporal_taps=temporal_taps,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        # Norms are built at the GLOBAL H/W: _gn_hw_sharded gathers the spatial extent
        # before calling them. Only the convs see the per-device shard.
        norm_kwargs = dict(num_frames=num_frames, height=height, width=width, mesh_device=mesh_device)
        # A resnet never changes T/H/W, so both norms share one shape.
        self.norm1 = MiniMaxH3DistributedFrameGroupNorm(in_channels, **norm_kwargs)
        self.conv1 = MiniMaxH3CausalConv3d(in_channels, out_channels, kernel_size=3, spatial_padding=1, **conv_kwargs)
        self.norm2 = MiniMaxH3DistributedFrameGroupNorm(out_channels, **norm_kwargs)
        self.conv2 = MiniMaxH3CausalConv3d(out_channels, out_channels, kernel_size=3, spatial_padding=1, **conv_kwargs)
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3CausalConv3d(
                in_channels, out_channels, kernel_size=1, spatial_padding=0, **conv_kwargs
            )

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        """``(1,T,H,W,C)`` ROW_MAJOR in and out -- what ``conv3d`` requires on both sides.

        Carrying the chain in TILE instead, to make the residual add cheaper, measures as a
        wash: it moves the cost into Tilize rather than removing it.
        """
        norm_kwargs = dict(parallel_config=self.parallel_config, ccl_manager=self.ccl_manager)
        h = self.conv1(_norm_silu(self.norm1, x_BTHWC, self.dtype, **norm_kwargs))
        h = self.conv2(_norm_silu(self.norm2, h, self.dtype, **norm_kwargs))
        # The residual carries the caller's dtype, which need not be ours: `_norm_silu` casts the
        # main path but nothing casts this one. An fp32 input then either reaches conv_shortcut,
        # which rejects a dtype mismatch against its weights, or reaches the add as an fp32 term
        # against a bf16 one, which does not fail -- it returns garbage that varies run to run.
        residual = x_BTHWC
        if residual.get_dtype() != self.dtype:
            residual = ttnn.typecast(residual, self.dtype)
        if self.in_channels != self.out_channels:
            residual = self.conv_shortcut(residual)
        return ttnn.add(residual, h)


class MiniMaxH3Downsample3d(Module):
    """Strided conv with H3's asymmetric bottom/right reflect pre-pad.

    The conv itself carries no spatial padding, so the pre-pad is what makes the
    output exactly ``ceil(size / 2)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        temporal_taps: int = 3,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        parallel_config=None,
        ccl_manager=None,
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        # A collapsed temporal axis cannot be strided: one real frame stays one frame.
        effective_temporal_stride = 1 if temporal_taps == 1 else temporal_stride
        self.conv = MiniMaxH3CausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(effective_temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            # H3's asymmetric bottom/right reflect pre-pad, which is what makes the strided
            # conv land on exactly ceil(size/2). Expressed as the conv's trailing pad rather
            # than a separate pre-pad so one path covers sharded and unsharded: under H/W
            # sharding only the device owning the global bottom/right edge may reflect, while
            # the interior devices need a real halo row from their neighbour.
            trailing_spatial_padding=1 if spatial_stride == 2 else 0,
            temporal_taps=temporal_taps,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv(x_BTHWC)


class MiniMaxH3DownBlock3d(Module):
    """``layers_per_block`` resnets, then an optional downsampler."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        num_frames: int,
        height: int,
        width: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        temporal_taps: int = 3,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        parallel_config=None,
        ccl_manager=None,
    ) -> None:
        super().__init__()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.out_channels = out_channels
        self.resnets = ModuleList(
            [
                MiniMaxH3ResnetBlock3d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    num_frames=num_frames,
                    height=height,
                    width=width,
                    temporal_taps=temporal_taps,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for i in range(num_layers)
            ]
        )
        self.has_downsamplers = temporal_downsample_factor * spatial_downsample_factor > 1
        if self.has_downsamplers:
            self.downsamplers = ModuleList(
                [
                    MiniMaxH3Downsample3d(
                        out_channels,
                        out_channels,
                        temporal_stride=temporal_downsample_factor,
                        spatial_stride=spatial_downsample_factor,
                        temporal_taps=temporal_taps,
                        mesh_device=mesh_device,
                        dtype=dtype,
                        parallel_config=parallel_config,
                        ccl_manager=ccl_manager,
                    )
                ]
            )

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        for resnet in self.resnets:
            x_BTHWC = resnet(x_BTHWC)
        if self.has_downsamplers:
            for downsampler in self.downsamplers:
                x_BTHWC = downsampler(x_BTHWC)
        return x_BTHWC


class MiniMaxH3Encoder3d(Module):
    """``conv_in -> 6 down_blocks -> norm_out + silu -> conv_out``.

    Returns the ``2 * latent_channels`` moments **before** ``quant_conv``, matching the
    reference ``MiniMaxH3VideoEncoder3d``. Sampling stays on host so the seed-42
    posterior draw the released model conditions on stays reproducible; see
    ``pipelines/minimax_h3/conditioning.py``.
    """

    def __init__(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        temporal_taps: int = 3,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        parallel_config=None,
        ccl_manager=None,
        pixel_norm: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    ) -> None:
        super().__init__()
        self.temporal_taps = temporal_taps
        # height/width here are the GLOBAL extents. Norms are built at them and gather to
        # them; the convs derive their per-device shard from parallel_config.
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.in_channels = in_channels
        self.dtype = dtype
        self.input_shape = (num_frames, height, width)

        conv_kwargs = dict(
            temporal_taps=temporal_taps,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        # `pixel_norm` reaches conv_in alone: it folds the pixel normalization into the first
        # conv (see MiniMaxH3CausalConv3d._prepare_torch_state), so this encoder consumes raw
        # 0..255 pixels; every later conv sees activations and stays untouched.
        self.conv_in = MiniMaxH3CausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, spatial_padding=1, pixel_norm=pixel_norm, **conv_kwargs
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        blocks = []
        t, h, w = num_frames, height, width
        for i in range(len(block_out_channels)):
            # Resnets at level i see the shape *entering* the level; the downsampler at
            # the end of the level is what shrinks it.
            blocks.append(
                MiniMaxH3DownBlock3d(
                    block_in_channels[i],
                    block_out_channels[i],
                    num_layers=layers_per_block,
                    num_frames=t,
                    height=h,
                    width=w,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    temporal_taps=temporal_taps,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
            )
            if temporal_downsample_factors[i] * spatial_downsample_factors[i] > 1:
                # A collapsed temporal axis cannot shrink: one real frame stays one.
                if temporal_taps > 1:
                    t = -(-t // temporal_downsample_factors[i])
                h = -(-h // spatial_downsample_factors[i])
                w = -(-w // spatial_downsample_factors[i])
        self.down_blocks = ModuleList(blocks)
        self.latent_shape = (t, h, w)

        self.norm_out = MiniMaxH3DistributedFrameGroupNorm(
            block_out_channels[-1], num_frames=t, height=h, width=w, mesh_device=mesh_device
        )
        self.conv_out = MiniMaxH3CausalConv3d(
            block_out_channels[-1], out_channels, kernel_size=3, spatial_padding=1, **conv_kwargs
        )

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        # conv3d requires input and weight dtypes to match exactly, so accept whatever the
        # caller has and cast once here rather than making every call site know the
        # encoder's compute dtype.
        if x_BTHWC.get_dtype() != self.dtype:
            x_BTHWC = ttnn.typecast(x_BTHWC, self.dtype)
        h = self.conv_in(x_BTHWC)
        for down_block in self.down_blocks:
            h = down_block(h)
        return self.conv_out(
            _norm_silu(self.norm_out, h, self.dtype, parallel_config=self.parallel_config, ccl_manager=self.ccl_manager)
        )
