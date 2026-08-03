# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 video VAE encoder: a causal 3D CNN, 6 levels, 16x spatial / 4x temporal.

The module tree **mirrors the diffusers key names one-for-one**
(``encoder.down_blocks.<L>.resnets.<i>.conv1``, ``...downsamplers.0.conv``,
``...resnets.0.conv_shortcut``), so almost nothing needs a ``_prepare_torch_state``
override. That is deliberate: the available checkpoint at
``/data/cglagovich/MiniMax-H3-diffusers/vae`` is diffusers-converted, not the raw
MiniMax layout an earlier draft assumed.

Two facts drive the plumbing:

* ``ttnn.experimental.conv3d`` requires **ROW_MAJOR** input.
* ``ttnn.group_norm`` has no fp32 path and needs a tilized input, so every norm is a
  **bf16** island inside an otherwise fp32 encoder. :func:`_norm_silu` is the single
  place that boundary is crossed, and it is the encoder's precision floor.

``MiniMaxH3VideoGroupNorm`` in the reference folds T into the batch axis so statistics
never mix across frames. That is obtained here by reusing ``GroupNorm3D`` **with T fed
as its batch axis**: its ``dims=3`` pooling over ``(C_group, T', H, W)`` with a
singleton ``T'`` degenerates to exactly per-frame ``(C_group, H, W)`` statistics.

Reusing ``GroupNorm3D`` rather than the plain 2D ``GroupNorm`` is load-bearing, not
stylistic. It pins ``core_grid`` at construction via
``ttnn.determine_expected_group_norm_dram_grid_size``, which guarantees **uniform
multicast groups**. A grid that merely satisfies the ``Ht % nvr == 0`` divisibility
rules can still **deadlock the mcast at small spatial sizes** -- observed: a heuristic
grid search picked ``CoreGrid(8, 5)`` for (C=128, T=5, HW=256) and hung the device hard
enough to need ``tt-smi -glx_reset``, where the pinned API picks ``(8, 10)`` and runs
clean. ``num_out_blocks`` is then tuned per site (:data:`MINIMAX_H3_GN_OUT_BLOCKS`),
because the built-in ``-1`` heuristic under-chunks at large spatial and overflows L1.

The consequence is that each norm is specialised to its ``(T, H, W)`` at construction.
That costs nothing: the encoder has exactly one activation shape per level, since the
reference always encodes a 256x256 tile over a 17-frame clip.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.module import Module, ModuleList, Parameter
from ....layers.normalization import GroupNorm3D
from .conv_minimax_h3 import MiniMaxH3CausalConv3d, reflect_pad_bottom_right

MINIMAX_H3_VAE_NUM_GROUPS = 32
MINIMAX_H3_VAE_NORM_EPS = 1e-6

# Measured on BH Galaxy: the minimal num_out_blocks whose circular buffers fit L1, per
# (C, T, H, W) site. -1 is GroupNorm3D's built-in heuristic and is fine almost
# everywhere; the three large-spatial clip sites need explicit chunking, e.g.
# (256, 17, 128, 128) at -1 asks for 5467008 B against a 1572864 B L1. Every entry was
# verified at pcc >= 0.99985 against torch per-frame GroupNorm.
MINIMAX_H3_GN_OUT_BLOCKS: dict[tuple[int, int, int, int], int] = {
    (256, 17, 128, 128): 4,
    (128, 17, 128, 128): 4,
    (128, 17, 256, 256): 16,
}


class MiniMaxH3FrameGroupNorm(GroupNorm3D):
    """Per-frame GroupNorm: statistics pool over ``(C_group, H, W)`` within one frame.

    Takes and returns ``(1, T, H, W, C)`` ROW_MAJOR. T is passed to ``GroupNorm3D`` as
    its batch axis, so no statistic ever crosses a frame boundary.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        num_frames: int,
        height: int,
        width: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__(
            num_channels=num_channels,
            num_groups=MINIMAX_H3_VAE_NUM_GROUPS,
            input_nhw=num_frames * height * width,
            num_batches=num_frames,
            eps=MINIMAX_H3_VAE_NORM_EPS,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_out_blocks = MINIMAX_H3_GN_OUT_BLOCKS.get((num_channels, num_frames, height, width), -1)

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        B, T, H, W, C = x_BTHWC.shape
        assert B == 1, f"one tile per device, got B={B}"
        assert (T, H, W) == (self.num_frames, self.height, self.width), (
            f"norm built for (T,H,W)=({self.num_frames},{self.height},{self.width}), "
            f"got ({T},{H},{W}) -- the grid is pinned at construction"
        )
        # (1,T,H,W,C) -> (T,1,H,W,C) puts T in the batch slot; row-major order is
        # unchanged, so this is free.
        x = ttnn.reshape(x_BTHWC, (T, 1, H, W, C))
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if x.get_dtype() != ttnn.bfloat16:
            # The group_norm kernel is bf16-only.
            x = ttnn.typecast(x, ttnn.bfloat16)

        tilized = ttnn.tilize_with_zero_padding(ttnn.reshape(x, (T, 1, H * W, C)), use_multicore=True)
        out = ttnn.group_norm(
            tilized,
            num_groups=self.num_groups,
            num_out_blocks=self.num_out_blocks,
            input_mask=self.mask.data,
            weight=self.weight.data,
            bias=self.bias.data,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=self.core_grid,
            inplace=False,
            use_welford=self.use_welford,
        )
        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(out, (B, T, H, W, C))


class MiniMaxH3DistributedFrameGroupNorm(Module):
    """Per-frame GroupNorm whose spatial extent is sharded across a mesh axis.

    Drop-in for :class:`MiniMaxH3FrameGroupNorm` -- same ``(1, T, H, W, C)`` in and out,
    with ``H`` the **local** height -- for the H/W-parallel encoder.

    Neither existing primitive can do this job:

    * ``ttnn.group_norm`` (what :class:`MiniMaxH3FrameGroupNorm` uses) has no notion of the
      mesh, so a sharded ``H`` gives each device a statistic over its own strip only.
    * ``ttnn.experimental.dit_fused_distributed_groupnorm`` hard-rejects ``N > 1``, and
      here ``N`` is ``T`` (17/9/5) because the statistic is per frame. It also takes a
      single ``cluster_axis``, so it cannot reduce over both mesh axes.

    So the statistics are computed directly: per ``(frame, group)`` local sums, an
    all-reduce of **only those** -- ``T x 32`` scalars, against a full-activation gather --
    then an elementwise normalise. Channel-to-group contraction is a matmul with a 0/1
    matrix, and its transpose spreads the result back over channels.

    Two passes, mean then *centred* variance, rather than one pass on ``E[x^2] - E[x]^2``:
    the group means here are not near zero, and that is the cancellation
    :class:`GroupNorm3D` uses Welford to avoid.

    Runs in the activation dtype, so unlike the local path it is **not** forced to bf16 --
    ``ttnn.group_norm`` is bf16-only, which STATE.md records as the encoder's precision
    floor. Here fp32 activations stay fp32.
    """

    def __init__(
        self,
        num_channels: int,
        *,
        num_frames: int,
        local_height: int,
        width: int,
        spatial_factor: int,
        cluster_axis: int,
        mesh_device: ttnn.MeshDevice,
        ccl_manager,
    ) -> None:
        super().__init__()
        assert num_channels % MINIMAX_H3_VAE_NUM_GROUPS == 0
        self.num_channels = num_channels
        self.num_groups = MINIMAX_H3_VAE_NUM_GROUPS
        self.eps = MINIMAX_H3_VAE_NORM_EPS
        self.num_frames = num_frames
        self.local_height = local_height
        self.width = width
        self.spatial_factor = spatial_factor
        self.cluster_axis = cluster_axis
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager

        # The divisor is the GLOBAL element count per (frame, group): the whole point is
        # that a device normalises by more elements than it holds.
        self.elements_per_group = (num_channels // self.num_groups) * local_height * spatial_factor * width

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
        # tt_dit exposes no all-reduce (not one call anywhere in the tree), so gather on the
        # singleton dim and sum locally. At T x 32 scalars the distinction is immaterial.
        gathered = self.ccl_manager.all_gather(group_sums, dim=1, mesh_axis=self.cluster_axis, use_hyperparams=False)
        return ttnn.sum(gathered, dim=1, keepdim=True)

    def _spread(self, per_group: ttnn.Tensor) -> ttnn.Tensor:
        """``(T,1,1,G)`` -> ``(T,1,1,C)``, each channel taking its own group's value."""
        return ttnn.matmul(per_group, self.from_groups.data)

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        B, T, H, W, C = x_BTHWC.shape
        assert B == 1, f"one unit per device, got B={B}"
        assert (T, H, W) == (
            self.num_frames,
            self.local_height,
            self.width,
        ), f"norm built for local (T,H,W)=({self.num_frames},{self.local_height},{self.width}), got ({T},{H},{W})"

        x = ttnn.reshape(x_BTHWC, (T, 1, H * W, C))
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        scale = 1.0 / self.elements_per_group
        mean = self._spread(ttnn.multiply(self._group_sums(x), scale))
        centred = ttnn.subtract(x, mean)
        variance = self._spread(ttnn.multiply(self._group_sums(ttnn.multiply(centred, centred)), scale))
        normed = ttnn.multiply(centred, ttnn.rsqrt(ttnn.add(variance, self.eps)))
        out = ttnn.add(ttnn.multiply(normed, self.weight.data), self.bias.data)

        out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        return ttnn.reshape(out, (B, T, H, W, C))


def _norm_silu(norm: MiniMaxH3FrameGroupNorm, x_BTHWC: ttnn.Tensor, dtype: ttnn.DataType = ttnn.float32) -> ttnn.Tensor:
    """``silu(norm(x))``, returned ROW_MAJOR in ``dtype``, ready for the next conv.

    The norm is bf16 whatever the surrounding precision -- ``ttnn.group_norm`` has no
    fp32 path -- so this is where the encoder's precision floor sits. Everything else
    stays fp32 to match the reference, which is why the cast back is explicit rather
    than letting bf16 propagate through the convolutions.
    """
    h = ttnn.silu(norm(x_BTHWC))
    if h.get_dtype() != dtype:
        h = ttnn.typecast(h, dtype)
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
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dtype = dtype
        conv_kwargs = dict(temporal_taps=temporal_taps, mesh_device=mesh_device, dtype=dtype)
        norm_kwargs = dict(num_frames=num_frames, height=height, width=width, mesh_device=mesh_device)
        # A resnet never changes T/H/W, so both norms share one shape.
        self.norm1 = MiniMaxH3FrameGroupNorm(in_channels, **norm_kwargs)
        self.conv1 = MiniMaxH3CausalConv3d(in_channels, out_channels, kernel_size=3, spatial_padding=1, **conv_kwargs)
        self.norm2 = MiniMaxH3FrameGroupNorm(out_channels, **norm_kwargs)
        self.conv2 = MiniMaxH3CausalConv3d(out_channels, out_channels, kernel_size=3, spatial_padding=1, **conv_kwargs)
        if in_channels != out_channels:
            self.conv_shortcut = MiniMaxH3CausalConv3d(
                in_channels, out_channels, kernel_size=1, spatial_padding=0, **conv_kwargs
            )

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        h = self.conv1(_norm_silu(self.norm1, x_BTHWC, self.dtype))
        h = self.conv2(_norm_silu(self.norm2, h, self.dtype))
        residual = self.conv_shortcut(x_BTHWC) if self.in_channels != self.out_channels else x_BTHWC
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
        dtype: ttnn.DataType = ttnn.float32,
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
            temporal_taps=temporal_taps,
            mesh_device=mesh_device,
            dtype=dtype,
        )

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        if self.spatial_stride == 2:
            x_BTHWC = reflect_pad_bottom_right(x_BTHWC)
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
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
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
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.temporal_taps = temporal_taps
        self.in_channels = in_channels
        self.dtype = dtype
        self.input_shape = (num_frames, height, width)

        conv_kwargs = dict(temporal_taps=temporal_taps, mesh_device=mesh_device, dtype=dtype)
        self.conv_in = MiniMaxH3CausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, spatial_padding=1, **conv_kwargs
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

        self.norm_out = MiniMaxH3FrameGroupNorm(
            block_out_channels[-1], num_frames=t, height=h, width=w, mesh_device=mesh_device
        )
        self.conv_out = MiniMaxH3CausalConv3d(
            block_out_channels[-1], out_channels, kernel_size=3, spatial_padding=1, **conv_kwargs
        )

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        h = self.conv_in(x_BTHWC)
        for down_block in self.down_blocks:
            h = down_block(h)
        return self.conv_out(_norm_silu(self.norm_out, h, self.dtype))


def pad_pixel_channels(x_BTHWC: torch.Tensor, aligned: int = 32) -> torch.Tensor:
    """Zero-pad the 3 pixel channels up to the tile-aligned count the conv expects."""
    channels = x_BTHWC.shape[-1]
    if channels >= aligned:
        return x_BTHWC
    return torch.nn.functional.pad(x_BTHWC, (0, aligned - channels))
