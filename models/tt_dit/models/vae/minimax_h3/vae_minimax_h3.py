# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 visual VAE encoder for the FL2VA keyframe path.

FL2VA conditioning only ever encodes **single frames**, and that collapses the
whole causal 3D encoder to a 2D one. ``BaseConv3d`` front-pads the temporal axis
with zeros, so a 3-tap conv on a lone frame sees ``[0, 0, x]`` and reduces to
``weight[:, :, -1] * x`` -- every temporal tap but the last is multiplied by zero.
Measured against the reference: rel err 1.5e-07 per conv, bit-exact for k1, and
1.1e-07 after chaining twelve, so it is fp32 accumulation order and does not
compound. There is therefore no temporal halo and no cache to carry here.

The T > 1 path, which Ref2VA video references would need, is a separate extension:
it keeps these weights but restores the temporal taps and the causal halo.

Built on :class:`Conv2dViaConv3d` so the convolutions stay on the same
``ttnn.experimental.conv3d`` path the WAN and LTX VAEs use, which is what makes
that extension additive rather than a rewrite.

The encoder runs **unsharded**. It is one frame, once per request -- about 3.7
TFLOP against a 49-step denoise -- so H/W sharding would buy nothing and would drag
in a halo exchange that cannot express H3's reflect padding
(``neighbor_pad_async`` offers only zeros and replicate). Unsharded, reflect is a
local slice-and-concat. The 36-layer ViT decoder is where sharding earns its keep.

Everything is fp32: the whole VAE checkpoint is, and the keyframe encode contract
depends on it.
"""

from __future__ import annotations

import torch

import ttnn

from ....layers.audio_ops import Conv2dViaConv3d
from ....layers.module import Module, ModuleList
from ....layers.normalization import GroupNorm

# GroupNorm settings are fixed by the checkpoint, not configurable.
MINIMAX_H3_VAE_NUM_GROUPS = 32
MINIMAX_H3_VAE_NORM_EPS = 1e-6


class MiniMaxH3VaeConv(Conv2dViaConv3d):
    """A keyframe-path conv: the last temporal tap of an H3 ``BaseConv3d``.

    Adds ``reflect`` to the padding modes and slices the 5D checkpoint weight down
    to the 4D one the base class expects.
    """

    def __init__(self, in_channels, out_channels, *, kernel_size, stride=1, reflect=True, **kwargs):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding_mode="zeros",
            **kwargs,
        )
        # Reflect is applied here and the conv itself pads by nothing, so the base
        # class's symmetric zero padding must be switched off.
        self.reflect = reflect and self.pad_h > 0
        if self.reflect:
            self.internal_padding = (0, 0, 0)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.get("weight")
        if weight is not None and weight.dim() == 5:
            # Only the final temporal tap survives a single frame.
            state["weight"] = weight[:, :, -1].contiguous()
        super()._prepare_torch_state(state)

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        if self.reflect:
            x_BHWC = _reflect_pad_hw(x_BHWC, self.internal_pad_h, self.internal_pad_w)
        return super().forward(x_BHWC)

    @property
    def internal_pad_h(self) -> int:
        return self.kernel_size[1] // 2

    @property
    def internal_pad_w(self) -> int:
        return self.kernel_size[2] // 2


def _reflect_pad_hw(x_BHWC: ttnn.Tensor, pad_h: int, pad_w: int) -> ttnn.Tensor:
    """Reflect-pad ``(B, H, W, C)`` by ``pad_h`` / ``pad_w``.

    Reflect excludes the edge itself, so a pad of 1 takes row 1 rather than row 0 --
    which is exactly where it differs from replicate, and the reason this is not
    left to the halo op.
    """
    for pad, dim in ((pad_h, 1), (pad_w, 2)):
        if pad <= 0:
            continue
        size = x_BHWC.shape[dim]
        if size < pad + 1:
            raise ValueError(f"cannot reflect-pad dim {dim} of size {size} by {pad}")
        leading = [_slice_dim(x_BHWC, dim, index, index + 1) for index in range(pad, 0, -1)]
        trailing = [_slice_dim(x_BHWC, dim, size - 1 - index, size - index) for index in range(1, pad + 1)]
        x_BHWC = ttnn.concat([*leading, x_BHWC, *trailing], dim=dim)
    return x_BHWC


def _slice_dim(x: ttnn.Tensor, dim: int, start: int, stop: int) -> ttnn.Tensor:
    starts = [0] * len(x.shape)
    stops = list(x.shape)
    starts[dim], stops[dim] = start, stop
    return ttnn.slice(x, starts, stops)


class MiniMaxH3VaeResnetBlock(Module):
    """``norm -> silu -> conv -> norm -> silu -> conv``, plus a k1 shortcut when widths differ."""

    def __init__(self, in_channels: int, out_channels: int, *, mesh_device, dtype=ttnn.float32) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_kwargs = dict(
            num_groups=MINIMAX_H3_VAE_NUM_GROUPS,
            mesh_device=mesh_device,
        )
        self.norm1 = GroupNorm(num_channels=in_channels, **norm_kwargs)
        self.norm2 = GroupNorm(num_channels=out_channels, **norm_kwargs)
        conv_kwargs = dict(kernel_size=3, mesh_device=mesh_device, dtype=dtype)
        self.conv1 = MiniMaxH3VaeConv(in_channels, out_channels, **conv_kwargs)
        self.conv2 = MiniMaxH3VaeConv(out_channels, out_channels, **conv_kwargs)
        if in_channels != out_channels:
            self.nin_shortcut = MiniMaxH3VaeConv(
                in_channels, out_channels, kernel_size=1, reflect=False, mesh_device=mesh_device, dtype=dtype
            )

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        h = self.conv1(ttnn.silu(self.norm1(x_BHWC)))
        h = self.conv2(ttnn.silu(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x_BHWC = self.nin_shortcut(x_BHWC)
        return ttnn.add(h, x_BHWC)


class MiniMaxH3VaeDownsample(Module):
    """Stride-2 spatial conv with H3's asymmetric pre-pad.

    The reference pads only the right and bottom by one before a stride-2 conv whose
    own spatial padding is zero, so the pad is not centred.
    """

    def __init__(self, in_channels: int, out_channels: int, *, mesh_device, dtype=ttnn.float32) -> None:
        super().__init__()
        self.conv = MiniMaxH3VaeConv(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            reflect=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.conv.internal_padding = (0, 0, 0)

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        x_BHWC = _reflect_pad_asymmetric(x_BHWC)
        return self.conv(x_BHWC)


def _reflect_pad_asymmetric(x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
    """Pad right and bottom by one, reflecting -- the reference's ``(0,1,0,1)``."""
    height, width = x_BHWC.shape[1], x_BHWC.shape[2]
    x_BHWC = ttnn.concat([x_BHWC, _slice_dim(x_BHWC, 1, height - 2, height - 1)], dim=1)
    return ttnn.concat([x_BHWC, _slice_dim(x_BHWC, 2, width - 2, width - 1)], dim=2)


class MiniMaxH3VaeEncoder(Module):
    """The FL2VA keyframe encoder: pixels to ``2 * z_channels`` moments.

    ``forward`` returns the ``[mean, logvar]`` moments. Sampling stays on host so
    the seed-42 posterior draw the released model conditions on is reproducible
    bit-for-bit; see ``pipelines/minimax_h3/conditioning.py``.
    """

    def __init__(
        self,
        *,
        ch: int = 128,
        ch_mult: tuple[int, ...] = (1, 2, 2, 4, 4, 8),
        num_res_blocks: int = 2,
        space_down: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        time_down: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        in_channels: int = 3,
        z_channels: int = 24,
        double_z: bool = True,
        mesh_device=None,
        dtype=ttnn.float32,
    ) -> None:
        super().__init__()
        num_levels = len(ch_mult)
        block_mid = [ch * mult for mult in ch_mult]
        block_in = [block_mid[0], *block_mid[:-1]]
        self.num_levels = num_levels
        self.num_res_blocks = num_res_blocks
        self.z_channels = z_channels

        conv_kwargs = dict(mesh_device=mesh_device, dtype=dtype)
        self.conv_in = MiniMaxH3VaeConv(in_channels, block_in[0], kernel_size=3, **conv_kwargs)

        blocks, downsamples = [], []
        for level in range(num_levels):
            for index in range(num_res_blocks):
                blocks.append(
                    MiniMaxH3VaeResnetBlock(
                        block_in[level] if index == 0 else block_mid[level],
                        block_mid[level],
                        **conv_kwargs,
                    )
                )
            # A level downsamples only when it actually reduces a dimension. With a
            # single frame `time_down` cannot bite -- stride 2 over a 3-deep
            # zero-padded axis still selects the one real frame -- but it still
            # decides whether the level owns a conv at all.
            if space_down[level] * time_down[level] > 1:
                downsamples.append((level, MiniMaxH3VaeDownsample(block_mid[level], block_mid[level], **conv_kwargs)))
        self.blocks = ModuleList(blocks)
        self.downsample_levels = [level for level, _ in downsamples]
        self.downsamples = ModuleList([module for _, module in downsamples])

        self.norm_out = GroupNorm(
            num_channels=block_mid[-1], num_groups=MINIMAX_H3_VAE_NUM_GROUPS, mesh_device=mesh_device
        )
        moments = 2 * z_channels if double_z else z_channels
        self.conv_out = MiniMaxH3VaeConv(block_mid[-1], moments, kernel_size=3, **conv_kwargs)
        self.quant_conv = MiniMaxH3VaeConv(moments, moments, kernel_size=1, reflect=False, **conv_kwargs)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Flatten the checkpoint's per-level nesting onto flat block lists.

        The checkpoint nests as ``encoder.down.<level>.block.<i>.*`` and
        ``encoder.down.<level>.downsample.conv.*``; ``quant_conv`` sits outside
        ``encoder.`` entirely.
        """
        for key in list(state):
            if key.startswith("encoder."):
                state[key.removeprefix("encoder.")] = state.pop(key)

        for key in list(state):
            if not key.startswith("down."):
                continue
            _, level, kind, rest = key.split(".", 3)
            level = int(level)
            if kind == "block":
                index, tail = rest.split(".", 1)
                flat = level * self.num_res_blocks + int(index)
                state[f"blocks.{flat}.{tail}"] = state.pop(key)
            elif kind == "downsample":
                flat = self.downsample_levels.index(level)
                state[f"downsamples.{flat}.{rest}"] = state.pop(key)

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        h = self.conv_in(x_BHWC)
        downsample = 0
        for level in range(self.num_levels):
            for index in range(self.num_res_blocks):
                h = self.blocks[level * self.num_res_blocks + index](h)
            if level in self.downsample_levels:
                h = self.downsamples[downsample](h)
                downsample += 1
        h = self.conv_out(ttnn.silu(self.norm_out(h)))
        return self.quant_conv(h)
