# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The convolution used throughout the MiniMax-H3 video VAE encoder.

Mirrors the reference ``MiniMaxH3VideoCausalConv3d``: spatial padding is **symmetric
reflect**, temporal padding is **causal zeros** of ``kernel_t - 1`` frames prepended
and nothing appended, and the convolution itself carries no padding.

Two things make this simpler than the WAN/LTX conv it is modelled on:

* It runs **unsharded** -- one 256x256 tile lives entirely on one device, because the
  reference computes every spatial tile independently and only cross-fades the
  overlaps at the end. So reflect padding is a local slice-and-concat rather than a
  halo exchange, which matters because ``neighbor_pad_async`` has no ``reflect`` mode.
* T is never sharded either, so the causal front-pad is a local concat of zeros.

``temporal_taps=1`` selects the keyframe fast path. That is **exact**, not an
approximation: the causal front-pad is zeros, so a 3-tap temporal conv on a lone
frame sees ``[0, 0, x]`` and every tap but the last is multiplied by zero. The weight
is sliced to ``weight[:, :, -1:]`` at load time.
"""

from __future__ import annotations

from typing import Sequence

import torch
from loguru import logger

import ttnn

from ....layers.module import Module, Parameter
from ....utils.conv3d import _FP32_BLOCKINGS, _ntuple, aligned_channels, get_conv3d_config, register_conv3d_configs

# Every conv shape in this encoder misses the fp32 blocking table and falls back to
# (32, 32, 1, 1, 1). As the LTX audio entries in conv3d.py note, an ``H_out=W_out=1``
# blocking "forces one output pixel per work-unit". Measured on the shipping tile, that
# fallback is slow but not fatal -- conv_in is 9 ms at (1,256,256) and 305 ms at
# (17,256,256) -- so it is a performance problem, not a correctness blocker.
#
# These are **stubs**, not swept values, shaped after the WAN 720p *encoder* entries in
# conv3d.py: keep ``H_out_block * W_out_block`` near 32 (they use 2x16, 16x2, 8x4) and let
# ``C_in_block`` follow the input width. `bruteforce_conv3d_sweep.py` is the tool for real
# tuning and that belongs in the performance pass, after all four VAE halves are correct.
#
# The ViT decoder needs no entry here: it is a pure transformer, so its only conv --
# ``post_quant_conv`` (24->24, 1x1x1) -- is folded into ``proj_in`` and everything else
# goes through ``Linear``'s matmul config.
# Constraint the kernel enforces: C_out_block must be a multiple of 32 *and* divide the
# padded output channel count evenly -- so 96 is legal against 384 output channels (as in
# the WAN entries these are shaped after) but not against 128.
_H3_ENCODER_BLOCKINGS = {
    (32, 128): (32, 32, 1, 2, 16),  # conv_in, C_in padded 3 -> 32
    (128, 128): (128, 32, 1, 2, 16),
    (128, 256): (128, 32, 1, 2, 16),
    (256, 256): (128, 32, 1, 4, 8),
    # C_in_block=256 with a 32-pixel H/W block overflows L1 in fp32: measured 2786176 B
    # against 1572864 B, i.e. 1.77x over, so these stay at 128.
    (256, 512): (128, 32, 1, 4, 8),
    (512, 512): (128, 32, 1, 4, 8),
    (512, 1024): (128, 32, 1, 4, 8),
    (1024, 1024): (128, 32, 1, 4, 8),
    (1024, 48): (128, 32, 1, 8, 4),  # conv_out, with quant_conv folded in
}
_H3_BLOCKING_ENTRIES = {
    (in_c, out_c, kernel): blocking
    for (in_c, out_c), blocking in _H3_ENCODER_BLOCKINGS.items()
    # The k1 shortcuts keep the fallback: a 1x1x1 conv is a matmul and cheap either way.
    for kernel in ((3, 3, 3), (1, 3, 3))
}
register_conv3d_configs(_H3_BLOCKING_ENTRIES)
# register_conv3d_configs only updates the bf16 fallback table, but get_conv3d_config
# short-circuits to _FP32_BLOCKINGS when the weights are fp32 -- so for an fp32 encoder the
# registration above is silently ignored. Seed the fp32 table as well. `setdefault`, so a
# swept value that lands in conv3d.py later wins over these stubs.
for _key, _blocking in _H3_BLOCKING_ENTRIES.items():
    _FP32_BLOCKINGS.setdefault(_key, _blocking)


class MiniMaxH3CausalConv3d(Module):
    """``ttnn.experimental.conv3d`` with H3's reflect-spatial / causal-temporal padding.

    Args:
        spatial_padding: symmetric reflect pad applied to H and W before the conv.
            The reference passes 1 for every 3x3 conv and **0** for the strided
            downsample convs, which instead get an asymmetric bottom/right pad from
            :class:`~.encoder_minimax_h3.MiniMaxH3Downsample3d`.
        temporal_taps: 3 for the general causal conv, 1 for the T=1 keyframe path.
            A ``kernel_size`` of 1 is temporally trivial either way.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 1,
        spatial_padding: int = 1,
        temporal_taps: int = 3,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()

        if temporal_taps not in (1, 3):
            raise ValueError(f"temporal_taps must be 1 or 3, got {temporal_taps}")

        kernel = list(_ntuple(kernel_size, 3))
        # A k1 conv (nin_shortcut / quant_conv) has no temporal extent to collapse.
        self.collapse_temporal = temporal_taps == 1 and kernel[0] > 1
        if self.collapse_temporal:
            kernel[0] = 1

        self.unpadded_in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = max(32, out_channels)
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size = tuple(kernel)
        self.stride = tuple(_ntuple(stride, 3))
        self.spatial_padding = spatial_padding
        self.mesh_device = mesh_device
        self.dtype = dtype

        # Causal front-pad, applied locally in forward. Zero once collapsed.
        self.time_pad = 0 if self.collapse_temporal else self.kernel_size[0] - 1

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
        )

        from models.common.utility_functions import is_blackhole

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if (is_blackhole() and dtype == ttnn.float32)
            else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Slice the temporal tap if collapsed, pad both channel axes, then prepare.

        Padding **input** channels is the step ``Conv2dViaConv3d`` omits, which makes
        it unable to load a 3-channel ``conv_in`` at all: it sizes the weight from the
        aligned in-channel count but prepares an unpadded weight.
        """
        if "weight" not in state:
            return

        weight = state["weight"]
        bias = state.get("bias")

        if self.collapse_temporal:
            # Only the final temporal tap survives a single frame.
            weight = weight[:, :, -1:].contiguous()

        # torch conv3d weight is (out, in, kt, kh, kw); dim 1 is the in-channel axis.
        if self.in_channels != self.unpadded_in_channels:
            weight = torch.nn.functional.pad(
                weight, (0, 0, 0, 0, 0, 0, 0, self.in_channels - self.unpadded_in_channels)
            )
        if self.out_channels != self.unpadded_out_channels:
            weight = torch.nn.functional.pad(
                weight, (0, 0, 0, 0, 0, 0, 0, 0, 0, self.out_channels - self.unpadded_out_channels)
            )
            if bias is not None:
                bias = torch.nn.functional.pad(bias, (0, self.out_channels - self.unpadded_out_channels))

        weight_tt = ttnn.from_torch(weight, dtype=self.dtype, pad_value=0)
        prepared = ttnn.experimental.prepare_conv3d_weights(
            weight_tensor=weight_tt, C_in_block=self.conv_config.C_in_block, device=self.mesh_device
        )
        state["weight"] = ttnn.to_torch(ttnn.get_device_tensors(prepared)[0])
        if bias is not None:
            state["bias"] = bias.reshape(1, -1)

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        """``x_BTHWC``: ``(B, T, H, W, C)`` ROW_MAJOR, C already tile-aligned."""
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT, f"conv3d needs ROW_MAJOR, got {x_BTHWC.layout}"

        if self.spatial_padding > 0:
            x_BTHWC = reflect_pad_hw(x_BTHWC, self.spatial_padding, self.spatial_padding)
        if self.time_pad > 0:
            x_BTHWC = causal_pad_t(x_BTHWC, self.time_pad, self.mesh_device)

        out = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0, 0),
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        return out


def causal_pad_t(x_BTHWC: ttnn.Tensor, pad: int, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Prepend ``pad`` zero frames on T. Nothing is appended -- that is the causality."""
    B, _, H, W, C = x_BTHWC.shape
    zeros = ttnn.zeros((B, pad, H, W, C), dtype=x_BTHWC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    return ttnn.concat([zeros, x_BTHWC], dim=1)


def reflect_edge_correction(
    padded_BTHWC: ttnn.Tensor,
    dim: int,
    pad: int,
    *,
    edge_masks: tuple[ttnn.Tensor, ttnn.Tensor] | None,
) -> ttnn.Tensor:
    """Turn a ``replicate`` halo pad into a ``reflect`` one at the **global** edges only.

    ``neighbor_pad_async`` offers ``zeros`` and ``replicate`` but not ``reflect``, and H3 pads
    reflect. That gap is narrower than it looks: at an interior shard boundary the halo *is*
    the neighbour's real data, so the padding mode never enters and ``replicate`` is already
    exact. The two differ only in the outermost pixel at the two global image edges -- after
    a replicate pad of 1 the layout is ``[x0, x0, x1, ...]`` where reflect wants
    ``[x1, x0, x1, ...]``.

    So the fix is local and data-driven. With a pad of ``p``, ``padded[p]`` is ``x[0]`` and
    ``padded[p + k]`` is ``x[k]``, so the reflect value for ``padded[p - 1 - j]`` is
    ``padded[p + 1 + j]``. Blending with a per-device 0/1 mask keeps every device running the
    **same ops** -- the mask is sharded data, not a branch -- which is what keeps the program
    SPMD-uniform. ``edge_masks`` is ``(leading, trailing)``; ``None`` means unsharded, where
    every device owns both global edges.

    A one-pixel border error passes PCC and reads as a faint vignette, so this needs its own
    gate rather than riding on a whole-encoder PCC number.
    """
    if pad <= 0:
        return padded_BTHWC
    size = padded_BTHWC.shape[dim]
    for j in range(pad):
        for leading in (True, False):
            target = pad - 1 - j if leading else size - pad + j
            source = pad + 1 + j if leading else size - pad - 2 - j
            mask = None if edge_masks is None else edge_masks[0 if leading else 1]
            target_slice = slice_dim(padded_BTHWC, dim, target, target + 1)
            source_slice = slice_dim(padded_BTHWC, dim, source, source + 1)
            if mask is None:
                corrected = source_slice
            else:
                # corrected = target + mask * (source - target); mask is 1 only on the device
                # that owns this global edge, so interior devices are untouched.
                corrected = ttnn.add(target_slice, ttnn.mul(ttnn.sub(source_slice, target_slice), mask))
            before = slice_dim(padded_BTHWC, dim, 0, target) if target > 0 else None
            after = slice_dim(padded_BTHWC, dim, target + 1, size) if target + 1 < size else None
            parts = [p for p in (before, corrected, after) if p is not None]
            padded_BTHWC = ttnn.concat(parts, dim=dim)
    return padded_BTHWC


def edge_mask_pair(
    mesh_device: ttnn.MeshDevice, factor: int, mesh_axis: int, dtype: ttnn.DataType
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Per-device 0/1 scalars marking which shard owns the leading / trailing global edge.

    Sharded *data*, so :func:`reflect_edge_correction` stays branch-free and every device
    issues an identical program.
    """
    leading = torch.zeros(factor, 1, 1, 1, 1)
    trailing = torch.zeros(factor, 1, 1, 1, 1)
    leading[0] = 1.0
    trailing[-1] = 1.0
    return tuple(
        ttnn.from_torch(
            mask,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
        )
        for mask in (leading, trailing)
    )


def reflect_pad_hw(x_BTHWC: ttnn.Tensor, pad_h: int, pad_w: int) -> ttnn.Tensor:
    """Symmetric reflect pad on H (dim 2) and W (dim 3).

    Reflect excludes the edge itself, so a pad of 1 takes row 1 rather than row 0 --
    which is exactly where it differs from ``replicate``, and the reason this is done
    locally rather than through ``neighbor_pad_async``.
    """
    for pad, dim in ((pad_h, 2), (pad_w, 3)):
        if pad <= 0:
            continue
        size = x_BTHWC.shape[dim]
        if size < pad + 1:
            raise ValueError(f"cannot reflect-pad dim {dim} of size {size} by {pad}")
        leading = [slice_dim(x_BTHWC, dim, i, i + 1) for i in range(pad, 0, -1)]
        trailing = [slice_dim(x_BTHWC, dim, size - 1 - i, size - i) for i in range(1, pad + 1)]
        x_BTHWC = ttnn.concat([*leading, x_BTHWC, *trailing], dim=dim)
    return x_BTHWC


def reflect_pad_bottom_right(x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
    """The downsample's asymmetric ``F.pad(..., (0,1,0,1), mode="reflect")``.

    One extra row at the bottom and one column at the right, reflected -- so the
    stride-2 conv that follows produces exactly ``ceil(size / 2)``.
    """
    height, width = x_BTHWC.shape[2], x_BTHWC.shape[3]
    x_BTHWC = ttnn.concat([x_BTHWC, slice_dim(x_BTHWC, 2, height - 2, height - 1)], dim=2)
    return ttnn.concat([x_BTHWC, slice_dim(x_BTHWC, 3, width - 2, width - 1)], dim=3)


def slice_dim(x: ttnn.Tensor, dim: int, start: int, stop: int) -> ttnn.Tensor:
    starts = [0] * len(x.shape)
    stops = list(x.shape)
    starts[dim], stops[dim] = start, stop
    return ttnn.slice(x, starts, stops)
