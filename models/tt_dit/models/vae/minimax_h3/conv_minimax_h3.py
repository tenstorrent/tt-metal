# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The convolution used throughout the MiniMax-H3 video VAE encoder.

Mirrors the reference ``MiniMaxH3VideoCausalConv3d``: spatial padding is **symmetric
reflect**, temporal padding is **causal zeros** of ``kernel_t - 1`` frames prepended
and nothing appended, and the convolution itself carries no padding.

Runs either unsharded (one 256x256 tile per device, which is what the reference's own
tiling makes natural) or H/W-sharded via ``VaeHWParallelConfig``. The spatial pad follows
the axis: a **sharded** axis gets it from a halo exchange, a **replicated** one keeps it as
a local slice-and-concat, and never both. T is never sharded, so the causal front-pad is
always a local concat of zeros.

The halo pads ``replicate`` because ``neighbor_pad_async`` has no ``reflect`` mode. That is
exact at every interior boundary -- the halo there is the neighbour's real data, so the mode
never enters -- and :func:`reflect_edge_correction` repairs the two global edges per axis.

``temporal_taps=1`` selects the keyframe fast path. That is **exact**, not an
approximation: the causal front-pad is zeros, so a 3-tap temporal conv on a lone
frame sees ``[0, 0, x]`` and every tap but the last is multiplied by zero. The weight
is sliced to ``weight[:, :, -1:]`` at load time.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from loguru import logger

import ttnn

from ....layers.module import Module, Parameter
from ....utils.conv3d import _FP32_BLOCKINGS, _ntuple, aligned_channels, get_conv3d_config, register_conv3d_configs
from ....utils.tensor import local_device_to_torch

# Every conv shape in this encoder misses the fp32 blocking table and falls back to
# (32, 32, 1, 1, 1). As the LTX audio entries in conv3d.py note, an ``H_out=W_out=1``
# blocking "forces one output pixel per work-unit". Measured on the shipping tile, that
# fallback is slow but not fatal -- conv_in is 9 ms at (1,256,256) and 305 ms at
# (17,256,256) -- so it is a performance problem, not a correctness blocker.
#
# **Swept**, not stubs: measured per shape with `wan2_2/bruteforce_conv3d_sweep.py`, which
# brute-forces every legal blocking and times it on hardware under a trace. Against the
# conv3d.py table baseline the winners are 2.5x-25.6x per layer and 9.5x summed; on the
# fallback the encoder runs at ~2.3 TFLOP/s against the ViT decoder's 14.0 purely because
# every one of its shapes misses the table.
#
# Keyed by (C_in, C_out), which a level's res and downsample convs share. Where they share a
# key the blocking must be legal for **both**: a strided downsample has a different input
# footprint than the res conv at the same channel pair, so one swept on the res conv alone
# can overflow L1 on the downsample (measured 1753984 B against a 1572864 B L1). Each entry
# is therefore the blocking minimising *total* time over every layer that shares the key,
# restricted to those every one of them measured OK.
#
# Constraint the kernel enforces: C_out_block must be a multiple of 32 *and* divide the
# **tile-aligned** output channel count -- so conv_out's 48 out-channels align to 64, which
# is why a C_out_block of 64 is legal there and 32 would not divide 48.

_H3_ENCODER_BLOCKINGS = {
    (32, 128): (32, 128, 3, 2, 16),  # conv_in: 5358 us
    (128, 128): (64, 128, 1, 16, 2),  # b0_res+b0_down: 32731 us
    (128, 256): (32, 256, 3, 16, 2),  # b1_res0: 5533 us
    (256, 256): (64, 128, 1, 16, 2),  # b1_res1+b1_down+b2_res+b2_down: 29478 us
    (256, 512): (64, 128, 1, 16, 2),  # b3_res0: 710 us
    (512, 512): (64, 128, 1, 16, 2),  # b3_res1+b3_down+b4_res: 2983 us
    (512, 1024): (64, 128, 1, 8, 4),  # b5_res0: 851 us
    (1024, 1024): (64, 64, 5, 16, 2),  # b5_res1: 2025 us
    (1024, 48): (128, 32, 1, 16, 2),  # conv_out: 197 us
}
_H3_BLOCKING_ENTRIES = {
    (in_c, out_c, kernel): blocking
    for (in_c, out_c), blocking in _H3_ENCODER_BLOCKINGS.items()
    # kernel (1, 1, 1) is deliberately absent here; the k1 shortcuts get their own entries below.
    for kernel in ((3, 3, 3), (1, 3, 3))
}

register_conv3d_configs(_H3_BLOCKING_ENTRIES)
# register_conv3d_configs only updates the bf16 fallback table, but get_conv3d_config
# short-circuits to _FP32_BLOCKINGS when the weights are fp32 -- so for an fp32 encoder the
# registration above is silently ignored. Seed the fp32 table as well. `setdefault`, so a
# swept value that lands in conv3d.py itself wins over these entries.
for _key, _blocking in _H3_BLOCKING_ENTRIES.items():
    _FP32_BLOCKINGS.setdefault(_key, _blocking)

# k1 shortcut convs, **bf16 only**. These were originally left on the fallback on the
# assumption that a 1x1x1 conv is a matmul and cheap either way -- false at block1's
# resolution, where the (32, 32, 1, 1, 1) fallback forces one output pixel of M per work
# unit: 182 ms measured under trace against ~0.95 ms here, and 36 ms inside the eager
# encoder. Swept with T_out_block ranging (the stock combo builder pins T to 1 for kT = 1
# kernels, which is how the shape was missed), then **correctness-checked against torch
# conv3d** -- which the sweep does not do, and which matters: the raw sweep winner
# (128, 256, 17, 2, 8) is marginally faster but SILENTLY WRONG (PCC 0.4% at T = 17, NaN at
# T = 1; its 17 * 2 * 8 = 272-patch M block is not tile-aligned). This entry is the fastest
# blocking that scores PCC 99.999% at both shapes the encoder runs it at, T = 17 (clip,
# 0.95 ms) and T = 1 (keyframe, 0.32 ms vs the fallback's 10.4 ms): M = 16 * 1 * 32 = 512
# patches, exactly 16 tiles, whole 128-channel K in one block.
#
# Deliberately NOT seeded into _FP32_BLOCKINGS: the k1 sweep and the correctness check ran
# bf16 alone, and a k1 sweep blocking at fp32 (doubled circular buffers) hung the device --
# fp32 keeps the default it always had. The b3/b5 shortcuts also keep the default
# everywhere: at 32^2 and 16^2 spatial they measure 1-2 ms and are not worth a table row.
register_conv3d_configs({(128, 256, (1, 1, 1)): (128, 256, 16, 1, 32)})  # b1_res0 conv_shortcut


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
        trailing_spatial_padding: int = 0,
        temporal_taps: int = 3,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config=None,
        ccl_manager=None,
        pixel_norm: tuple[Sequence[float], Sequence[float]] | None = None,
    ) -> None:
        super().__init__()

        if temporal_taps not in (1, 3):
            raise ValueError(f"temporal_taps must be 1 or 3, got {temporal_taps}")

        # H/W sharding: the spatial pad becomes a halo exchange on the sharded axes and stays
        # a local slice-and-concat on the replicated ones. Same external/internal split as
        # LTXCausalConv3d, which is the clearer of the two in-tree versions -- it zeroes
        # `internal` where sharded and `external` where not, so neither is ambiguous.
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.height_factor = 1 if parallel_config is None else parallel_config.height_parallel.factor
        self.width_factor = 1 if parallel_config is None else parallel_config.width_parallel.factor
        self.is_sharded = self.height_factor > 1 or self.width_factor > 1
        if self.is_sharded and ccl_manager is None:
            raise ValueError("a sharded parallel_config needs a ccl_manager for the halo exchange")

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

        # Causal front-pad, applied locally in forward. Zero once collapsed. T is never
        # sharded, so this stays a local concat of zeros.
        self.time_pad = 0 if self.collapse_temporal else self.kernel_size[0] - 1

        # ``pixel_norm`` folds `(x/255 - mean)/std` into this conv (see _prepare_torch_state),
        # so the input is the decoder's raw uint8 pixels as floats. The causal front-pad must
        # then carry the raw value that normalizes to ZERO -- `255 * mean` per channel -- or
        # the first output frame of every clip diverges from the reference's zero-padded
        # normalized input. taps=1 collapses the pad away, so keyframes never hit this.
        self.pixel_norm = pixel_norm
        self.causal_pad_values: tuple[float, ...] | None = None
        if pixel_norm is not None:
            mean, std = pixel_norm
            assert (
                len(mean) == len(std) == in_channels
            ), f"pixel_norm has {len(mean)} channels for a {in_channels}-channel conv"
            if self.time_pad > 0:
                self.causal_pad_values = tuple(255.0 * m for m in mean)

        # Padding is asymmetric in general: H3's downsamplers pre-pad ``(0,1,0,1)`` reflect
        # (one extra row at the bottom, one column at the right) so a stride-2 conv lands on
        # exactly ceil(size/2). Carrying that here rather than in the model means one code
        # path covers it sharded and unsharded -- and under sharding it *has* to be here,
        # because only the device owning the global bottom/right edge may add that row, and
        # the interior devices need a real halo row from their neighbour instead.
        self.trailing_spatial_padding = trailing_spatial_padding
        pad_before = spatial_padding
        pad_after = spatial_padding + trailing_spatial_padding

        # A sharded axis gets its pad from the halo (external); a replicated one keeps it
        # local (internal). Never both, or the pad is applied twice.
        self.external_pad_h = (pad_before, pad_after) if self.height_factor > 1 else (0, 0)
        self.external_pad_w = (pad_before, pad_after) if self.width_factor > 1 else (0, 0)
        self.local_pad_h = (0, 0) if self.height_factor > 1 else (pad_before, pad_after)
        self.local_pad_w = (0, 0) if self.width_factor > 1 else (pad_before, pad_after)

        self._edge_masks: dict[int, tuple] = {}
        if self.is_sharded and pad_after > 0:
            if self.height_factor > 1:
                self._edge_masks[2] = edge_mask_pair(
                    mesh_device, self.height_factor, parallel_config.height_parallel.mesh_axis, dtype
                )
            if self.width_factor > 1:
                self._edge_masks[3] = edge_mask_pair(
                    mesh_device, self.width_factor, parallel_config.width_parallel.mesh_axis, dtype
                )

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=self.height_factor,
            w_factor=self.width_factor,
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
        # Persistent zero block for the causal front-pad; see causal_pad_t.
        self._causal_zeros: dict = {}

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

        if self.pixel_norm is not None:
            # Fold `(x/255 - mean)/std` into the weight and bias, so raw uint8 pixels (as
            # floats) enter this conv directly: `conv((ax + d)) = conv_scaled(x) + bias_shift`
            # with per-channel `a = 1/(255 std)`, `d = -mean/std`. AFTER the collapse slice,
            # because the bias shift sums `W * d` over exactly the taps this conv applies --
            # a taps=1 keyframe conv must not carry the two sliced-away taps' shift. The
            # taps=3 causal front-pad is handled by `causal_pad_values` above.
            mean, std = self.pixel_norm
            scale = torch.tensor([1.0 / (255.0 * s) for s in std], dtype=weight.dtype).view(1, -1, 1, 1, 1)
            shift = torch.tensor([-m / s for m, s in zip(mean, std)], dtype=weight.dtype).view(1, -1, 1, 1, 1)
            if bias is None:
                bias = torch.zeros(weight.shape[0], dtype=weight.dtype)
            bias = bias + (weight * shift).sum(dim=(1, 2, 3, 4))
            weight = weight * scale

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
        state["weight"] = local_device_to_torch(prepared)
        if bias is not None:
            state["bias"] = bias.reshape(1, -1)

    def _halo_pad(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        """Exchange the spatial halo on the sharded axes, then fix the global edges.

        ``neighbor_pad_async`` has no ``reflect`` mode, so this pads ``replicate`` -- exact at
        every interior boundary, where the halo is the neighbour's real data -- and then
        :func:`reflect_edge_correction` repairs the two global edges per axis. Both axes go in
        one fused call when both are sharded, following ``WanCausalConv3d``.
        """
        dims, pad_left, pad_right, axes, semaphores, links = [], [], [], [], [], []
        for dim, (before, after), factor, factor_config in (
            (2, self.external_pad_h, self.height_factor, "height_parallel"),
            (3, self.external_pad_w, self.width_factor, "width_parallel"),
        ):
            if (before <= 0 and after <= 0) or factor <= 1:
                continue
            mesh_axis = getattr(self.parallel_config, factor_config).mesh_axis
            dims.append(dim)
            pad_left.append(before)
            pad_right.append(after)
            axes.append(mesh_axis)
            semaphores.append(self.ccl_manager.get_np_ping_pong_semaphore(mesh_axis))
            # list(): ttnn.Shape supports integer indexing only, not slicing.
            links.append(max(1, min(math.prod(list(x_BTHWC.shape)[:dim]), self.ccl_manager.num_links)))

        if not dims:
            return x_BTHWC

        x_BTHWC = self.ccl_manager.neighbor_pad_persistent_buffer(
            x_BTHWC,
            dims=dims,
            pad_left=pad_left,
            pad_right=pad_right,
            padding_mode="replicate",
            axes=axes,
            neighbor_sems=semaphores,
            num_links=links,
        )
        for dim, before, after in zip(dims, pad_left, pad_right):
            x_BTHWC = reflect_edge_correction(x_BTHWC, dim, before, after, edge_masks=self._edge_masks.get(dim))
        return x_BTHWC

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        """``x_BTHWC``: ``(B, T, H, W, C)`` ROW_MAJOR, C already tile-aligned."""
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT, f"conv3d needs ROW_MAJOR, got {x_BTHWC.layout}"

        if any(self.local_pad_h) or any(self.local_pad_w):
            x_BTHWC = reflect_pad_hw(x_BTHWC, self.local_pad_h, self.local_pad_w)
        if any(self.external_pad_h) or any(self.external_pad_w):
            x_BTHWC = self._halo_pad(x_BTHWC)
        if self.time_pad > 0:
            x_BTHWC = causal_pad_t(
                x_BTHWC, self.time_pad, self.mesh_device, self._causal_zeros, values=self.causal_pad_values
            )

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


def causal_pad_t(
    x_BTHWC: ttnn.Tensor,
    pad: int,
    mesh_device: ttnn.MeshDevice,
    cache: dict | None = None,
    values: Sequence[float] | None = None,
) -> ttnn.Tensor:
    """Prepend ``pad`` constant frames on T. Nothing is appended -- that is the causality.

    ``values`` is the per-channel fill, zeros by default; a pixel-norm-folded ``conv_in``
    passes ``255 * mean`` so the pad normalizes to the reference's zero (channels past
    ``len(values)`` -- the tile-alignment pad -- stay zero, their weights are zero anyway).

    ``cache`` holds the block across calls. Without it this allocates and **writes**
    a fresh tensor on every convolution -- 34 MB at block 0, thirteen times per unit --
    and a host-to-device write both costs PCIe time on the critical path and makes the
    encoder impossible to capture into a trace ("Writes are not supported during trace
    capture"). The block is constant, so one allocation serves every call.
    """
    B, _, H, W, C = x_BTHWC.shape
    key = (B, pad, H, W, C, x_BTHWC.get_dtype())
    block = None if cache is None else cache.get(key)
    if block is None:
        if values is None:
            block = ttnn.zeros(
                (B, pad, H, W, C), dtype=x_BTHWC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device
            )
        else:
            filled = torch.zeros(B, pad, H, W, C)
            filled[..., : len(values)] = torch.tensor(list(values))
            block = ttnn.from_torch(filled, dtype=x_BTHWC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        if cache is not None:
            cache[key] = block
    return ttnn.concat([block, x_BTHWC], dim=1)


def reflect_edge_correction(
    padded_BTHWC: ttnn.Tensor,
    dim: int,
    pad_before: int,
    pad_after: int | None = None,
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
    if pad_after is None:
        pad_after = pad_before
    if pad_before <= 0 and pad_after <= 0:
        return padded_BTHWC
    size = padded_BTHWC.shape[dim]
    # Each edge is corrected against its own pad width, so an asymmetric halo -- the
    # downsamplers' (0, 1) -- corrects only the trailing edge and leaves the leading one
    # alone, which is what having no leading pad means.
    for leading, pad in ((True, pad_before), (False, pad_after)):
        for j in range(pad):
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

    The mask has to be placed along **the same mesh axis the activation is sharded on**.
    ``ShardTensorToMesh(dim=0)`` distributes linearly over all 32 devices and ignores which
    axis a shard belongs to, so on a 4x8 mesh it lands correctly only for the
    fastest-varying axis: with width on axis 1 the device at ``(i, j)`` happens to receive
    mask row ``j``, which is right, while with height on axis 0 it receives row ``j`` when it
    needs row ``i``, putting the edge corrections on the wrong devices (measured worst-element
    error 7.117 for ``h4`` against 2.4e-07 for ``w8``). ``ShardTensor2dMesh`` with the other
    axis ``None`` replicates along it and is correct for either.
    """
    leading = torch.zeros(factor, 1, 1, 1, 1)
    trailing = torch.zeros(factor, 1, 1, 1, 1)
    leading[0] = 1.0
    trailing[-1] = 1.0
    dims: list[int | None] = [None, None]
    dims[mesh_axis] = 0
    return tuple(
        ttnn.from_torch(
            mask,
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=tuple(dims), mesh_shape=tuple(mesh_device.shape)),
        )
        for mask in (leading, trailing)
    )


def reflect_pad_hw(x_BTHWC: ttnn.Tensor, pad_h: int | tuple[int, int], pad_w: int | tuple[int, int]) -> ttnn.Tensor:
    """Reflect pad on H (dim 2) and W (dim 3); each may be ``(before, after)``.

    Reflect excludes the edge itself, so a pad of 1 takes row 1 rather than row 0 --
    which is exactly where it differs from ``replicate``, and the reason this is done
    locally rather than through ``neighbor_pad_async``. An asymmetric ``(0, 1)`` is the
    downsamplers' bottom/right pre-pad.
    """
    for pad, dim in ((pad_h, 2), (pad_w, 3)):
        before, after = (pad, pad) if isinstance(pad, int) else pad
        if before <= 0 and after <= 0:
            continue
        size = x_BTHWC.shape[dim]
        if size < max(before, after) + 1:
            raise ValueError(f"cannot reflect-pad dim {dim} of size {size} by {(before, after)}")
        leading = [slice_dim(x_BTHWC, dim, i, i + 1) for i in range(before, 0, -1)]
        trailing = [slice_dim(x_BTHWC, dim, size - 1 - i, size - i) for i in range(1, after + 1)]
        x_BTHWC = ttnn.concat([*leading, x_BTHWC, *trailing], dim=dim)
    return x_BTHWC


def slice_dim(x: ttnn.Tensor, dim: int, start: int, stop: int) -> ttnn.Tensor:
    starts = [0] * len(x.shape)
    stops = list(x.shape)
    starts[dim], stops[dim] = start, stop
    return ttnn.slice(x, starts, stops)
