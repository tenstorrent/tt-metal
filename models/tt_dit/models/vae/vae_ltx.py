# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 Video VAE decoder for tt_dit; reuses the Wan VAE Conv3D infra (blocking, weight prep)."""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING, Sequence

import torch
from einops import rearrange
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

import ttnn
from models.common.utility_functions import is_blackhole

from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import RMSNorm
from ...parallel.config import DiTParallelConfig, VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...utils import cache as cache_module
from ...utils.conv3d import (
    ConvDims,
    _ntuple,
    aligned_channels,
    conv3d_blocking_hash,
    conv_pad_height,
    conv_pad_in_channels,
    conv_pad_width,
    get_conv3d_config,
)
from ...utils.ltx import pad_hw_replicate
from ...utils.tensor import fast_device_to_host, float_to_uint8, typed_tensor, typed_tensor_2dshard

if TYPE_CHECKING:
    from ..upsampler.latent_upsampler_ltx import LTXLatentUpsampler


def _get_w_mask(cache, x_BTHWC, logical_w, parallel_config, mesh_device, dtype):
    """Cached mask that zeros width-padding columns beyond logical_w (neighbor_pad has no W mask)."""
    sharded_w = x_BTHWC.shape[3]
    key = (sharded_w, logical_w)
    if key not in cache:
        padded_w = sharded_w * parallel_config.width_parallel.factor
        mask = torch.ones(1, 1, 1, padded_w, 1)
        mask[:, :, :, logical_w:, :] = 0.0
        cache[key] = typed_tensor(
            mask,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axis=parallel_config.width_parallel.mesh_axis,
            shard_dim=3,
            dtype=dtype,
        )
    return cache[key]


class LTXCausalConv3d(Module):
    """LTX-2 CausalConv3d (ttnn.experimental.conv3d + halo exchange); mirrors WanCausalConv3d.

    Temporal pad is frame-repeat (causal front-only or symmetric); ``temporal_padding_mode="zeros"``
    switches to torch-style symmetric zero padding (required by the LTX latent upsampler).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: Sequence[int] | int = 1,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        conv_dims: ConvDims | None = None,
        temporal_padding_mode: str = "repeat",
        depth_to_space_stride: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()

        # When set, output channels are reordered at load to (p1,p2,p3,C); see
        # _depth_to_space_channels_last.
        self.depth_to_space_stride = depth_to_space_stride

        if temporal_padding_mode not in ("repeat", "zeros"):
            raise ValueError(f"temporal_padding_mode must be 'repeat' or 'zeros' (got {temporal_padding_mode!r})")
        self.temporal_padding_mode = temporal_padding_mode

        self.unpadded_in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = max(32, out_channels)  # Minimum tile width
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        self.dtype = dtype

        # Temporal pad (first-frame copies) is local — T is replicated, not sharded.
        self.time_pad = self.kernel_size[0] - 1

        # Spatial pad split (Wan pattern): external neighbor_pad when sharded, else internal conv pad.
        pad_h = self.kernel_size[1] // 2
        pad_w = self.kernel_size[2] // 2
        external_padding = [0, pad_h, pad_w]
        internal_padding = [0, pad_h, pad_w]
        if self.parallel_config.height_parallel.factor > 1:
            internal_padding[1] = 0
        else:
            external_padding[1] = 0
        if self.parallel_config.width_parallel.factor > 1:
            internal_padding[2] = 0
        else:
            external_padding[2] = 0

        if temporal_padding_mode == "zeros":
            # torch.nn.Conv3d(padding=k//2)-equivalent: conv3d handles symmetric temporal zero pad.
            self.time_pad = 0
            internal_padding[0] = self.kernel_size[0] // 2

        self.external_padding = tuple(external_padding)
        self.internal_padding = tuple(internal_padding)

        dims_T, dims_H, dims_W = (conv_dims.T, conv_dims.H, conv_dims.W) if conv_dims is not None else (0, 0, 0)
        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=self.parallel_config.height_parallel.factor,
            w_factor=self.parallel_config.width_parallel.factor,
            T=dims_T,
            H=dims_H,
            W=dims_W,
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if (is_blackhole() and dtype == ttnn.float32)
            else ttnn.MathFidelity.HiFi2,  # Do not use HiFi3/4 with fp32_dest_acc on WH due to accuracy issues.
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)

        self._w_mask_cache: dict[tuple, ttnn.Tensor] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # LTX-2 stores weights under "conv.weight" and "conv.bias"
        if "conv.weight" in state:
            state["weight"] = state.pop("conv.weight")
        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias")

        if "weight" in state:
            weight = state["weight"]
            bias = state.get("bias")

            # Pad input channels up to the tile-aligned count (e.g. encoder conv_in: 48 -> 64).
            # Torch conv weight layout is (out, in, kt, kh, kw); dim 1 is the in-channel axis.
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

            if self.depth_to_space_stride is not None:
                weight = _prepare_depth_to_space_channels(weight, self.depth_to_space_stride)
                if bias is not None:
                    bias = _prepare_depth_to_space_channels(bias, self.depth_to_space_stride)
                    state["bias"] = bias

            weight_tt = ttnn.from_torch(weight, dtype=self.dtype, pad_value=0)
            prepared = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=weight_tt, C_in_block=self.conv_config.C_in_block, device=self.mesh_device
            )
            state["weight"] = ttnn.to_torch(ttnn.get_device_tensors(prepared)[0])
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        causal: bool = True,
        logical_h: int = 0,
        logical_w: int = 0,
    ) -> ttnn.Tensor:
        # x_BTHWC: (B, T, H_per_device, W_per_device, C) ROW_MAJOR, H/W fractured on the mesh.
        # logical_h/logical_w: pre-pad full spatial dims for pad masking (0 = no masking).
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT

        # Temporal padding (T is not sharded — local op on every device).
        if self.time_pad > 0:
            first_frame = x_BTHWC[:, :1, :, :, :]
            if causal:
                padding_frames = [first_frame] * self.time_pad
                x_BTHWC = ttnn.concat([*padding_frames, x_BTHWC], dim=1)
            else:
                last_frame = x_BTHWC[:, -1:, :, :, :]
                half_pad = self.time_pad // 2
                front_frames = [first_frame] * half_pad
                back_frames = [last_frame] * half_pad
                parts = front_frames + [x_BTHWC] + back_frames
                x_BTHWC = ttnn.concat(parts, dim=1)

        # Halo exchange on H/W when sharded (external_padding nonzero only where factor > 1).
        h_pad_needed = self.external_padding[1] > 0 and self.parallel_config.height_parallel.factor > 1
        w_pad_needed = self.external_padding[2] > 0 and self.parallel_config.width_parallel.factor > 1

        # Width pre-conv mul-mask: zero pad columns before the halo (neighbor_pad has no W-mask).
        if (
            logical_w > 0
            and self.parallel_config.width_parallel.factor > 1
            and x_BTHWC.shape[3] * self.parallel_config.width_parallel.factor > logical_w
        ):
            x_BTHWC = ttnn.mul(
                x_BTHWC,
                _get_w_mask(self._w_mask_cache, x_BTHWC, logical_w, self.parallel_config, self.mesh_device, self.dtype),
            )

        if h_pad_needed or w_pad_needed:
            dims, pad_left, pad_right = [], [], []
            axes, neighbor_sems, links = [], [], []
            if h_pad_needed:
                dims.append(2)
                pad_left.append(self.external_padding[1])
                pad_right.append(self.external_padding[1])
                axes.append(self.parallel_config.height_parallel.mesh_axis)
                neighbor_sems.append(
                    self.ccl_manager.get_np_ping_pong_semaphore(self.parallel_config.height_parallel.mesh_axis)
                )
                links.append(_neighbor_pad_num_links(self.ccl_manager, x_BTHWC, 2))
            if w_pad_needed:
                dims.append(3)
                pad_left.append(self.external_padding[2])
                pad_right.append(self.external_padding[2])
                axes.append(self.parallel_config.width_parallel.mesh_axis)
                neighbor_sems.append(
                    self.ccl_manager.get_np_ping_pong_semaphore(self.parallel_config.width_parallel.mesh_axis)
                )
                links.append(_neighbor_pad_num_links(self.ccl_manager, x_BTHWC, 3))

            x_BTHWC = self.ccl_manager.neighbor_pad_persistent_buffer(
                x_BTHWC,
                dims=dims,
                pad_left=pad_left,
                pad_right=pad_right,
                padding_mode="zeros",
                axes=axes,
                neighbor_sems=neighbor_sems,
                num_links=links,
                logical_h=(logical_h if h_pad_needed else 0),
                t_front_pad=0,
            )

        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.internal_padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )

        return x_BTHWC


def _prepare_depth_to_space_channels(t: torch.Tensor, stride: tuple[int, int, int]) -> torch.Tensor:
    """Reorder the output-channel dim (dim 0) from (C,p1,p2,p3) grouping to (p1,p2,p3,C).

    Out channels must be divisible by p1*p2*p3.
    """
    p1, p2, p3 = stride
    out = t.shape[0]
    assert out % (p1 * p2 * p3) == 0, f"out_channels {out} not divisible by {p1 * p2 * p3}"
    C = out // (p1 * p2 * p3)
    rest = t.shape[1:]
    t = t.reshape(C, p1, p2, p3, *rest)
    t = t.permute(1, 2, 3, 0, *range(4, t.ndim))
    return t.reshape(out, *rest)


def _neighbor_pad_num_links(ccl_manager: CCLManager, input_tensor: ttnn.Tensor, dim: int) -> int:
    """Neighbor pad uses at most the product of upper dims as link count."""
    upper_dims = 1
    for i in range(dim):
        upper_dims *= input_tensor.shape[i]
    return min(upper_dims, ccl_manager.num_links)


class LTXResnetBlock3D(Module):
    """LTX-2 residual block: RMSNorm+SiLU → CausalConv3d ×2 with optional shortcut projection."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        eps: float = 1e-6,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        conv_dims: ConvDims | None = None,
    ) -> None:
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels

        conv_kwargs = dict(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
        )

        self.norm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.norm1 = RMSNorm(
            embedding_dim=in_channels,
            norm_eps=1e-8,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv1 = LTXCausalConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, conv_dims=conv_dims, **conv_kwargs
        )

        self.norm2 = RMSNorm(
            embedding_dim=out_channels,
            norm_eps=1e-8,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv2 = LTXCausalConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, conv_dims=conv_dims, **conv_kwargs
        )

        self.has_shortcut = in_channels != out_channels
        if self.has_shortcut:
            # 1x1x1 channel-projection conv
            self.conv_shortcut = LTXCausalConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, conv_dims=conv_dims, **conv_kwargs
            )
            # norm3 is GroupNorm(1) = LayerNorm over channels; stored as Parameters, applied manually.
            self.norm3_weight = Parameter(total_shape=[1, in_channels], device=mesh_device, dtype=dtype)
            self.norm3_bias = Parameter(total_shape=[1, in_channels], device=mesh_device, dtype=dtype)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if not self.has_shortcut:
            # Remove shortcut/norm3 keys when using nn.Identity
            keys_to_remove = [k for k in state if k.startswith("conv_shortcut") or k.startswith("norm3")]
            for k in keys_to_remove:
                del state[k]
        else:
            # norm3 is GroupNorm(1) — remap weight/bias to norm3_weight/norm3_bias
            if "norm3.weight" in state:
                state["norm3_weight"] = state.pop("norm3.weight").unsqueeze(0)
            if "norm3.bias" in state:
                state["norm3_bias"] = state.pop("norm3.bias").unsqueeze(0)

        # Remove non_linearity, dropout (no params)
        keys_to_remove = [k for k in state if k.startswith("non_linearity") or k.startswith("dropout")]
        for k in keys_to_remove:
            del state[k]

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        causal: bool = True,
        logical_h: int = 0,
        logical_w: int = 0,
    ) -> ttnn.Tensor:
        residual = x_BTHWC

        # Main path: (norm+silu fused) → conv → (norm+silu fused) → conv. The fused norm outputs
        # TILE; conv3d needs ROW_MAJOR.
        h = self.norm1(x_BTHWC, compute_kernel_config=self.norm_compute_kernel_config)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = self.conv1(h, causal=causal, logical_h=logical_h, logical_w=logical_w)

        h = self.norm2(h, compute_kernel_config=self.norm_compute_kernel_config)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT)
        h = self.conv2(h, causal=causal, logical_h=logical_h, logical_w=logical_w)

        # Skip connection
        if self.has_shortcut:
            residual = ttnn.layer_norm(residual, weight=self.norm3_weight.data, bias=self.norm3_bias.data)
            residual = (
                ttnn.to_layout(residual, ttnn.ROW_MAJOR_LAYOUT)
                if residual.layout != ttnn.ROW_MAJOR_LAYOUT
                else residual
            )
            residual = self.conv_shortcut(residual, causal=causal, logical_h=logical_h, logical_w=logical_w)

        return ttnn.add(residual, h)


class LTXUNetMidBlock3D(Module):
    """LTX-2 UNet mid block (res_x): stack of ResnetBlock3D, same in/out channels, no attention."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_layers: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        conv_dims: ConvDims | None = None,
    ) -> None:
        super().__init__()
        self.res_blocks = ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(
                LTXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    mesh_device=mesh_device,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                    dtype=dtype,
                    conv_dims=conv_dims,
                )
            )

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        causal: bool = True,
        logical_h: int = 0,
        logical_w: int = 0,
    ) -> ttnn.Tensor:
        for block in self.res_blocks:
            x_BTHWC = block(x_BTHWC, causal=causal, logical_h=logical_h, logical_w=logical_w)
        return x_BTHWC


class LTXDepthToSpaceUpsample(Module):
    """LTX-2 upsampler: Conv3d → depth-to-space (B,T,H,W,C) → (B,T*p1,H*p2,W*p3,C)."""

    def __init__(
        self,
        *,
        in_channels: int,
        stride: tuple[int, int, int],
        out_channels_reduction_factor: int = 1,
        residual: bool = False,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        conv_dims: ConvDims | None = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.out_channels_reduction_factor = out_channels_reduction_factor
        self.residual = residual
        conv_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor

        self.conv = LTXCausalConv3d(
            in_channels,
            conv_out_channels,
            kernel_size=3,
            stride=1,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
            conv_dims=conv_dims,
            depth_to_space_stride=stride,
        )

    def _depth_to_space_bthwc(self, x: ttnn.Tensor, B: int, T: int, H: int, W: int) -> ttnn.Tensor:
        """Depth-to-space in BTHWC, channel order C,p1,p2,p3: (B,T,H,W,C*p1*p2*p3) -> (B,T*p1,H*p2,W*p3,C).

        For the residual path, whose input is not channel-reordered; conv output uses
        _depth_to_space_channels_last.
        """
        p1, p2, p3 = self.stride
        total_c = x.shape[-1]
        C = total_c // (p1 * p2 * p3)
        x = ttnn.reshape(x, (B, T, H, W, C, p1, p2, p3))
        x = ttnn.permute(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = ttnn.reshape(x, (B, T * p1, H * p2, W * p3, C))
        return x

    def _depth_to_space_channels_last(self, x: ttnn.Tensor, B: int, T: int, H: int, W: int) -> ttnn.Tensor:
        """Depth-to-space for conv output in channel order p1,p2,p3,C (keeps C as the last dim)."""
        p1, p2, p3 = self.stride
        C = x.shape[-1] // (p1 * p2 * p3)
        x = ttnn.reshape(x, (B, T, H, W, p1, p2, p3, C))
        x = ttnn.permute(x, (0, 1, 4, 2, 5, 3, 6, 7))
        x = ttnn.reshape(x, (B, T * p1, H * p2, W * p3, C))
        return x

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        causal: bool = True,
        logical_h: int = 0,
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        """Upsample by `stride`; returns (out, new_logical_h, new_logical_w) scaled by p2/p3."""
        B, T, H, W, _ = x_BTHWC.shape
        p1, p2, p3 = self.stride

        # Residual path: depth-to-space the input, repeat channels to match conv output.
        if self.residual:
            x_in = self._depth_to_space_bthwc(x_BTHWC, B, T, H, W)
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            if num_repeat > 1:
                x_in = ttnn.repeat(x_in, ttnn.Shape([1, 1, 1, 1, num_repeat]))
            if p1 == 2:
                x_in = x_in[:, 1:, :, :, :]

        x_BTHWC = self.conv(x_BTHWC, causal=causal, logical_h=logical_h, logical_w=logical_w)

        # Depth-to-space on conv output (channels reordered to p1,p2,p3,C by self.conv).
        x = self._depth_to_space_channels_last(x_BTHWC, B, T, H, W)

        # Remove first frame if temporal upsampling (causal padding artifact)
        if p1 == 2:
            x = x[:, 1:, :, :, :]

        if self.residual:
            x = ttnn.add(x, x_in)

        new_logical_h = logical_h * p2 if logical_h else 0
        new_logical_w = logical_w * p3 if logical_w else 0
        return x, new_logical_h, new_logical_w

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pass  # Conv handles its own state via LTXCausalConv3d._prepare_torch_state


# Decoder upsample strides (depth-to-space expansion factors).
_DECODER_STRIDE_MAP = {
    "compress_all": (2, 2, 2),
    "compress_space": (1, 2, 2),
    "compress_time": (2, 1, 1),
}


def _compute_ltx_decoder_dims(
    *,
    decoder_blocks: list[tuple[str, dict]],
    num_frames: int | None,
    height: int | None,
    width: int | None,
    h_factor: int,
    w_factor: int,
) -> list[ConvDims] | None:
    """One ConvDims per construction site (in LTXVideoDecoder.__init__ order); None falls back to channel-only blocking.

    Only T/H/W geometry matters for blocking. T is the post-temporal-pad value the conv3d kernel
    sees (cur_T + 2 for kernel_t=3); H/W are per-device.
    """
    if num_frames is None or height is None or width is None:
        return None

    # H/W that don't divide the mesh factor are zero-padded up to the next multiple at
    # runtime (conv_pad_height/width), per stage, so the per-device shard is ceil(full/factor).
    spatial_compression = 32
    full_H = height // spatial_compression
    full_W = width // spatial_compression

    def _dev(full: int, factor: int) -> int:
        return (full + factor - 1) // factor

    # Latent T = (num_frames - 1) // 8 + 1 (temporal compression factor 8 in the decoder).
    cur_T = (num_frames - 1) // 8 + 1

    def k3_dims() -> ConvDims:
        return ConvDims(T=cur_T + 2, H=_dev(full_H, h_factor), W=_dev(full_W, w_factor))

    dims: list[ConvDims] = [k3_dims()]  # conv_in
    for block_name, _block_params in reversed(decoder_blocks):
        if block_name in _DECODER_STRIDE_MAP:
            dims.append(k3_dims())  # conv runs at the PRE-upsample shape
            # Post-upsample: depth-to-space expands H,W by p2,p3 and T by p1 (first frame dropped if p1==2).
            p1, p2, p3 = _DECODER_STRIDE_MAP[block_name]
            full_H = full_H * p2
            full_W = full_W * p3
            cur_T = cur_T * p1 - (1 if p1 == 2 else 0)
        elif block_name in ("res_x_y", "res_x"):
            dims.append(k3_dims())
        else:
            raise ValueError(f"Unknown decoder block: {block_name}")

    dims.append(k3_dims())  # conv_out
    return dims


class LTXVideoDecoder(Module):
    """LTX-2 Video VAE decoder (TTNN): (B, 128, F', H', W') latent → (B, 3, F, H, W) pixels."""

    def __init__(
        self,
        *,
        decoder_blocks: list[tuple[str, dict]],
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        base_channels: int = 128,
        causal: bool = True,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.causal = causal
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        out_channels_with_patch = out_channels * patch_size**2  # 3 * 16 = 48

        feature_channels = base_channels * 8  # 1024

        # Per-channel mean/std for denormalization; replicated across H/W shards (local per device).
        self.per_channel_mean = Parameter(total_shape=[1, in_channels], device=mesh_device, dtype=ttnn.float32)
        self.per_channel_std = Parameter(total_shape=[1, in_channels], device=mesh_device, dtype=ttnn.float32)

        dims_list = _compute_ltx_decoder_dims(
            decoder_blocks=decoder_blocks,
            num_frames=num_frames,
            height=height,
            width=width,
            h_factor=parallel_config.height_parallel.factor,
            w_factor=parallel_config.width_parallel.factor,
        )
        dims_iter = list(dims_list) if dims_list is not None else None

        def _pop_dims() -> ConvDims | None:
            return dims_iter.pop(0) if dims_iter else None

        conv_kwargs = dict(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
        )

        # conv_in: in_channels → 1024
        self.conv_in = LTXCausalConv3d(
            in_channels,
            feature_channels,
            kernel_size=3,
            stride=1,
            conv_dims=_pop_dims(),
            **conv_kwargs,
        )

        # Up blocks (decoder_blocks reversed)
        self.up_blocks = ModuleList()
        ch = feature_channels
        for block_name, block_params in reversed(decoder_blocks):
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            if block_name in _DECODER_STRIDE_MAP:
                multiplier = block_config.get("multiplier", 1)
                residual = block_config.get("residual", False)
                new_ch = ch // multiplier
                self.up_blocks.append(
                    LTXDepthToSpaceUpsample(
                        in_channels=ch,
                        stride=_DECODER_STRIDE_MAP[block_name],
                        out_channels_reduction_factor=multiplier,
                        residual=residual,
                        conv_dims=_pop_dims(),
                        **conv_kwargs,
                    )
                )
                ch = new_ch
            elif block_name == "res_x_y":
                multiplier = block_config.get("multiplier", 2)
                new_ch = ch // multiplier
                self.up_blocks.append(
                    LTXResnetBlock3D(
                        in_channels=ch,
                        out_channels=new_ch,
                        conv_dims=_pop_dims(),
                        **conv_kwargs,
                    )
                )
                ch = new_ch
            elif block_name == "res_x":
                num_layers = block_config.get("num_layers", 1)
                self.up_blocks.append(
                    LTXUNetMidBlock3D(
                        in_channels=ch,
                        num_layers=num_layers,
                        conv_dims=_pop_dims(),
                        **conv_kwargs,
                    )
                )
                # ch stays the same (in == out for mid block)
            else:
                raise ValueError(f"Unknown decoder block: {block_name}")

        # Output: RMSNorm+SiLU fused → conv_out
        self.norm_out_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.norm_out = RMSNorm(
            embedding_dim=ch,
            norm_eps=1e-8,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        self.conv_out = LTXCausalConv3d(
            ch,
            out_channels_with_patch,
            kernel_size=3,
            stride=1,
            conv_dims=_pop_dims(),
            **conv_kwargs,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Map per_channel_statistics (keys use dashes: mean-of-means, std-of-means)
        if "per_channel_statistics.mean-of-means" in state:
            state["per_channel_mean"] = state.pop("per_channel_statistics.mean-of-means").unsqueeze(0)
        if "per_channel_statistics.std-of-means" in state:
            state["per_channel_std"] = state.pop("per_channel_statistics.std-of-means").unsqueeze(0)
        # Remove timestep conditioning keys if present
        keys_to_remove = [
            k
            for k in state
            if k.startswith("timestep_scale_multiplier")
            or k.startswith("last_time_embedder")
            or k.startswith("last_scale_shift_table")
        ]
        for k in keys_to_remove:
            del state[k]
        # Remove conv_act (SiLU, no params) and conv_norm_out (PixelNorm, no params)
        keys_to_remove = [k for k in state if k.startswith("conv_act") or k.startswith("conv_norm_out")]
        for k in keys_to_remove:
            del state[k]

    def forward(self, sample_BCTHW: torch.Tensor, *, output_type: str = "float") -> torch.Tensor:
        """Decode latent (B, 128, F', H', W') → video.

        output_type: "float" → (B, 3, F, H, W) float32 [-1, 1]; "rgb" → (B, 3, F, H, W) uint8 RGB planar.
        """
        # Pad H/W to mesh factors; track pre-pad dims as logical_h/logical_w for conv pad masking.
        sample = sample_BCTHW.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        sample, logical_h = conv_pad_height(sample, self.parallel_config.height_parallel.factor)
        sample, logical_w = conv_pad_width(sample, self.parallel_config.width_parallel.factor)

        sample_tt = typed_tensor_2dshard(
            sample,
            self.mesh_device,
            shard_mapping={
                self.parallel_config.height_parallel.mesh_axis: 2,
                self.parallel_config.width_parallel.mesh_axis: 3,
            },
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        # Denormalize: x = x * std + mean (per-channel stats replicated on the mesh).
        mean = self.per_channel_mean.data
        std = self.per_channel_std.data
        sample_tt = ttnn.add(ttnn.multiply(sample_tt, std), mean)
        sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)

        # logical_h/logical_w scale up through each upsample so later convs mask the right region.
        sample_tt = self.conv_in(sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w)
        for up_block in self.up_blocks:
            if isinstance(up_block, LTXDepthToSpaceUpsample):
                sample_tt, logical_h, logical_w = up_block(
                    sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w
                )
            else:
                sample_tt = up_block(sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w)

        sample_tt = self.norm_out(sample_tt, compute_kernel_config=self.norm_out_compute_kernel_config)
        sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)
        sample_tt = self.conv_out(sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w)

        # Depth-to-space unpatch on device, output BCTHW so the gather's innermost dim stays large
        # (channels-last would gather a length-3 innermost). conv_out channels are ordered (c, p, r, q).
        p, q, r = 1, self.patch_size, self.patch_size
        B_, T_, H4, W4, _ = sample_tt.shape
        sample_tt = ttnn.reshape(sample_tt, (B_, T_, H4, W4, 3, p, r, q))
        sample_tt = ttnn.permute(sample_tt, (0, 4, 1, 5, 2, 7, 3, 6))
        sample_tt = ttnn.reshape(sample_tt, (B_, 3, T_ * p, H4 * q, W4 * r))  # (B, 3, T, H, W)

        concat_dims = [None, None]
        concat_dims[self.parallel_config.height_parallel.mesh_axis] = 3
        concat_dims[self.parallel_config.width_parallel.mesh_axis] = 4
        # uint8 cast rides the gather pre-transfer so the d2h moves one byte/elem (reshape rejects uint8).
        pre_fn = float_to_uint8 if output_type == "rgb" else None
        result = fast_device_to_host(
            sample_tt,
            self.mesh_device,
            concat_dims,
            ccl_manager=self.ccl_manager,
            pre_transfer_fn=pre_fn,
        )
        return result[:, :, :, : logical_h * q, : logical_w * r]  # crop mesh padding


# =============================================================================
# Encoder (I2V image conditioning): pixels -> latent. Symmetric inverse of the decoder.
# =============================================================================

# Encoder downsample strides (mirror reference _make_encoder_block).
_ENCODER_STRIDE_MAP = {
    "compress_time": (2, 1, 1),
    "compress_space": (1, 2, 2),
    "compress_all": (2, 2, 2),
    "compress_all_x_y": (2, 2, 2),
    "compress_time_res": (2, 1, 1),
    "compress_space_res": (1, 2, 2),
    "compress_all_res": (2, 2, 2),
}


class LTXSpaceToDepthDownsample(Module):
    """LTX-2 ``compress_*_res`` downsample (inverse of ``LTXDepthToSpaceUpsample``).

    Stride-1 causal conv at the pre-downsample resolution, followed by a space-to-depth
    rearrange that folds the ``(p1, p2, p3)`` spatial/temporal patch into channels. A residual
    mean-pool skip (also space-to-depth'd) is added. Operates in BTHWC; the space-to-depth is a
    pure per-device reshape (patch groups never cross a shard boundary while per-device H/W stay
    divisible by the spatial stride).
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int, int],
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        conv_dims: ConvDims | None = None,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        prod = math.prod(stride)
        # Skip path groups (in_channels * prod) channels down to out_channels by mean-pooling.
        self.group_size = in_channels * prod // out_channels
        # Inner conv emits out_channels // prod; space-to-depth then expands back to out_channels.
        conv_out_channels = out_channels // prod

        self.conv = LTXCausalConv3d(
            in_channels,
            conv_out_channels,
            kernel_size=3,
            stride=1,
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
            conv_dims=conv_dims,
        )

    def _space_to_depth(self, x: ttnn.Tensor, B: int, d: int, h: int, w: int) -> ttnn.Tensor:
        """(B, d*p1, h*p2, w*p3, C) -> (B, d, h, w, C*p1*p2*p3), channel order (C, p1, p2, p3)."""
        p1, p2, p3 = self.stride
        C = x.shape[-1]
        x = ttnn.reshape(x, (B, d, p1, h, p2, w, p3, C))
        x = ttnn.permute(x, (0, 1, 3, 5, 7, 2, 4, 6))
        x = ttnn.reshape(x, (B, d, h, w, C * p1 * p2 * p3))
        return x

    def _fold_residual(self, x_full: ttnn.Tensor, B: int, d: int, h: int, w: int) -> ttnn.Tensor:
        """Space-to-depth + mean-pool the skip path to out_channels (on the full, cropped tensor)."""
        x_in = self._space_to_depth(x_full, B, d, h, w)
        if self.group_size > 1:
            x_in = ttnn.reshape(x_in, (B, d, h, w, self.out_channels, self.group_size))
            x_in = ttnn.mean(x_in, dim=5)
        return x_in

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        causal: bool = True,
        logical_h: int = 0,
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        p1, p2, p3 = self.stride

        # Temporal causal pad: duplicate the first frame so T becomes divisible by p1.
        if p1 == 2:
            first_frame = x_BTHWC[:, :1, :, :, :]
            x_BTHWC = ttnn.concat([first_frame, x_BTHWC], dim=1)

        # A per-device space-to-depth fold is only correct when each shard's spatial extent is
        # divisible by the stride; otherwise a (p2, p3) patch straddles a shard boundary and the
        # per-device zero-pad below injects garbage into the *interior* of the gathered latent.
        # When that happens — global dim divisible by the stride but per-device dim not — gather the sharded spatial
        # axes, fold on the full extent, then re-shard.
        h_factor = self.parallel_config.height_parallel.factor
        w_factor = self.parallel_config.width_parallel.factor
        needs_h_gather = h_factor > 1 and p2 > 1 and x_BTHWC.shape[2] % p2 != 0
        needs_w_gather = w_factor > 1 and p3 > 1 and x_BTHWC.shape[3] % p3 != 0
        if needs_h_gather or needs_w_gather:
            return self._forward_gathered(x_BTHWC, causal, logical_h, logical_w)

        B, T, H, W, C = x_BTHWC.shape

        # Pad H/W so the space-to-depth reshape is exact. ttnn.pad only pads the low dims of a <=4D
        # tensor, so collapse (B, T), pad both spatial dims, then restore 5D.
        pad_h = (p2 - H % p2) % p2 if p2 > 1 else 0
        pad_w = (p3 - W % p3) % p3 if p3 > 1 else 0
        if pad_h or pad_w:
            x_BTHWC = ttnn.reshape(x_BTHWC, (B * T, H, W, C))
            x_BTHWC = ttnn.pad(x_BTHWC, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], value=0.0)
            H += pad_h
            W += pad_w
            x_BTHWC = ttnn.reshape(x_BTHWC, (B, T, H, W, C))

        d, h, w = T // p1, H // p2, W // p3

        # Residual skip: space-to-depth the input, then mean-pool channel groups to out_channels.
        x_in = self._fold_residual(x_BTHWC, B, d, h, w)

        # Main path: stride-1 conv at full res, then space-to-depth.
        x = self.conv(x_BTHWC, causal=causal, logical_h=logical_h, logical_w=logical_w)
        x = self._space_to_depth(x, B, d, h, w)

        x = ttnn.add(x, x_in)

        new_logical_h = (logical_h // p2) if logical_h else 0
        new_logical_w = (logical_w // p3) if logical_w else 0
        return x, new_logical_h, new_logical_w

    def _all_gather_hw(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Gather the H/W-sharded tensor to full (replicated) spatial extent on every device."""
        pc = self.parallel_config
        if pc.height_parallel.factor > 1:
            x = self.ccl_manager.all_gather(x, dim=2, mesh_axis=pc.height_parallel.mesh_axis, use_hyperparams=False)
        if pc.width_parallel.factor > 1:
            x = self.ccl_manager.all_gather(x, dim=3, mesh_axis=pc.width_parallel.mesh_axis, use_hyperparams=False)
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    def _reshard_hw(self, x: ttnn.Tensor, B: int, d: int, h: int, w: int) -> ttnn.Tensor:
        """Re-shard a replicated (B, d, h, w, C) tensor back across the mesh, zero-padding each
        spatial dim up to a multiple of its mesh factor (tail only) so the split divides evenly."""
        pc = self.parallel_config
        h_factor = pc.height_parallel.factor
        w_factor = pc.width_parallel.factor
        # mesh_partition slices each shard out of the replicated tensor; a sub-tile-wide shard can
        # only be sliced in ROW_MAJOR (tilized slicing requires tile-aligned begin indices).
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        pad_h = (-h) % h_factor if h_factor > 1 else 0
        pad_w = (-w) % w_factor if w_factor > 1 else 0
        if pad_h or pad_w:
            C = x.shape[-1]
            x = ttnn.reshape(x, (B * d, h, w, C))
            x = ttnn.pad(x, [(0, 0), (0, pad_h), (0, pad_w), (0, 0)], value=0.0)
            x = ttnn.reshape(x, (B, d, h + pad_h, w + pad_w, C))
        if h_factor > 1:
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=pc.height_parallel.mesh_axis)
        if w_factor > 1:
            x = ttnn.mesh_partition(x, dim=3, cluster_axis=pc.width_parallel.mesh_axis)
        return x

    def _forward_gathered(
        self, x_BTHWC: ttnn.Tensor, causal: bool, logical_h: int, logical_w: int
    ) -> tuple[ttnn.Tensor, int, int]:
        """Sharding-safe space-to-depth: run the (sharded, halo-aware) conv, then gather H/W to the
        full extent, fold + residual on the full tensor, and re-shard. Used when a per-device
        spatial dim is not divisible by the stride (patch would straddle a shard boundary)."""
        p1, p2, p3 = self.stride

        # Conv first, while still sharded (its halo exchange handles shard boundaries correctly).
        x_conv = self.conv(x_BTHWC, causal=causal, logical_h=logical_h, logical_w=logical_w)
        x_conv = ttnn.to_layout(x_conv, ttnn.ROW_MAJOR_LAYOUT)

        x_res_full = self._all_gather_hw(x_BTHWC)
        x_conv_full = self._all_gather_hw(x_conv)

        # Crop the mesh-factor padding so the fold operates on the true (logical) extent, which is
        # divisible by the stride globally even when the per-device shards are not.
        T = x_res_full.shape[1]
        full_h = logical_h if logical_h > 0 else x_res_full.shape[2]
        full_w = logical_w if logical_w > 0 else x_res_full.shape[3]
        x_res_full = x_res_full[:, :, :full_h, :full_w, :]
        x_conv_full = x_conv_full[:, :, :full_h, :full_w, :]

        B = x_res_full.shape[0]
        d, h, w = T // p1, full_h // p2, full_w // p3

        x_in = self._fold_residual(x_res_full, B, d, h, w)
        x = self._space_to_depth(x_conv_full, B, d, h, w)
        x = ttnn.add(x, x_in)

        x = self._reshard_hw(x, B, d, h, w)
        return x, h, w

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pass  # Conv handles its own state via LTXCausalConv3d._prepare_torch_state


def _compute_ltx_encoder_dims(
    *,
    encoder_blocks: list[tuple[str, dict]],
    num_frames: int | None,
    height: int | None,
    width: int | None,
    h_factor: int,
    w_factor: int,
    patch_size: int = 4,
) -> list[ConvDims] | None:
    """One ConvDims per construction site (in LTXVideoEncoder.__init__ order); None falls back to
    channel-only blocking. T is the post-temporal-pad value the conv3d kernel sees.
    """
    if num_frames is None or height is None or width is None:
        return None

    # Patchify reduces H/W by patch_size before conv_in.
    full_H = height // patch_size
    full_W = width // patch_size

    def _dev(full: int, factor: int) -> int:
        return (full + factor - 1) // factor

    cur_T = num_frames

    def res_dims() -> ConvDims:
        return ConvDims(T=cur_T + 2, H=_dev(full_H, h_factor), W=_dev(full_W, w_factor))

    dims: list[ConvDims] = [res_dims()]  # conv_in
    for block_name, _block_params in encoder_blocks:
        if block_name in _ENCODER_STRIDE_MAP:
            p1, p2, p3 = _ENCODER_STRIDE_MAP[block_name]
            # The inner conv runs at pre-downsample res; temporal prepend adds a frame when p1==2.
            t_pre = cur_T + 1 if p1 == 2 else cur_T
            dims.append(ConvDims(T=t_pre + 2, H=_dev(full_H, h_factor), W=_dev(full_W, w_factor)))
            full_H //= p2
            full_W //= p3
            cur_T = t_pre // p1
        elif block_name in ("res_x", "res_x_y"):
            dims.append(res_dims())
        else:
            raise ValueError(f"Unknown encoder block: {block_name}")

    dims.append(res_dims())  # conv_out
    return dims


class LTXVideoEncoder(Module):
    """LTX-2 Video VAE encoder (TTNN): (B, 3, F, H, W) pixels in [-1, 1] -> (B, 128, F', H', W')
    normalized latent means. Always causal; mirrors the reference ``VideoEncoder``."""

    def __init__(
        self,
        *,
        encoder_blocks: list[tuple[str, dict]],
        in_channels: int = 3,
        out_channels: int = 128,
        patch_size: int = 4,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.bfloat16,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.latent_channels = out_channels
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        in_channels_with_patch = in_channels * patch_size**2  # 3 * 16 = 48

        # Per-channel stats for normalizing the latent means; shared with the decoder.
        self.per_channel_mean = Parameter(total_shape=[1, out_channels], device=mesh_device, dtype=ttnn.float32)
        self.per_channel_std = Parameter(total_shape=[1, out_channels], device=mesh_device, dtype=ttnn.float32)

        dims_list = _compute_ltx_encoder_dims(
            encoder_blocks=encoder_blocks,
            num_frames=num_frames,
            height=height,
            width=width,
            h_factor=parallel_config.height_parallel.factor,
            w_factor=parallel_config.width_parallel.factor,
            patch_size=patch_size,
        )
        dims_iter = list(dims_list) if dims_list is not None else None

        def _pop_dims() -> ConvDims | None:
            return dims_iter.pop(0) if dims_iter else None

        conv_kwargs = dict(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
        )

        # conv_in: 48 -> 128 (feature_channels == latent_channels in the reference encoder).
        feature_channels = out_channels
        self.conv_in = LTXCausalConv3d(
            in_channels_with_patch,
            feature_channels,
            kernel_size=3,
            stride=1,
            conv_dims=_pop_dims(),
            **conv_kwargs,
        )

        self.down_blocks = ModuleList()
        ch = feature_channels
        for block_name, block_params in encoder_blocks:
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params
            if block_name in ("compress_time", "compress_space", "compress_all"):
                # Plain strided causal conv (channels unchanged).
                self.down_blocks.append(
                    LTXCausalConv3d(
                        ch,
                        ch,
                        kernel_size=3,
                        stride=_ENCODER_STRIDE_MAP[block_name],
                        conv_dims=_pop_dims(),
                        **conv_kwargs,
                    )
                )
            elif block_name == "compress_all_x_y":
                multiplier = block_config.get("multiplier", 2)
                new_ch = ch * multiplier
                self.down_blocks.append(
                    LTXCausalConv3d(
                        ch,
                        new_ch,
                        kernel_size=3,
                        stride=_ENCODER_STRIDE_MAP[block_name],
                        conv_dims=_pop_dims(),
                        **conv_kwargs,
                    )
                )
                ch = new_ch
            elif block_name in ("compress_time_res", "compress_space_res", "compress_all_res"):
                multiplier = block_config.get("multiplier", 2)
                new_ch = ch * multiplier
                self.down_blocks.append(
                    LTXSpaceToDepthDownsample(
                        in_channels=ch,
                        out_channels=new_ch,
                        stride=_ENCODER_STRIDE_MAP[block_name],
                        conv_dims=_pop_dims(),
                        **conv_kwargs,
                    )
                )
                ch = new_ch
            elif block_name == "res_x_y":
                multiplier = block_config.get("multiplier", 2)
                new_ch = ch * multiplier
                self.down_blocks.append(
                    LTXResnetBlock3D(in_channels=ch, out_channels=new_ch, conv_dims=_pop_dims(), **conv_kwargs)
                )
                ch = new_ch
            elif block_name == "res_x":
                num_layers = block_config.get("num_layers", 1)
                self.down_blocks.append(
                    LTXUNetMidBlock3D(in_channels=ch, num_layers=num_layers, conv_dims=_pop_dims(), **conv_kwargs)
                )
            else:
                raise ValueError(f"Unknown encoder block: {block_name}")

        # conv_norm_out is PixelNorm (RMS over the channel dim) + SiLU, fused via RMSNorm.
        self.norm_out_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.norm_out = RMSNorm(
            embedding_dim=ch,
            norm_eps=1e-6,
            norm_elementwise_affine=False,
            mesh_device=mesh_device,
            dtype=dtype,
            fused_activation=ttnn.UnaryOpType.SILU,
        )
        # conv_out emits the latent means only (reference outputs 129 = 128 means + 1 logvar; we
        # slice the weight to the first 128 rows in _prepare_torch_state and drop the logvar).
        self.conv_out = LTXCausalConv3d(
            ch,
            out_channels,
            kernel_size=3,
            stride=1,
            conv_dims=_pop_dims(),
            **conv_kwargs,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Map per_channel_statistics (dash keys) -> per_channel_mean/std.
        if "per_channel_statistics.mean-of-means" in state:
            state["per_channel_mean"] = state.pop("per_channel_statistics.mean-of-means").unsqueeze(0)
        if "per_channel_statistics.std-of-means" in state:
            state["per_channel_std"] = state.pop("per_channel_statistics.std-of-means").unsqueeze(0)
        # Drop the logvar output channel(s): means are the first latent_channels rows of conv_out.
        for key in ("conv_out.conv.weight", "conv_out.conv.bias"):
            if key in state and state[key].shape[0] > self.latent_channels:
                state[key] = state[key][: self.latent_channels]
        # conv_norm_out (PixelNorm) and conv_act (SiLU) carry no params.
        keys_to_remove = [k for k in state if k.startswith("conv_act") or k.startswith("conv_norm_out")]
        for k in keys_to_remove:
            del state[k]

    def forward(self, sample_BCTHW: torch.Tensor) -> torch.Tensor:
        """Encode video (B, 3, F, H, W) in [-1, 1] -> normalized latent means (B, 128, F', H', W')."""
        # Crop trailing frames so F == 1 + 8*k (mirrors the reference encoder).
        frames_count = sample_BCTHW.shape[2]
        if (frames_count - 1) % 8 != 0:
            frames_to_crop = (frames_count - 1) % 8
            logger.warning(f"Encoder: cropping last {frames_to_crop} frames to satisfy 1 + 8*k (got {frames_count})")
            sample_BCTHW = sample_BCTHW[:, :, : frames_count - frames_to_crop, ...]

        # Patchify (space-to-depth, patch_size on H/W): (B, 3, F, H, W) -> (B, 48, F, H/4, W/4).
        sample = rearrange(
            sample_BCTHW,
            "b c (f p) (h q) (w r) -> b (c p r q) f h w",
            p=1,
            q=self.patch_size,
            r=self.patch_size,
        )
        sample = sample.permute(0, 2, 3, 4, 1)  # (B, F, H/4, W/4, 48)
        sample = conv_pad_in_channels(sample)  # 48 -> 64 (tile-aligned)

        sample, logical_h = conv_pad_height(sample, self.parallel_config.height_parallel.factor)
        sample, logical_w = conv_pad_width(sample, self.parallel_config.width_parallel.factor)

        sample_tt = typed_tensor_2dshard(
            sample,
            self.mesh_device,
            shard_mapping={
                self.parallel_config.height_parallel.mesh_axis: 2,
                self.parallel_config.width_parallel.mesh_axis: 3,
            },
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        sample_tt = self.conv_in(sample_tt, causal=True, logical_h=logical_h, logical_w=logical_w)
        for down_block in self.down_blocks:
            # conv3d / space-to-depth blocks require ROW_MAJOR input. LTXSpaceToDepthDownsample
            # (and the conv-only compress blocks) return TILE, and two compress blocks can be
            # adjacent in the encoder, so normalize at every block boundary. Resnet/mid blocks
            # also accept ROW_MAJOR (their fused norm re-tilizes internally — same as the decoder).
            sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)
            if isinstance(down_block, LTXSpaceToDepthDownsample):
                sample_tt, logical_h, logical_w = down_block(
                    sample_tt, causal=True, logical_h=logical_h, logical_w=logical_w
                )
            else:
                sample_tt = down_block(sample_tt, causal=True, logical_h=logical_h, logical_w=logical_w)

        sample_tt = self.norm_out(sample_tt, compute_kernel_config=self.norm_out_compute_kernel_config)
        sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)
        sample_tt = self.conv_out(sample_tt, causal=True, logical_h=logical_h, logical_w=logical_w)

        # Normalize means on device: (x - mean) / std (per-channel stats replicated on the mesh).
        mean = self.per_channel_mean.data
        std = self.per_channel_std.data
        sample_tt = ttnn.multiply(ttnn.subtract(sample_tt, mean), ttnn.reciprocal(std))

        concat_dims = [None, None]
        concat_dims[self.parallel_config.height_parallel.mesh_axis] = 2
        concat_dims[self.parallel_config.width_parallel.mesh_axis] = 3
        result = fast_device_to_host(
            sample_tt,
            self.mesh_device,
            concat_dims,
            ccl_manager=self.ccl_manager,
        )  # (B, F', H', W', 128)

        result = result[:, :, :logical_h, :logical_w, :]
        # (B, F', H', W', C) -> (B, C, F', H', W')
        return result.permute(0, 4, 1, 2, 3)


# =============================================================================
# Latent normalization stats + spatial-2x latent upsample (bridges the VAE
# per-channel stats with the standalone latent upsampler).
# =============================================================================


def read_vae_per_channel_stats(checkpoint_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read ``(mean-of-means, std-of-means)`` from a checkpoint and reshape for ``(B, C, F, H, W)``
    broadcast — the un_normalize/normalize bookends matching ``ltx_core.upsample_video``."""
    with safe_open(checkpoint_path, framework="pt") as f:
        mean = f.get_tensor("vae.per_channel_statistics.mean-of-means").float()
        std = f.get_tensor("vae.per_channel_statistics.std-of-means").float()
    return mean.view(1, -1, 1, 1, 1), std.view(1, -1, 1, 1, 1)


class LTXVideoVAEAdapter:
    """Owns the LTX video VAE lifecycle: parses the checkpoint's VAE config, builds the
    decoder + encoder, loads/reloads their weights, and holds the shared per-channel stats.

    Decoder and encoder share ``vae.per_channel_statistics.*`` and the VAE H/W parallel config,
    so a single adapter owns both (mirrors ``WanVAEDecoderAdapter``, which owns just the decoder).
    Either submodule may be ``None`` when the checkpoint lacks the corresponding blocks (or when the
    encoder has no concrete H/W). ``reload_decoder`` / ``reload_encoder`` are idempotent and driven
    by the pipeline; ``decoder`` / ``encoder`` expose the underlying ``Module`` for coresident
    exclusion registration.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        mesh_device: ttnn.MeshDevice,
        vae_parallel_config: VaeHWParallelConfig,
        vae_ccl_manager: CCLManager,
        dit_parallel_config: DiTParallelConfig,
        num_frames: int,
        height: int,
        width: int,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._mesh_device = mesh_device
        self._vae_parallel_config = vae_parallel_config
        self._vae_ccl_manager = vae_ccl_manager
        self._dit_parallel_config = dit_parallel_config
        self._pcs_cache: tuple[torch.Tensor, torch.Tensor] | None = None

        # VAE config from JSON metadata header (no tensor loads).
        with open(checkpoint_path, "rb") as f:
            header_size = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(header_size))
        vae_cfg = json.loads(header.get("__metadata__", {}).get("config", "{}")).get("vae", {})
        self.decoder_blocks = vae_cfg.get("decoder_blocks", [])
        self.encoder_blocks = vae_cfg.get("encoder_blocks", [])
        self._causal = vae_cfg.get("causal_decoder", False)
        self._base_channels = vae_cfg.get("decoder_base_channels", 128)
        self._patch_size = vae_cfg.get("patch_size", 4)
        if self.decoder_blocks:
            logger.info(f"VAE config: {len(self.decoder_blocks)} blocks, causal={self._causal}")
        if self.encoder_blocks:
            logger.info(f"VAE encoder config: {len(self.encoder_blocks)} blocks")

        self._decoder: LTXVideoDecoder | None = None
        if self.decoder_blocks:
            self._decoder = LTXVideoDecoder(
                decoder_blocks=self.decoder_blocks,
                causal=self._causal,
                base_channels=self._base_channels,
                mesh_device=mesh_device,
                parallel_config=vae_parallel_config,
                ccl_manager=vae_ccl_manager,
                num_frames=num_frames or None,
                height=height or None,
                width=width or None,
            )

        # Image-conditioning VAE encoder (I2V): pixels -> latent. Single-frame at construction.
        self._encoder: LTXVideoEncoder | None = None
        if self.encoder_blocks and height > 0 and width > 0:
            self._encoder = LTXVideoEncoder(
                encoder_blocks=self.encoder_blocks,
                patch_size=self._patch_size,
                mesh_device=mesh_device,
                parallel_config=vae_parallel_config,
                ccl_manager=vae_ccl_manager,
                num_frames=1,
                height=height or None,
                width=width or None,
            )

    @property
    def decoder(self) -> "LTXVideoDecoder | None":
        return self._decoder

    @property
    def encoder(self) -> "LTXVideoEncoder | None":
        return self._encoder

    def per_channel_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Cached ``(mean-of-means, std-of-means)`` reshaped for ``(B, C, F, H, W)`` broadcast —
        the un_normalize/normalize bookends matching ``ltx_core.upsample_video``."""
        if self._pcs_cache is None:
            self._pcs_cache = read_vae_per_channel_stats(self._checkpoint_path)
        return self._pcs_cache

    def reload_decoder(self) -> None:
        """Push VAE decoder weights onto the mesh via the disk cache. Blocking-hash subfolder
        forces re-load when conv3d ``C_in_block`` changes (mirrors Wan). A static load keeps the
        VAE resident across the audio decode — skip the per-request reload once loaded."""
        if self._decoder is None or self._decoder.is_loaded():
            return

        def _state_provider() -> dict[str, torch.Tensor]:
            logger.info(f"VAE cache miss — loading safetensors: {self._checkpoint_path}")
            raw = load_file(self._checkpoint_path)
            vae_state = {}
            for k, v in raw.items():
                if k.startswith("vae.decoder."):
                    vae_state[k.removeprefix("vae.decoder.")] = v
                elif k.startswith("vae.per_channel_statistics."):
                    short_key = k.removeprefix("vae.")
                    if short_key in ("per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"):
                        vae_state[short_key] = v
            return vae_state

        blocking_key = conv3d_blocking_hash(self._decoder)
        subfolder = f"vae_{blocking_key}" if blocking_key else "vae"
        cache_module.load_model(
            self._decoder,
            model_name=os.path.basename(self._checkpoint_path).removesuffix(".safetensors"),
            subfolder=subfolder,
            parallel_config=self._dit_parallel_config,
            mesh_shape=tuple(self._mesh_device.shape),
            get_torch_state_dict=_state_provider,
        )
        logger.info(f"Loaded TTNN VAE decoder ({len(self.decoder_blocks)} blocks)")

    def reload_encoder(self) -> None:
        """Push VAE encoder weights onto the mesh (I2V image conditioning). Mirrors
        ``reload_decoder``; shares the decoder's ``vae.per_channel_statistics.*``."""
        if self._encoder is None or self._encoder.is_loaded():
            return

        def _state_provider() -> dict[str, torch.Tensor]:
            logger.info(f"VAE encoder cache miss — loading safetensors: {self._checkpoint_path}")
            raw = load_file(self._checkpoint_path)
            enc_state = {}
            for k, v in raw.items():
                if k.startswith("vae.encoder."):
                    enc_state[k.removeprefix("vae.encoder.")] = v
                elif k.startswith("vae.per_channel_statistics."):
                    short_key = k.removeprefix("vae.")
                    if short_key in ("per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"):
                        enc_state[short_key] = v
            return enc_state

        blocking_key = conv3d_blocking_hash(self._encoder)
        subfolder = f"vae_enc_{blocking_key}" if blocking_key else "vae_enc"
        cache_module.load_model(
            self._encoder,
            model_name=os.path.basename(self._checkpoint_path).removesuffix(".safetensors"),
            subfolder=subfolder,
            parallel_config=self._dit_parallel_config,
            mesh_shape=tuple(self._mesh_device.shape),
            get_torch_state_dict=_state_provider,
        )
        logger.info(f"Loaded TTNN VAE encoder ({len(self.encoder_blocks)} blocks)")


def upsample_latent(
    upsampler: "LTXLatentUpsampler",
    video_latent: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """Spatial-2x latent upsample. Mirrors ``ltx_core.upsample_video``: un_normalize
    → bare upsampler → re_normalize. ``(B, C, F, H, W)`` host in/out.

    ``mean``/``std`` are the VAE per-channel stats reshaped for ``(B, C, F, H, W)`` broadcast
    (see ``read_vae_per_channel_stats``). The caller must load the ``upsampler`` weights onto
    the mesh before invoking this.
    """
    x = video_latent.float() * std + mean
    # Pad to even mesh shards so the upsampler's sharded convs skip the uneven-dim halo
    # crop-masking that seams the 2x4 boundaries (s1 17x30); crop the 2x-upsampled margin
    # off to preserve the field of view. The upsampler is built at these rounded dims so its
    # pinned GroupNorm/conv shapes match.
    pc = upsampler.parallel_config
    x, H, W = pad_hw_replicate(x, pc.height_parallel.factor, pc.width_parallel.factor)
    x = upsampler(x)
    x = x[:, :, :, : H * 2, : W * 2]
    return (x.float() - mean) / std
