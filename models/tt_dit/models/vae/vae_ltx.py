# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
LTX-2 Video VAE for tt_dit.

Implements the CausalConv3d building block using ttnn.experimental.conv3d,
reusing the Wan VAE's Conv3D infrastructure (blocking configs, weight preparation).

The LTX-2 VAE uses:
- CausalConv3d: 3D convolution with causal temporal padding (repeat first frame)
- PixelNorm: Per-pixel normalization
- SpaceToDepthDownsample: Reshape-based spatial downsampling with residual
- DepthToSpaceUpsample: Reshape-based spatial upsampling
- ResnetBlock3D: Standard pre-norm residual block with two CausalConv3d layers

Reference: LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/
"""

from __future__ import annotations

from typing import Sequence

import torch
from loguru import logger

import ttnn

from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...utils.conv3d import ConvDims, _ntuple, aligned_channels, conv_pad_height, conv_pad_width, get_conv3d_config
from ...utils.tensor import typed_tensor


def _get_w_mask(cache, x_BTHWC, logical_w, parallel_config, mesh_device, dtype):
    """Cached mask that zeros width-padding columns beyond logical_w.

    C++ neighbor_pad supports H masking via ``logical_h`` but not W; this
    pre-conv mul-mask covers the W case.
    """
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
    """
    LTX-2 CausalConv3d using ``ttnn.experimental.conv3d`` with halo exchange.

    Mirrors ``WanCausalConv3d``: H and W are sharded across the mesh and the
    border rows/cols are exchanged via ``ccl_manager.neighbor_pad_persistent_buffer``
    before the conv. Temporal padding (repeat first frame ``kernel_t - 1`` times)
    is handled externally on every device — T is not sharded.

    LTX-specific differences vs Wan's conv:
    - No cache mechanism (processes full video, no chunked streaming).
    - Temporal pad is *frame repeat* (causal=True front-only, causal=False
      symmetric front+back) rather than the zero-pad Wan uses. Opt into
      torch-style symmetric zero padding via ``temporal_padding_mode="zeros"``
      (matches ``torch.nn.Conv3d(padding=k//2)``; required by the LTX latent
      upsampler).
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
    ) -> None:
        super().__init__()

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

        # Temporal padding: ``time_pad`` first-frame copies prepended (causal) or
        # split front/back (non-causal). T is replicated, so this is local.
        self.time_pad = self.kernel_size[0] - 1

        # Spatial padding split (Wan pattern): when H/W is sharded we do the
        # padding externally via neighbor_pad (halo exchange); otherwise internal
        # padding inside conv3d handles it.
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
            # torch.nn.Conv3d(padding=k//2)-equivalent: disable external frame
            # repeat and let conv3d handle symmetric temporal zero padding.
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

        from models.common.utility_functions import is_blackhole

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if is_blackhole() else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)

        self._w_mask_cache: dict[tuple, ttnn.Tensor] = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Prepare Conv3d weights from PyTorch format."""
        # LTX-2 stores weights under "conv.weight" and "conv.bias"
        if "conv.weight" in state:
            state["weight"] = state.pop("conv.weight")
        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias")

        if "weight" in state:
            weight = state["weight"]
            bias = state.get("bias")

            # Pad out_channels if needed
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
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        causal: bool = True,
        logical_h: int = 0,
        logical_w: int = 0,
    ) -> ttnn.Tensor:
        """
        Args:
            x_BTHWC: (B, T, H_per_device, W_per_device, C) ROW_MAJOR, H/W fractured on the mesh
            causal: True = causal temporal pad (front only). False = symmetric.
            logical_h: pre-pad H (full, unsharded). 0 = no H masking needed.
            logical_w: pre-pad W (full, unsharded). 0 = no W masking needed.

        Returns:
            (B, T_out, H_per_device, W_per_device, C_out) ROW_MAJOR.
        """
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT

        # Temporal padding (T is not sharded — local op on every device).
        if self.time_pad > 0:
            first_frame = x_BTHWC[:, :1, :, :, :]
            if causal:
                padding_frames = [first_frame] * self.time_pad
                x_BTHWC = ttnn.concat([*padding_frames, x_BTHWC], dim=1)
            else:
                last_frame = x_BTHWC[:, -1:, :, :, :]
                front_pad = self.time_pad // 2
                back_pad = self.time_pad // 2
                front_frames = [first_frame] * front_pad
                back_frames = [last_frame] * back_pad
                parts = front_frames + [x_BTHWC] + back_frames
                x_BTHWC = ttnn.concat(parts, dim=1)

        # Halo exchange on H/W when sharded. ``external_padding`` is non-zero
        # only on the dimensions that have ``factor > 1``.
        h_pad_needed = self.external_padding[1] > 0 and self.parallel_config.height_parallel.factor > 1
        w_pad_needed = self.external_padding[2] > 0 and self.parallel_config.width_parallel.factor > 1

        # Width pre-conv mul-mask: zero pad columns before the halo so they don't
        # propagate non-zero values through the conv (neighbor_pad has no W-mask).
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


def _neighbor_pad_num_links(ccl_manager: CCLManager, input_tensor: ttnn.Tensor, dim: int) -> int:
    """Neighbor pad uses at most the product of upper dims as link count."""
    upper_dims = 1
    for i in range(dim):
        upper_dims *= input_tensor.shape[i]
    return min(upper_dims, ccl_manager.num_links)


class LTXPixelNorm(Module):
    """
    Per-pixel RMS normalization: x / sqrt(mean(x², dim=channel) + eps).

    No learned parameters. In BTHWC layout, normalizes along the last (channel) dim.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x shape: (B, T, H, W, C) or (1, B, T*H*W, C)
        # PixelNorm: x / sqrt(mean(x², dim=-1, keepdim=True) + eps)
        x_sq = ttnn.multiply(x, x)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_sq, self.eps))
        return ttnn.multiply(x, ttnn.reciprocal(rms))


class LTXResnetBlock3D(Module):
    """
    LTX-2 Residual block: PixelNorm → SiLU → CausalConv3d × 2 with skip connection.

    Architecture:
        x → norm1 → silu → conv1 → norm2 → silu → conv2 → + residual
        x → [norm3 → conv_shortcut if in_c != out_c] ------↗
    """

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
        self.in_channels = in_channels
        self.out_channels = out_channels

        conv_kwargs = dict(
            mesh_device=mesh_device,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
        )

        self.norm1 = LTXPixelNorm()
        self.conv1 = LTXCausalConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, conv_dims=conv_dims, **conv_kwargs
        )

        self.norm2 = LTXPixelNorm()
        self.conv2 = LTXCausalConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, conv_dims=conv_dims, **conv_kwargs
        )

        self.has_shortcut = in_channels != out_channels
        if self.has_shortcut:
            # 1x1x1 conv for channel projection (no spatial/temporal change)
            self.conv_shortcut = LTXCausalConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, conv_dims=conv_dims, **conv_kwargs
            )
            # norm3 is GroupNorm(1) in PyTorch = LayerNorm over channels
            # We store the learned weight/bias as Parameters and apply manually
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
        """
        Args:
            x_BTHWC: (B, T, H, W, C) in ROW_MAJOR layout
            causal: Temporal padding mode for convolutions
            logical_h / logical_w: pre-pad full spatial dims for masking inside convs

        Returns:
            (B, T, H, W, C_out) in ROW_MAJOR layout
        """
        residual = x_BTHWC

        # Main path: norm → silu → conv → norm → silu → conv
        h = self.norm1(x_BTHWC)
        h = ttnn.silu(h)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT) if h.layout != ttnn.ROW_MAJOR_LAYOUT else h
        h = self.conv1(h, causal=causal, logical_h=logical_h, logical_w=logical_w)

        h = self.norm2(h)
        h = ttnn.silu(h)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT) if h.layout != ttnn.ROW_MAJOR_LAYOUT else h
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
    """
    LTX-2 UNet mid block (res_x type): stack of ResnetBlock3D with same in/out channels.

    No attention, no timestep conditioning (22B checkpoint: timestep_conditioning=False).

    Reference: LTX-2/packages/ltx-core/src/ltx_core/model/video_vae/resnet.py UNetMidBlock3D
    """

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
    """
    LTX-2 upsampler: Conv3d → depth-to-space reshape.

    Converts channels back to spatial/temporal dimensions:
    (B, T, H, W, C_out) → reshape → permute → (B, T*p1, H*p2, W*p3, C)

    If temporal stride=2, removes the first frame (causal padding artifact).
    """

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
        import math

        self.stride = stride
        self.out_channels_reduction_factor = out_channels_reduction_factor
        self.residual = residual
        self.in_channels = in_channels
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
        )

    def _depth_to_space_bthwc(self, x: ttnn.Tensor, B: int, T: int, H: int, W: int) -> ttnn.Tensor:
        """Apply depth-to-space in BTHWC format: (B,T,H,W,C*p1*p2*p3) -> (B,T*p1,H*p2,W*p3,C)."""
        p1, p2, p3 = self.stride
        total_c = x.shape[-1]
        C = total_c // (p1 * p2 * p3)
        x = ttnn.reshape(x, (B, T, H, W, C, p1, p2, p3))
        x = ttnn.permute(x, (0, 1, 5, 2, 6, 3, 7, 4))
        x = ttnn.reshape(x, (B, T * p1, H * p2, W * p3, C))
        return x

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        causal: bool = True,
        logical_h: int = 0,
        logical_w: int = 0,
    ) -> tuple[ttnn.Tensor, int, int]:
        """Upsample by `stride`; returns (out, new_logical_h, new_logical_w).

        Depth-to-space scales H by p2 and W by p3, so logicals scale the same way.
        ``logical_h``/``logical_w`` of 0 stays 0 (no masking needed downstream).
        """
        import math

        B, T, H, W, _ = x_BTHWC.shape
        p1, p2, p3 = self.stride

        # Residual path: depth-to-space the input, repeat channels to match output
        if self.residual:
            x_in = self._depth_to_space_bthwc(x_BTHWC, B, T, H, W)
            # x_in shape: (B, T*p1, H*p2, W*p3, C_small)
            # Repeat channels to match conv output after depth-to-space
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            if num_repeat > 1:
                x_in = ttnn.repeat(x_in, ttnn.Shape([1, 1, 1, 1, num_repeat]))
            if p1 == 2:
                x_in = x_in[:, 1:, :, :, :]

        x_BTHWC = self.conv(x_BTHWC, causal=causal, logical_h=logical_h, logical_w=logical_w)

        # Depth-to-space on conv output
        x = self._depth_to_space_bthwc(x_BTHWC, B, T, H, W)

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


class LTXSpaceToDepthDownsample(Module):
    """
    LTX-2 downsampler: space-to-depth reshape + Conv3d + residual mean-pool.

    Converts spatial/temporal dimensions to channels:
    (B, T, H, W, C) → reshape → (B, T//p1, H//p2, W//p3, C*p1*p2*p3)
    Then conv + residual (mean-pooled skip).
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
        import math

        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels

        conv_out_channels = out_channels // math.prod(stride)
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

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        B, T, H, W, C = x_BTHWC.shape
        p1, p2, p3 = self.stride

        # Temporal causal padding: duplicate first frame
        if p1 == 2:
            first = x_BTHWC[:, :1, :, :, :]
            x_BTHWC = ttnn.concat([first, x_BTHWC], dim=1)
            T = T + 1

        # Space-to-depth for skip connection: (B, T, H, W, C) → (B, T//p1, H//p2, W//p3, C*p1*p2*p3)
        x_skip = ttnn.reshape(x_BTHWC, (B, T // p1, p1, H // p2, p2, W // p3, p3, C))
        x_skip = ttnn.permute(x_skip, (0, 1, 3, 5, 7, 2, 4, 6))
        x_skip = ttnn.reshape(x_skip, (B, T // p1, H // p2, W // p3, C * p1 * p2 * p3))

        # Group and mean-pool the skip
        out_c = x_skip.shape[-1] // self.group_size
        x_skip = ttnn.reshape(x_skip, (B, T // p1, H // p2, W // p3, out_c, self.group_size))
        x_skip = ttnn.mean(x_skip, dim=-1)
        x_skip = ttnn.reshape(x_skip, (B, T // p1, H // p2, W // p3, out_c))

        # Conv path: space-to-depth on conv output
        x_conv = self.conv(x_BTHWC)
        T_conv = x_conv.shape[1]
        H_conv, W_conv = x_conv.shape[2], x_conv.shape[3]
        C_conv = x_conv.shape[4]
        x_conv = ttnn.reshape(x_conv, (B, T_conv // p1, p1, H_conv // p2, p2, W_conv // p3, p3, C_conv))
        x_conv = ttnn.permute(x_conv, (0, 1, 3, 5, 7, 2, 4, 6))
        x_conv = ttnn.reshape(x_conv, (B, T_conv // p1, H_conv // p2, W_conv // p3, C_conv * p1 * p2 * p3))

        return ttnn.add(x_conv, x_skip)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pass


def _compute_ltx_decoder_dims(
    *,
    decoder_blocks: list[tuple[str, dict]],
    in_channels: int,
    base_channels: int,
    patch_size: int,
    num_frames: int | None,
    height: int | None,
    width: int | None,
    h_factor: int,
    w_factor: int,
) -> list[ConvDims] | None:
    """Walk the decoder graph to produce one ``ConvDims`` per construction site.

    Returns a list in the exact construction order used by ``LTXVideoDecoder.__init__``:
    ``[conv_in_dims, *(up_block_dims for up_block in reversed(decoder_blocks)), conv_out_dims]``.
    Returns ``None`` when any of ``num_frames``/``height``/``width`` is ``None`` —
    the caller falls back to channel-only blocking lookup.

    Convention (matches Wan ``compute_decoder_dims``):
    - ``T = current_T + (kernel_t - 1)`` — value the conv3d kernel actually sees
      after the external temporal pad. For LTX kernel_t=3, that's ``cur_T + 2``.
    - ``H`` / ``W`` = per-device unpadded spatial dim (after H/W sharding).
    """
    if num_frames is None or height is None or width is None:
        return None

    # Spatial latent geometry. Match pipeline: latent_h = height // 32 / h_factor,
    # latent_w = width // 32 / w_factor. Both must divide evenly for these dims to
    # be valid; otherwise we punt to the channel-only fallback path.
    spatial_compression = 32
    full_lat_h = height // spatial_compression
    full_lat_w = width // spatial_compression
    if full_lat_h % h_factor != 0 or full_lat_w % w_factor != 0:
        return None
    cur_H = full_lat_h // h_factor
    cur_W = full_lat_w // w_factor

    # Latent T = (num_frames - 1) // 8 + 1 (temporal compression factor 8 in the decoder).
    cur_T = (num_frames - 1) // 8 + 1

    feature_channels = base_channels * 8

    def k3_dims(c_in: int, c_out: int) -> ConvDims:
        return ConvDims(T=cur_T + 2, H=cur_H, W=cur_W)

    dims: list[ConvDims] = []

    # conv_in: in_channels → feature_channels, kernel=(3,3,3)
    dims.append(k3_dims(in_channels, feature_channels))

    ch = feature_channels
    for block_name, block_params in reversed(decoder_blocks):
        block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

        if block_name in ("compress_all", "compress_space", "compress_time"):
            stride_map = {
                "compress_all": (2, 2, 2),
                "compress_space": (1, 2, 2),
                "compress_time": (2, 1, 1),
            }
            p1, p2, p3 = stride_map[block_name]
            multiplier = block_config.get("multiplier", 1)
            # conv inside the upsample runs at the PRE-upsample shape
            dims.append(k3_dims(ch, ch * p1 * p2 * p3 // multiplier))
            # Post-upsample shape: depth-to-space expands H,W by p2,p3 and T by p1,
            # then for p1==2 the first frame is removed.
            cur_H = cur_H * p2
            cur_W = cur_W * p3
            cur_T = cur_T * p1 - (1 if p1 == 2 else 0)
            ch = ch // multiplier
        elif block_name == "res_x_y":
            multiplier = block_config.get("multiplier", 2)
            dims.append(k3_dims(ch, ch // multiplier))
            ch = ch // multiplier
        elif block_name == "res_x":
            dims.append(k3_dims(ch, ch))
            # shape unchanged
        else:
            raise ValueError(f"Unknown decoder block: {block_name}")

    # conv_out: ch → out_channels * patch_size**2
    dims.append(k3_dims(ch, 1))  # c_out unused; only T/H/W matter

    return dims


class LTXVideoDecoder(Module):
    """
    LTX-2 Video VAE Decoder (TTNN).

    Decodes latent representation to video:
    (B, 128, F', H', W') → (B, 3, F, H, W)

    Architecture:
    1. Per-channel denormalization (learned mean/std)
    2. conv_in: CausalConv3d 128 → 1024
    3. up_blocks: sequence of DepthToSpaceUpsample/ResnetBlock3D
    4. PixelNorm → SiLU → conv_out: CausalConv3d → 48
    5. unpatchify: reshape 48 → 3 with 4x spatial expansion
    """

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

        # Per-channel statistics (learned mean/std for denormalization). Replicated
        # across H/W shards — applied as a per-channel bias/scale, which is local
        # to each device.
        self.per_channel_mean = Parameter(total_shape=[1, in_channels], device=mesh_device, dtype=ttnn.float32)
        self.per_channel_std = Parameter(total_shape=[1, in_channels], device=mesh_device, dtype=ttnn.float32)

        dims_list = _compute_ltx_decoder_dims(
            decoder_blocks=decoder_blocks,
            in_channels=in_channels,
            base_channels=base_channels,
            patch_size=patch_size,
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

            if block_name in ("compress_all", "compress_space", "compress_time"):
                stride_map = {
                    "compress_all": (2, 2, 2),
                    "compress_space": (1, 2, 2),
                    "compress_time": (2, 1, 1),
                }
                multiplier = block_config.get("multiplier", 1)
                residual = block_config.get("residual", False)
                new_ch = ch // multiplier
                self.up_blocks.append(
                    LTXDepthToSpaceUpsample(
                        in_channels=ch,
                        stride=stride_map[block_name],
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

        # Output: PixelNorm → SiLU → conv_out
        self.norm_out = LTXPixelNorm()
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

    def forward(self, sample_BCTHW: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to video.

        Args:
            sample_BCTHW: (B, 128, F', H', W') torch tensor (latent space)

        Returns:
            (B, 3, F, H, W) torch tensor (pixel space)
        """
        from ...utils.tensor import fast_device_to_host, typed_tensor_2dshard

        # Pad H/W to mesh factors so each device gets an integer shard; track the
        # pre-pad dims as logical_h/logical_w so each conv masks the pad slots.
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

        # Denormalize: x = x * std + mean. Per-channel stats are replicated on the
        # mesh — applies elementwise per device.
        mean = self.per_channel_mean.data
        std = self.per_channel_std.data
        sample_tt = ttnn.add(ttnn.multiply(sample_tt, std), mean)
        sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)

        # logical_h / logical_w scale up through each LTXDepthToSpaceUpsample so
        # later convs mask the correct (now-larger) real region.
        sample_tt = self.conv_in(sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w)
        for up_block in self.up_blocks:
            if isinstance(up_block, LTXDepthToSpaceUpsample):
                sample_tt, logical_h, logical_w = up_block(
                    sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w
                )
            else:
                sample_tt = up_block(sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w)

        sample_tt = self.norm_out(sample_tt)
        sample_tt = ttnn.silu(sample_tt)
        sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)
        sample_tt = self.conv_out(sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w)

        # Gather fractured (B, T, H_per_device, W_per_device, C_out) back to a
        # single torch tensor. fast_device_to_host concatenates along the dims
        # we mapped each mesh axis to.
        concat_dims = [None, None]
        concat_dims[self.parallel_config.height_parallel.mesh_axis] = 2
        concat_dims[self.parallel_config.width_parallel.mesh_axis] = 3
        result = fast_device_to_host(
            sample_tt,
            self.mesh_device,
            concat_dims,
            ccl_manager=self.ccl_manager,
        )  # (B, T_out, H_out, W_out, C_out)

        # Crop padded H/W rows/columns before the final depth-to-space unpatch.
        result = result[:, :, :logical_h, :logical_w, :]

        # Unpatchify: (B, T, H/4, W/4, 48) → (B, 3, T, H, W) via depth-to-space.
        result = result.permute(0, 4, 1, 2, 3)
        from einops import rearrange

        result = rearrange(
            result,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=1,
            q=self.patch_size,
            r=self.patch_size,
        )

        return result
