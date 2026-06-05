# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 Video VAE decoder for tt_dit; reuses the Wan VAE Conv3D infra (blocking, weight prep)."""

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

        from models.common.utility_functions import is_blackhole

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
                front_pad = self.time_pad // 2
                back_pad = self.time_pad // 2
                front_frames = [first_frame] * front_pad
                back_frames = [last_frame] * back_pad
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


def _neighbor_pad_num_links(ccl_manager: CCLManager, input_tensor: ttnn.Tensor, dim: int) -> int:
    """Neighbor pad uses at most the product of upper dims as link count."""
    upper_dims = 1
    for i in range(dim):
        upper_dims *= input_tensor.shape[i]
    return min(upper_dims, ccl_manager.num_links)


class LTXPixelNorm(Module):
    """Per-pixel RMS norm over the channel dim: x / sqrt(mean(x², dim=-1) + eps). No learned params."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x_sq = ttnn.multiply(x, x)
        mean_sq = ttnn.mean(x_sq, dim=-1, keepdim=True)
        rms = ttnn.sqrt(ttnn.add(mean_sq, self.eps))
        return ttnn.multiply(x, ttnn.reciprocal(rms))


class LTXResnetBlock3D(Module):
    """LTX-2 residual block: PixelNorm → SiLU → CausalConv3d ×2 with optional shortcut projection."""

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
        import math

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
        """Upsample by `stride`; returns (out, new_logical_h, new_logical_w) scaled by p2/p3."""
        import math

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

    stride_map = {"compress_all": (2, 2, 2), "compress_space": (1, 2, 2), "compress_time": (2, 1, 1)}
    dims: list[ConvDims] = [k3_dims()]  # conv_in
    for block_name, _block_params in reversed(decoder_blocks):
        if block_name in stride_map:
            dims.append(k3_dims())  # conv runs at the PRE-upsample shape
            # Post-upsample: depth-to-space expands H,W by p2,p3 and T by p1 (first frame dropped if p1==2).
            p1, p2, p3 = stride_map[block_name]
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
        Decode latent (B, 128, F', H', W') → video (B, 3, F, H, W).
        """
        from ...utils.tensor import fast_device_to_host, typed_tensor_2dshard

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

        sample_tt = self.norm_out(sample_tt)
        sample_tt = ttnn.silu(sample_tt)
        sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)
        sample_tt = self.conv_out(sample_tt, causal=self.causal, logical_h=logical_h, logical_w=logical_w)

        # Gather fractured (B, T, H_per_device, W_per_device, C_out) back to a single torch tensor.
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
