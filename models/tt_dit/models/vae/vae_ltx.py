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
from ...utils.conv3d import _ntuple, aligned_channels, get_conv3d_config


class LTXCausalConv3d(Module):
    """
    LTX-2 CausalConv3d using ttnn.experimental.conv3d.

    Temporal padding: repeats the first frame (kernel_t - 1) times before conv.
    Spatial padding: symmetric (kernel_h//2, kernel_w//2), handled internally by conv3d op.

    This is simpler than Wan's WanCausalConv3d:
    - No cache mechanism (processes full video)
    - No halo exchange (no mesh-parallel spatial sharding for VAE)
    - Simpler temporal padding (repeat first frame vs explicit cache)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: Sequence[int] | int = 1,
        spatial_padding_mode: str = "zeros",
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.unpadded_in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = max(32, out_channels)  # Minimum tile width
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.spatial_padding_mode = spatial_padding_mode

        # Temporal padding: repeat first frame (kernel_t - 1) times
        self.time_pad = self.kernel_size[0] - 1

        # Spatial padding amounts
        self.pad_h = self.kernel_size[1] // 2
        self.pad_w = self.kernel_size[2] // 2

        # For reflect mode, we do manual padding and pass (0,0,0) to conv3d
        # For zeros mode, let conv3d handle it internally
        if spatial_padding_mode == "reflect":
            self.internal_padding = (0, 0, 0)
        else:
            self.internal_padding = (0, self.pad_h, self.pad_w)

        # Get conv3d config (blocking)
        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
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

            # New API: prepare weights via ttnn tensor + C_in_block
            weight_tt = ttnn.from_torch(weight, dtype=self.dtype, pad_value=0)
            prepared = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=weight_tt, C_in_block=self.conv_config.C_in_block, device=self.mesh_device
            )
            state["weight"] = ttnn.to_torch(ttnn.get_device_tensors(prepared)[0])
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x_BTHWC: ttnn.Tensor, causal: bool = True) -> ttnn.Tensor:
        """
        Args:
            x_BTHWC: (B, T, H, W, C) in ROW_MAJOR layout
            causal: If True, pad temporally at front only (causal).
                    If False, pad symmetrically at front and back.

        Returns:
            (B, T_out, H_out, W_out, C_out) in ROW_MAJOR layout
        """
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT

        # Temporal padding
        if self.time_pad > 0:
            first_frame = x_BTHWC[:, :1, :, :, :]
            if causal:
                # Causal: repeat first frame at front only
                padding_frames = [first_frame] * self.time_pad
                x_BTHWC = ttnn.concat([*padding_frames, x_BTHWC], dim=1)
            else:
                # Symmetric: repeat first frame at front, last frame at back
                last_frame = x_BTHWC[:, -1:, :, :, :]
                front_pad = self.time_pad // 2
                back_pad = self.time_pad // 2
                front_frames = [first_frame] * front_pad
                back_frames = [last_frame] * back_pad
                parts = front_frames + [x_BTHWC] + back_frames
                x_BTHWC = ttnn.concat(parts, dim=1)

        # Reflect spatial padding: manually pad H and W dims before conv
        if self.spatial_padding_mode == "reflect" and (self.pad_h > 0 or self.pad_w > 0):
            if self.pad_h > 0:
                # Reflect pad H: front=reversed slice from index 1, back=reversed slice from index -2
                front_h = x_BTHWC[:, :, 1 : self.pad_h + 1, :, :]
                back_h = x_BTHWC[:, :, -(self.pad_h + 1) : -1, :, :]
                x_BTHWC = ttnn.concat([front_h, x_BTHWC, back_h], dim=2)
            if self.pad_w > 0:
                # Reflect pad W
                front_w = x_BTHWC[:, :, :, 1 : self.pad_w + 1, :]
                back_w = x_BTHWC[:, :, :, -(self.pad_w + 1) : -1, :]
                x_BTHWC = ttnn.concat([front_w, x_BTHWC, back_w], dim=3)

        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
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
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = LTXPixelNorm()
        self.conv1 = LTXCausalConv3d(
            in_channels, out_channels, kernel_size=3, stride=1, mesh_device=mesh_device, dtype=dtype
        )

        self.norm2 = LTXPixelNorm()
        self.conv2 = LTXCausalConv3d(
            out_channels, out_channels, kernel_size=3, stride=1, mesh_device=mesh_device, dtype=dtype
        )

        self.has_shortcut = in_channels != out_channels
        if self.has_shortcut:
            # 1x1x1 conv for channel projection (no spatial/temporal change)
            self.conv_shortcut = LTXCausalConv3d(
                in_channels, out_channels, kernel_size=1, stride=1, mesh_device=mesh_device, dtype=dtype
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

    def forward(self, x_BTHWC: ttnn.Tensor, causal: bool = True) -> ttnn.Tensor:
        """
        Args:
            x_BTHWC: (B, T, H, W, C) in ROW_MAJOR layout
            causal: Temporal padding mode for convolutions

        Returns:
            (B, T, H, W, C_out) in ROW_MAJOR layout
        """
        residual = x_BTHWC

        # Main path: norm → silu → conv → norm → silu → conv
        h = self.norm1(x_BTHWC)
        h = ttnn.silu(h)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT) if h.layout != ttnn.ROW_MAJOR_LAYOUT else h
        h = self.conv1(h, causal=causal)

        h = self.norm2(h)
        h = ttnn.silu(h)
        h = ttnn.to_layout(h, ttnn.ROW_MAJOR_LAYOUT) if h.layout != ttnn.ROW_MAJOR_LAYOUT else h
        h = self.conv2(h, causal=causal)

        # Skip connection
        if self.has_shortcut:
            residual = ttnn.layer_norm(residual, weight=self.norm3_weight.data, bias=self.norm3_bias.data)
            residual = (
                ttnn.to_layout(residual, ttnn.ROW_MAJOR_LAYOUT)
                if residual.layout != ttnn.ROW_MAJOR_LAYOUT
                else residual
            )
            residual = self.conv_shortcut(residual, causal=causal)

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
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        self.res_blocks = ModuleList()
        for _ in range(num_layers):
            self.res_blocks.append(
                LTXResnetBlock3D(
                    in_channels=in_channels, out_channels=in_channels, mesh_device=mesh_device, dtype=dtype
                )
            )

    def forward(self, x_BTHWC: ttnn.Tensor, causal: bool = True) -> ttnn.Tensor:
        for block in self.res_blocks:
            x_BTHWC = block(x_BTHWC, causal=causal)
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
        spatial_padding_mode: str = "zeros",
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
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
            spatial_padding_mode=spatial_padding_mode,
            mesh_device=mesh_device,
            dtype=dtype,
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

    def forward(self, x_BTHWC: ttnn.Tensor, causal: bool = True) -> ttnn.Tensor:
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

        x_BTHWC = self.conv(x_BTHWC, causal=causal)

        # Depth-to-space on conv output
        x = self._depth_to_space_bthwc(x_BTHWC, B, T, H, W)

        # Remove first frame if temporal upsampling (causal padding artifact)
        if p1 == 2:
            x = x[:, 1:, :, :, :]

        if self.residual:
            x = ttnn.add(x, x_in)

        return x

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
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()
        import math

        self.stride = stride
        self.group_size = in_channels * math.prod(stride) // out_channels

        conv_out_channels = out_channels // math.prod(stride)
        self.conv = LTXCausalConv3d(
            in_channels, conv_out_channels, kernel_size=3, stride=1, mesh_device=mesh_device, dtype=dtype
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
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.causal = causal
        self.mesh_device = mesh_device
        out_channels_with_patch = out_channels * patch_size**2  # 3 * 16 = 48

        feature_channels = base_channels * 8  # 1024

        # Per-channel statistics (learned mean/std for denormalization)
        self.per_channel_mean = Parameter(total_shape=[1, in_channels], device=mesh_device, dtype=ttnn.float32)
        self.per_channel_std = Parameter(total_shape=[1, in_channels], device=mesh_device, dtype=ttnn.float32)

        # conv_in: 128 → 1024
        self.conv_in = LTXCausalConv3d(
            in_channels,
            feature_channels,
            kernel_size=3,
            stride=1,
            spatial_padding_mode="reflect",
            mesh_device=mesh_device,
            dtype=dtype,
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
                        spatial_padding_mode="reflect",
                        mesh_device=mesh_device,
                        dtype=dtype,
                    )
                )
                ch = new_ch
            elif block_name == "res_x_y":
                multiplier = block_config.get("multiplier", 2)
                new_ch = ch // multiplier
                self.up_blocks.append(
                    LTXResnetBlock3D(in_channels=ch, out_channels=new_ch, mesh_device=mesh_device, dtype=dtype)
                )
                ch = new_ch
            elif block_name == "res_x":
                num_layers = block_config.get("num_layers", 1)
                self.up_blocks.append(
                    LTXUNetMidBlock3D(in_channels=ch, num_layers=num_layers, mesh_device=mesh_device, dtype=dtype)
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
            spatial_padding_mode="reflect",
            mesh_device=mesh_device,
            dtype=dtype,
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
        B, C, T, H, W = sample_BCTHW.shape

        # Denormalize using per-channel statistics
        mean = self.per_channel_mean.data  # (1, C) on device
        std = self.per_channel_std.data

        # Convert to BTHWC and push to device
        sample = sample_BCTHW.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
        sample_tt = ttnn.from_torch(sample, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)

        # Denormalize: x = x * std + mean
        sample_tt = ttnn.add(ttnn.multiply(sample_tt, std), mean)
        sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)

        # conv_in
        sample_tt = self.conv_in(sample_tt, causal=self.causal)

        # Up blocks
        for up_block in self.up_blocks:
            sample_tt = up_block(sample_tt, causal=self.causal)

        # Output: PixelNorm → SiLU → conv_out
        sample_tt = self.norm_out(sample_tt)
        sample_tt = ttnn.silu(sample_tt)
        sample_tt = ttnn.to_layout(sample_tt, ttnn.ROW_MAJOR_LAYOUT)
        sample_tt = self.conv_out(sample_tt, causal=self.causal)

        # Convert back to host (take device 0's copy — output is replicated)
        result = ttnn.to_torch(ttnn.get_device_tensors(sample_tt)[0])  # (B, T_out, H_out, W_out, C_out)

        # Unpatchify: (B, T, H/4, W/4, 48) → (B, T, H, W, 3) via depth-to-space
        result = result.permute(0, 4, 1, 2, 3)  # (B, C, T, H, W) — 48 channels
        from einops import rearrange

        result = rearrange(
            result,
            "b (c p r q) f h w -> b c (f p) (h q) (w r)",
            p=1,
            q=self.patch_size,
            r=self.patch_size,
        )

        return result


# =============================================================================
# Torch-only VAE Decoder wrapper (for pipeline, runs on CPU)
# =============================================================================


class LTXVideoDecoderTorch:
    """
    Torch-only wrapper for the LTX-2 Video VAE decoder.

    Runs on CPU/GPU. Used by the pipeline to decode latents → video.
    TTNN implementation of the full decoder (with DepthToSpaceUpsample,
    UNetMidBlock3D, etc.) is future work.

    Usage:
        decoder = LTXVideoDecoderTorch.from_config(config_dict)
        video = decoder.decode(latent)  # (B, 128, F', H', W') → (B, 3, F, H, W)
    """

    def __init__(self, torch_decoder):
        self.decoder = torch_decoder
        self.decoder.eval()

    @classmethod
    def from_config(
        cls,
        decoder_blocks: list,
        *,
        in_channels: int = 128,
        out_channels: int = 3,
        patch_size: int = 4,
        causal: bool = False,
        timestep_conditioning: bool = False,
        base_channels: int = 128,
    ) -> "LTXVideoDecoderTorch":
        """Create decoder from block config."""
        import sys

        sys.path.insert(0, "LTX-2/packages/ltx-core/src")
        from ltx_core.model.video_vae.enums import NormLayerType
        from ltx_core.model.video_vae.video_vae import VideoDecoder

        decoder = VideoDecoder(
            convolution_dimensions=3,
            in_channels=in_channels,
            out_channels=out_channels,
            decoder_blocks=decoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType.PIXEL_NORM,
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            base_channels=base_channels,
        )
        return cls(decoder)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.decoder.load_state_dict(state_dict)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent tensor to video.

        Args:
            latent: (B, C, F', H', W') latent tensor (C=128)

        Returns:
            (B, 3, F, H, W) decoded video
        """
        return self.decoder(latent)


class LTXVideoEncoderTorch:
    """
    Torch-only wrapper for the LTX-2 Video VAE encoder.

    Runs on CPU/GPU. Used for image conditioning (i2v) or testing.

    Usage:
        encoder = LTXVideoEncoderTorch.from_config(encoder_blocks)
        latent = encoder.encode(video)  # (B, 3, F, H, W) → (B, 128, F', H', W')
    """

    def __init__(self, torch_encoder):
        self.encoder = torch_encoder
        self.encoder.eval()

    @classmethod
    def from_config(
        cls,
        encoder_blocks: list,
        *,
        in_channels: int = 3,
        out_channels: int = 128,
        patch_size: int = 4,
    ) -> "LTXVideoEncoderTorch":
        """Create encoder from block config."""
        import sys

        sys.path.insert(0, "LTX-2/packages/ltx-core/src")
        from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType
        from ltx_core.model.video_vae.video_vae import VideoEncoder

        encoder = VideoEncoder(
            convolution_dimensions=3,
            in_channels=in_channels,
            out_channels=out_channels,
            encoder_blocks=encoder_blocks,
            patch_size=patch_size,
            norm_layer=NormLayerType.PIXEL_NORM,
            latent_log_var=LogVarianceType.UNIFORM,
        )
        return cls(encoder)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.encoder.load_state_dict(state_dict)

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent.

        Args:
            video: (B, 3, F, H, W) video tensor

        Returns:
            (B, 128, F', H', W') latent tensor
        """
        return self.encoder(video)
