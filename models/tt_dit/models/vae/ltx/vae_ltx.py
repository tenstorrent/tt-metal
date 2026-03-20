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

from ....layers.module import Module, Parameter
from ....utils.conv3d import _ntuple, aligned_channels, get_conv3d_config, prepare_conv3d_weights


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

        # Temporal padding: repeat first frame (kernel_t - 1) times
        self.time_pad = self.kernel_size[0] - 1

        # Spatial padding is handled by the conv3d op
        padding_h = self.kernel_size[1] // 2
        padding_w = self.kernel_size[2] // 2
        self.internal_padding = (0, padding_h, padding_w)

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

        if "weight" in state and "bias" in state:
            weight = state["weight"]
            bias = state["bias"]

            # Pad out_channels if needed
            if self.out_channels != self.unpadded_out_channels:
                weight = torch.nn.functional.pad(
                    weight, (0, 0, 0, 0, 0, 0, 0, 0, 0, self.out_channels - self.unpadded_out_channels)
                )
                bias = torch.nn.functional.pad(bias, (0, self.out_channels - self.unpadded_out_channels))

            state["weight"], state["bias"] = prepare_conv3d_weights(weight, bias, self.conv_config)

    def forward(self, x_BTHWC: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            x_BTHWC: (B, T, H, W, C) in ROW_MAJOR layout

        Returns:
            (B, T_out, H_out, W_out, C_out) in ROW_MAJOR layout
        """
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT

        # Causal temporal padding: repeat first frame
        if self.time_pad > 0:
            first_frame = x_BTHWC[:, :1, :, :, :]
            # Build padding by repeating first frame
            B, T, H, W, C = x_BTHWC.shape
            # Concat first_frame repeated time_pad times with the input
            padding_frames = []
            for _ in range(self.time_pad):
                padding_frames.append(first_frame)
            x_BTHWC = ttnn.concat([*padding_frames, x_BTHWC], dim=1)

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
