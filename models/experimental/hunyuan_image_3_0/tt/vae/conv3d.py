# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Symmetric Conv3d for Hunyuan VAE (replicated mesh, Phase 1)."""

from __future__ import annotations

from collections.abc import Sequence

import torch

import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_dit.layers.audio_ops import prepare_conv3d_weight_state
from models.tt_dit.layers.module import Module, Parameter
from models.tt_dit.utils.conv3d import _ntuple, aligned_channels, get_conv3d_config, register_conv3d_configs

# Replicated 1x4 mesh @ 64x64 — conservative blocks (LTX-2.3 fallbacks).
register_conv3d_configs(
    {
        (1024, 1024, (3, 3, 3)): (64, 32, 1, 2, 2),
        (1024, 1024, (1, 1, 1)): (256, 32, 1, 1, 1),
    }
)

Z_CHANNELS = 32
BLOCK_IN_CHANNELS = 1024
LATENT_T = 1
LATENT_H = 64
LATENT_W = 64


class HunyuanSymmetricConv3d(Module):
    """Conv3d with symmetric padding on T, H, W. Input/output layout: BTHWC ROW_MAJOR."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Sequence[int] | int = 3,
        stride: Sequence[int] | int = 1,
        padding: Sequence[int] | int = 1,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        t: int = LATENT_T,
        h: int = LATENT_H,
        w: int = LATENT_W,
    ) -> None:
        super().__init__()

        self.unpadded_in_channels = in_channels
        self.in_channels = aligned_channels(in_channels)
        self.unpadded_out_channels = out_channels
        self.out_channels = out_channels

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.padding = _ntuple(padding, 3)
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
            T=t,
            H=h,
            W=w,
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if (is_blackhole() and dtype == ttnn.float32)
            else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        weight_elems = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(
            total_shape=[weight_elems, self.out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )
        self.bias = Parameter(
            total_shape=[1, self.out_channels],
            device=mesh_device,
            pad_value=0,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            prepare_conv3d_weight_state(
                state,
                state["weight"],
                conv_config=self.conv_config,
                mesh_device=self.mesh_device,
                dtype=self.dtype,
                unpadded_out=self.unpadded_out_channels,
                out_channels=self.out_channels,
                unpadded_in=self.unpadded_in_channels,
                in_channels=self.in_channels,
            )
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x_bthwc: ttnn.Tensor) -> ttnn.Tensor:
        assert (
            x_bthwc.layout == ttnn.ROW_MAJOR_LAYOUT
        ), f"HunyuanSymmetricConv3d expects ROW_MAJOR, got {x_bthwc.layout}"
        return ttnn.experimental.conv3d(
            input_tensor=x_bthwc,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self.mesh_device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
