# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn

from .tt_conv2dWrapper import (
    TtConv2d,
    TtConv2dParameters,
    SliceConfig,
    SliceMode,
)
from .tt_maxpool2d_wrapper import TtMaxPool2d


class TtStem(nn.Module):
    """
    TTNN implementation of DeepLabStem.

    Based on the model structure, DeepLabStem contains:
    - conv1: Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    - conv2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    - conv3: Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    Each with SyncBatchNorm and ReLU activation.
    """

    def __init__(
        self,
        device: ttnn.MeshDevice,
        state_dict: dict[str, torch.Tensor],
        dtype: ttnn.DataType = ttnn.bfloat16,
        channel_slice_factor: int = 4,
    ):
        super().__init__()
        self.device = device
        self.channel_slice_factor = channel_slice_factor

        # Extract weights for conv1, conv2, conv3
        conv1_state = {k.replace("conv1.", ""): v for k, v in state_dict.items() if k.startswith("conv1.")}
        conv2_state = {k.replace("conv2.", ""): v for k, v in state_dict.items() if k.startswith("conv2.")}
        conv3_state = {k.replace("conv3.", ""): v for k, v in state_dict.items() if k.startswith("conv3.")}

        # Initialize conv layers with width slicing (4 slices)
        width_slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=4)

        self.conv1 = TtConv2d(
            TtConv2dParameters.from_torch(conv1_state, device=device, dtype=dtype, slice_config=width_slice_config),
            stride=(2, 2),
            padding=(1, 1),
        )

        self.conv2 = TtConv2d(
            TtConv2dParameters.from_torch(conv2_state, device=device, dtype=dtype, slice_config=width_slice_config),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.conv3 = TtConv2d(
            TtConv2dParameters.from_torch(conv3_state, device=device, dtype=dtype, slice_config=width_slice_config),
            stride=(1, 1),
            padding=(1, 1),
        )

        # Extract normalization parameters
        conv1_norm_state = {
            k.replace("conv1.norm.", ""): v for k, v in state_dict.items() if k.startswith("conv1.norm.")
        }
        conv2_norm_state = {
            k.replace("conv2.norm.", ""): v for k, v in state_dict.items() if k.startswith("conv2.norm.")
        }
        conv3_norm_state = {
            k.replace("conv3.norm.", ""): v for k, v in state_dict.items() if k.startswith("conv3.norm.")
        }

        # Convert normalization parameters to TTNN tensors
        mesh_mapper = ttnn.ReplicateTensorToMesh(device) if isinstance(device, ttnn.MeshDevice) else None

        # Conv1 normalization (64 channels)
        self.conv1_norm_weight = ttnn.from_torch(
            conv1_norm_state["weight"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv1_norm_bias = ttnn.from_torch(
            conv1_norm_state["bias"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv1_norm_running_mean = ttnn.from_torch(
            conv1_norm_state["running_mean"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv1_norm_running_var = ttnn.from_torch(
            conv1_norm_state["running_var"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )

        # Conv2 normalization (64 channels)
        self.conv2_norm_weight = ttnn.from_torch(
            conv2_norm_state["weight"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv2_norm_bias = ttnn.from_torch(
            conv2_norm_state["bias"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv2_norm_running_mean = ttnn.from_torch(
            conv2_norm_state["running_mean"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv2_norm_running_var = ttnn.from_torch(
            conv2_norm_state["running_var"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )

        # Conv3 normalization (128 channels)
        self.conv3_norm_weight = ttnn.from_torch(
            conv3_norm_state["weight"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv3_norm_bias = ttnn.from_torch(
            conv3_norm_state["bias"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv3_norm_running_mean = ttnn.from_torch(
            conv3_norm_state["running_mean"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )
        self.conv3_norm_running_var = ttnn.from_torch(
            conv3_norm_state["running_var"].view(1, -1, 1, 1),
            dtype=dtype,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
        )

        # Initialize maxpool with channel slicing
        self.maxpool = TtMaxPool2d.create_with_channel_slicing(
            device=device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), num_slices=self.channel_slice_factor
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Conv1 + BatchNorm + ReLU
        x = self.conv1(x)
        # Convert NHWC to NCHW for batch_norm
        x_permuted = ttnn.permute(x, (0, 3, 1, 2))
        ttnn.deallocate(x)
        x_normed = ttnn.batch_norm(
            x_permuted,
            running_mean=self.conv1_norm_running_mean,
            running_var=self.conv1_norm_running_var,
            weight=self.conv1_norm_weight,
            bias=self.conv1_norm_bias,
            eps=1e-05,
            training=False,
        )
        ttnn.deallocate(x_permuted)

        # Convert back to NHWC
        x_permuted = ttnn.permute(x_normed, (0, 2, 3, 1))
        ttnn.deallocate(x_normed)

        x_relued = ttnn.relu(x_permuted)
        ttnn.deallocate(x_permuted)

        # Conv2 + BatchNorm + ReLU
        x = self.conv2(x_relued)
        ttnn.deallocate(x_relued)
        # Convert NHWC to NCHW for batch_norm
        x_permuted = ttnn.permute(x, (0, 3, 1, 2))
        ttnn.deallocate(x)
        x_normed = ttnn.batch_norm(
            x_permuted,
            running_mean=self.conv2_norm_running_mean,
            running_var=self.conv2_norm_running_var,
            weight=self.conv2_norm_weight,
            bias=self.conv2_norm_bias,
            eps=1e-05,
            training=False,
        )
        ttnn.deallocate(x_permuted)
        # Convert back to NHWC
        x_permuted = ttnn.permute(x_normed, (0, 2, 3, 1))
        ttnn.deallocate(x_normed)
        x_relued = ttnn.relu(x_permuted)
        ttnn.deallocate(x_permuted)
        ttnn.move(x_relued)

        # Conv3 + BatchNorm + ReLU
        x = self.conv3(x_relued)
        ttnn.deallocate(x_relued)
        # Convert NHWC to NCHW for batch_norm
        x_permuted = ttnn.permute(x, (0, 3, 1, 2))
        ttnn.deallocate(x)
        x_normed = ttnn.batch_norm(
            x_permuted,
            running_mean=self.conv3_norm_running_mean,
            running_var=self.conv3_norm_running_var,
            weight=self.conv3_norm_weight,
            bias=self.conv3_norm_bias,
            eps=1e-05,
            training=False,
        )
        ttnn.deallocate(x_permuted)
        # Convert back to NHWC
        x_permuted = ttnn.permute(x_normed, (0, 2, 3, 1))
        ttnn.deallocate(x_normed)
        x_relued = ttnn.relu(x_permuted)
        ttnn.deallocate(x_permuted)

        # Max pooling with kernel_size=3, stride=2, padding=1
        x_pooled = self.maxpool(x_relued)
        ttnn.deallocate(x_relued)

        return x_pooled
