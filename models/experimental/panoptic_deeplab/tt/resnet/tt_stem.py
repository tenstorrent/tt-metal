# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import ttnn

from models.experimental.panoptic_deeplab.tt.tt_conv2dWrapper import TtConv2d, TtConv2dParameters


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

        # Initialize conv layers
        self.conv1 = TtConv2d(
            TtConv2dParameters.from_torch(conv1_state, device=device, dtype=dtype),
            stride=(2, 2),
            padding=(1, 1),
        )

        self.conv2 = TtConv2d(
            TtConv2dParameters.from_torch(conv2_state, device=device, dtype=dtype),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.conv3 = TtConv2d(
            TtConv2dParameters.from_torch(conv3_state, device=device, dtype=dtype),
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

    def _channel_sliced_max_pool2d(
        self,
        x: ttnn.Tensor,
        kernel_size: list[int],
        stride: list[int],
        padding: list[int],
        dilation: list[int] = [1, 1],
        ceil_mode: bool = False,
    ) -> ttnn.Tensor:
        """
        Performs max_pool2d with channel slicing when channels are divisible by slice factor.
        """
        batch_size, input_h, input_w, channels = x.shape

        # Check if channels are divisible by slice factor
        if channels % self.channel_slice_factor != 0:
            # No slicing possible, use regular max_pool2d
            x_reshaped = ttnn.reshape(x, (1, 1, batch_size * input_h * input_w, channels))
            x_pooled = ttnn.max_pool2d(
                x_reshaped,
                batch_size=batch_size,
                input_h=input_h,
                input_w=input_w,
                channels=channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )
            ttnn.deallocate(x_reshaped)
            # Calculate output dimensions
            output_h = (input_h + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
            output_w = (input_w + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
            x_pooled = ttnn.reshape(x_pooled, (batch_size, output_h, output_w, channels))
            return x_pooled

        # Channel slicing
        print(
            f"--- Running MaxPool2d with Channel Slicing (channels={channels}, slices={self.channel_slice_factor}) ---"
        )

        slice_channels = channels // self.channel_slice_factor

        # Calculate output dimensions
        output_h = (input_h + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        output_w = (input_w + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

        output_slices = []

        for i in range(self.channel_slice_factor):
            print(f"  Processing channel slice {i+1}/{self.channel_slice_factor}...")
            start_idx = i * slice_channels
            end_idx = (i + 1) * slice_channels

            # Slice input along channel dimension
            x_slice = ttnn.slice(x, [0, 0, 0, start_idx], [batch_size, input_h, input_w, end_idx])

            # Reshape for max_pool2d
            x_slice_reshaped = ttnn.reshape(x_slice, (1, 1, batch_size * input_h * input_w, slice_channels))
            ttnn.deallocate(x_slice)

            # Apply max_pool2d to slice
            x_slice_pooled = ttnn.max_pool2d(
                x_slice_reshaped,
                batch_size=batch_size,
                input_h=input_h,
                input_w=input_w,
                channels=slice_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )
            ttnn.deallocate(x_slice_reshaped)

            # Reshape back to output dimensions
            x_slice_output = ttnn.reshape(x_slice_pooled, (batch_size, output_h, output_w, slice_channels))
            ttnn.deallocate(x_slice_pooled)
            x_slice_output = ttnn.to_memory_config(x_slice_output, ttnn.DRAM_MEMORY_CONFIG)
            output_slices.append(x_slice_output)

        # Concatenate slices along channel dimension
        x_pooled = ttnn.concat(output_slices, dim=3)

        # Deallocate slice tensors
        for slice_tensor in output_slices:
            ttnn.deallocate(slice_tensor)

        return x_pooled

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Conv1 + BatchNorm + ReLU
        x = self.conv1(x, slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=4))
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
        x = self.conv2(x_relued, slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=4))
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
        x = self.conv3(x_relued, slice_config=ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dSliceWidth, num_slices=4))
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
        x_pooled = self._channel_sliced_max_pool2d(
            x_relued,
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[1, 1],
            dilation=[1, 1],
            ceil_mode=False,
        )
        ttnn.deallocate(x_relued)

        return x_pooled
