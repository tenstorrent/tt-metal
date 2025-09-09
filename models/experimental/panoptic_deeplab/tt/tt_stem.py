# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_conv2d_wrapper import (
    TtConv2d,
    TtConv2dParameters,
    SliceConfig,
    SliceMode,
)
from models.experimental.panoptic_deeplab.tt.tt_maxpool2d_wrapper import TtMaxPool2d


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
        parameters,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        channel_slice_factor: int = 4,
    ):
        super().__init__()
        self.device = device
        self.channel_slice_factor = channel_slice_factor

        logger.debug(f"Initializing TtStem with unified parameters, channel_slice_factor: {channel_slice_factor}")

        # Parameters are now organized as parameters["conv1"], parameters["conv2"], parameters["conv3"]
        conv1_params = parameters["conv1"]
        conv2_params = parameters["conv2"]
        conv3_params = parameters["conv3"]

        # Initialize conv layers with width slicing (4 slices)
        width_slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=4)

        self.conv1 = TtConv2d(
            TtConv2dParameters.from_preprocessed_parameters(
                conv1_params, device=device, dtype=dtype, slice_config=width_slice_config
            ),
            stride=(2, 2),
            padding=(1, 1),
        )

        self.conv2 = TtConv2d(
            TtConv2dParameters.from_preprocessed_parameters(
                conv2_params, device=device, dtype=dtype, slice_config=width_slice_config
            ),
            stride=(1, 1),
            padding=(1, 1),
        )

        self.conv3 = TtConv2d(
            TtConv2dParameters.from_preprocessed_parameters(
                conv3_params, device=device, dtype=dtype, slice_config=width_slice_config
            ),
            stride=(1, 1),
            padding=(1, 1),
        )

        # Extract normalization parameters from preprocessed structure
        conv1_norm_params = conv1_params.get("norm", None)
        conv2_norm_params = conv2_params.get("norm", None)
        conv3_norm_params = conv3_params.get("norm", None)

        # Normalization parameters are already TTNN tensors from preprocessing
        # Conv1 normalization (64 channels)
        if conv1_norm_params:
            self.conv1_norm_weight = conv1_norm_params["weight"]
            self.conv1_norm_bias = conv1_norm_params["bias"]
            self.conv1_norm_running_mean = conv1_norm_params["running_mean"]
            self.conv1_norm_running_var = conv1_norm_params["running_var"]
        else:
            # Default normalization if not available
            self.conv1_norm_weight = None
            self.conv1_norm_bias = None
            self.conv1_norm_running_mean = None
            self.conv1_norm_running_var = None

        # Conv2 normalization (64 channels)
        if conv2_norm_params:
            self.conv2_norm_weight = conv2_norm_params["weight"]
            self.conv2_norm_bias = conv2_norm_params["bias"]
            self.conv2_norm_running_mean = conv2_norm_params["running_mean"]
            self.conv2_norm_running_var = conv2_norm_params["running_var"]
        else:
            self.conv2_norm_weight = None
            self.conv2_norm_bias = None
            self.conv2_norm_running_mean = None
            self.conv2_norm_running_var = None

        # Conv3 normalization (128 channels)
        if conv3_norm_params:
            self.conv3_norm_weight = conv3_norm_params["weight"]
            self.conv3_norm_bias = conv3_norm_params["bias"]
            self.conv3_norm_running_mean = conv3_norm_params["running_mean"]
            self.conv3_norm_running_var = conv3_norm_params["running_var"]
        else:
            self.conv3_norm_weight = None
            self.conv3_norm_bias = None
            self.conv3_norm_running_mean = None
            self.conv3_norm_running_var = None

        # Initialize maxpool with channel slicing
        self.maxpool = TtMaxPool2d.create_with_channel_slicing(
            device=device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), num_slices=self.channel_slice_factor
        )
        logger.debug("TtStem initialization complete")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        logger.debug(f"TtStem forward pass starting - input shape: {x.shape}")

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

        logger.debug(f"TtStem forward pass complete - output shape: {x_pooled.shape}")
        return x_pooled
