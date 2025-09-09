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
    TTNN implementation of DeepLabStem with fused Conv+BatchNorm.

    Based on the model structure, DeepLabStem contains:
    - conv1: Conv2d(3, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=True)
    - conv2: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    - conv3: Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
    Each with ReLU activation. BatchNorm operations are fused into the Conv weights and biases.
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

        # With fused Conv+BN, we no longer need separate normalization parameters
        # All BatchNorm operations are now fused into the Conv weights and biases

        # Initialize maxpool with channel slicing
        self.maxpool = TtMaxPool2d.create_with_channel_slicing(
            device=device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), num_slices=self.channel_slice_factor
        )
        logger.debug("TtStem initialization complete")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        logger.debug(f"TtStem forward pass starting - input shape: {x.shape}")

        # Conv1 + ReLU (BatchNorm is now fused into Conv1 weights)
        x = self.conv1(x)
        x = ttnn.relu(x)
        logger.debug(f"Conv1 + ReLU complete, output shape: {x.shape}")

        # Conv2 + ReLU (BatchNorm is now fused into Conv2 weights)
        x = self.conv2(x)
        x = ttnn.relu(x)
        logger.debug(f"Conv2 + ReLU complete, output shape: {x.shape}")

        # Conv3 + ReLU (BatchNorm is now fused into Conv3 weights)
        x = self.conv3(x)
        x = ttnn.relu(x)
        logger.debug(f"Conv3 + ReLU complete, output shape: {x.shape}")

        # Max pooling with kernel_size=3, stride=2, padding=1
        x = self.maxpool(x)
        logger.debug(f"TtStem forward pass complete - output shape: {x.shape}")
        return x
