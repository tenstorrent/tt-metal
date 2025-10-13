# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_conv2d_wrapper import (
    TtConv2d,
    TtConv2dParameters,
    SliceConfig,
    SliceMode,
)
from models.experimental.panoptic_deeplab.tt.tt_maxpool2d_wrapper import TtMaxPool2d
from models.common.lightweightmodule import LightweightModule


class TtStem(LightweightModule):
    """
    TTNN implementation of DeepLabStem with fused Conv+BatchNorm.

    Architecture:
    - conv1: 3→64 channels, stride=2, 3x3 kernel
    - conv2: 64→64 channels, stride=1, 3x3 kernel
    - conv3: 64→128 channels, stride=1, 3x3 kernel
    - maxpool: 3x3 kernel, stride=2

    All conv layers include fused BatchNorm and ReLU activation.
    """

    def __init__(
        self,
        parameters,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
        channel_slice_factor: int = 4,
        model_configs=None,
    ):
        super().__init__()
        self.device = device
        self.channel_slice_factor = channel_slice_factor
        self.model_configs = model_configs

        logger.debug(f"Initializing TtStem - channel_slice_factor: {channel_slice_factor}")

        # Extract layer parameters
        conv1_params = parameters["conv1"]
        conv2_params = parameters["conv2"]
        conv3_params = parameters["conv3"]

        # Configure width slicing for all conv layers using model_configs
        if self.model_configs is not None:
            stem_slice_config = self.model_configs.get_slice_config("stem.conv1")
            width_slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=stem_slice_config["num_slices"])
        else:
            logger.warning(
                "FALLBACK STEM SLICE CONFIG: Using default width slicing with num_slices=4 instead of model_configs"
            )
            width_slice_config = SliceConfig(mode=SliceMode.WIDTH, num_slices=4)

        # Initialize conv layers
        self.conv1 = self._create_conv_layer(
            conv1_params, device, dtype, width_slice_config, stride=(2, 2), conv_path="stem.conv1"
        )
        self.conv2 = self._create_conv_layer(
            conv2_params, device, dtype, width_slice_config, stride=(1, 1), conv_path="stem.conv2"
        )
        self.conv3 = self._create_conv_layer(
            conv3_params, device, dtype, width_slice_config, stride=(1, 1), conv_path="stem.conv3"
        )

        # Initialize maxpool with channel slicing
        self.maxpool = TtMaxPool2d.create_with_channel_slicing(
            device=device, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), num_slices=self.channel_slice_factor
        )
        logger.debug("TtStem initialization complete")

    def _create_conv_layer(self, params, device, dtype, slice_config, stride, conv_path):
        """Helper method to create conv layers with consistent configuration"""
        return TtConv2d(
            TtConv2dParameters.from_preprocessed_parameters(
                params, device=device, dtype=dtype, slice_config=slice_config
            ),
            stride=stride,
            padding=(1, 1),
            conv_path=conv_path,
            model_configs=self.model_configs,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        logger.debug(f"TtStem forward - input: {x.shape}")

        # Conv1 + ReLU
        x = self._conv_relu_block(self.conv1, x, "Conv1")

        # Conv2 + ReLU
        x = self._conv_relu_block(self.conv2, x, "Conv2")

        # Conv3 + ReLU
        x = self._conv_relu_block(self.conv3, x, "Conv3")

        # MaxPool
        x = self.maxpool(x)
        logger.debug(f"TtStem complete - output: {x.shape}")

        return x

    def _conv_relu_block(self, conv_layer, x: ttnn.Tensor, layer_name: str) -> ttnn.Tensor:
        """Helper method for conv + relu operations"""
        x = conv_layer(x)
        x = ttnn.relu(x)
        logger.debug(f"{layer_name} + ReLU - output: {x.shape}")
        return x
