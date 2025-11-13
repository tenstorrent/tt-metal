# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger

from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
from models.common.lightweightmodule import LightweightModule


class TtStem(LightweightModule):
    """
    TTNN implementation of DeepLabStem with fused Conv+BatchNorm using TT CNN Builder API.

    Architecture:
    - conv1: 3→64 channels, stride=2, 3x3 kernel
    - conv2: 64→64 channels, stride=1, 3x3 kernel
    - conv3: 64→128 channels, stride=1, 3x3 kernel
    - maxpool: 3x3 kernel, stride=2

    All conv layers include fused BatchNorm and ReLU activation.
    Uses TT CNN Builder's Conv2dConfiguration for layer configuration.
    """

    def __init__(
        self,
        parameters,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat8_b,
        model_configs=None,
    ):
        super().__init__()
        self.device = device
        self.model_configs = model_configs

        logger.debug(f"Initializing TtStem with TT CNN Builder API")

        # Extract layer parameters (Conv2dConfiguration or MaxPool2dConfiguration objects)
        conv1_params = parameters["conv1"]
        conv2_params = parameters["conv2"]
        conv3_params = parameters["conv3"]

        # Initialize conv layers using TT CNN Builder
        self.conv1, self.conv1_out_shape = self._create_conv_layer(conv1_params, "stem.conv1")
        self.conv2, self.conv2_out_shape = self._create_conv_layer(conv2_params, "stem.conv2")
        self.conv3, self.conv3_out_shape = self._create_conv_layer(conv3_params, "stem.conv3")

        # Apply model-specific overrides if model_configs is provided
        if self.model_configs is not None:
            maxpool_config = self.model_configs.apply_maxpool_overrides(
                parameters["maxpool"]["pool_config"], maxpool_path="stem.maxpool"
            )
            logger.debug(f"Applied config overrides for stem.maxpool")
        else:
            maxpool_config = parameters["maxpool"]["pool_config"]
            logger.debug(f"No model_configs provided for stem.maxpool, using base config")

        self.maxpool = TtMaxPool2d(maxpool_config, device)
        logger.debug("TtStem initialization complete")

    def _create_conv_layer(self, params, conv_path: str):
        """Helper method to create conv layers using TT CNN Builder with config overrides"""

        # Get base Conv2dConfiguration from preprocessing
        if "conv_config" in params:
            base_config = params["conv_config"]
            logger.debug(f"Using Conv2dConfiguration from preprocessing for {conv_path}")
        else:
            # Fallback: extract from old-style parameters (should not happen with new preprocessing)
            logger.warning(f"Conv2dConfiguration not found for {conv_path}, using fallback extraction")
            raise ValueError(
                f"Expected 'conv_config' in parameters for {conv_path}. Please use new preprocessing system."
            )

        # Apply model-specific overrides if model_configs is provided
        if self.model_configs is not None:
            final_config = self.model_configs.apply_conv_overrides(base_config, conv_path=conv_path)
            logger.debug(f"Applied config overrides for {conv_path}")
        else:
            final_config = base_config
            logger.debug(f"No model_configs provided for {conv_path}, using base config")

        # Compute output shape (NHWC format) based on conv2d formula
        kernel_h, kernel_w = final_config.kernel_size
        stride_h, stride_w = final_config.stride
        pad_h, pad_w = final_config.padding
        dil_h, dil_w = final_config.dilation

        out_h = (final_config.input_height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
        out_w = (final_config.input_width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
        output_shape = (final_config.batch_size, out_h, out_w, final_config.out_channels)

        # Create TtConv2d using TT CNN Builder
        conv_layer = TtConv2d(final_config, self.device)
        logger.debug(
            f"Created {conv_path} with TT CNN Builder - in={final_config.in_channels}, out={final_config.out_channels}, kernel={final_config.kernel_size}, out_shape={output_shape}"
        )

        # Debug print memory config
        shard_type = type(final_config.sharding_strategy).__name__ if final_config.sharding_strategy else "None"
        act_block_h = (
            final_config.sharding_strategy.act_block_h_override
            if hasattr(final_config.sharding_strategy, "act_block_h_override")
            else 0
        )
        logger.info(f"[MEMORY_CONFIG] {conv_path}: sharding={shard_type}, act_block_h_override={act_block_h}")

        return conv_layer, output_shape

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        logger.debug(f"TtStem forward - input: {x.shape}")

        assert x.storage_type() == ttnn.StorageType.DEVICE, "Input tensor must be on device"
        x = self.conv1(x)  # self._conv_relu_block(self.conv1, x, "Conv1", self.conv1_out_shape)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)  # next conv is sliced
        x = self.conv2(x)  # self._conv_relu_block(self.conv2, x, "Conv2", self.conv2_out_shape)
        x = self.conv3(x)
        x = self.maxpool(x)

        return x
