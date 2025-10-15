# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger

from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d, ChannelSliceStrategyConfiguration
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
        channel_slice_factor: int = 4,
        model_configs=None,
    ):
        super().__init__()
        self.device = device
        self.channel_slice_factor = channel_slice_factor
        self.model_configs = model_configs

        logger.debug(f"Initializing TtStem with TT CNN Builder API - channel_slice_factor: {channel_slice_factor}")

        # Extract layer parameters (Conv2dConfiguration or MaxPool2dConfiguration objects)
        conv1_params = parameters["conv1"]
        conv2_params = parameters["conv2"]
        conv3_params = parameters["conv3"]
        maxpool_params = parameters.get("maxpool", None)

        # Initialize conv layers using TT CNN Builder
        self.conv1, self.conv1_out_shape = self._create_conv_layer(conv1_params, "stem.conv1")
        self.conv2, self.conv2_out_shape = self._create_conv_layer(conv2_params, "stem.conv2")
        self.conv3, self.conv3_out_shape = self._create_conv_layer(conv3_params, "stem.conv3")

        # Initialize maxpool using TT CNN Builder
        if maxpool_params and "pool_config" in maxpool_params:
            # Use MaxPool2dConfiguration from preprocessing
            pool_config = maxpool_params["pool_config"]

            # Apply channel slicing override if needed
            if self.channel_slice_factor > 1:
                from dataclasses import replace

                pool_config = replace(
                    pool_config, slice_strategy=ChannelSliceStrategyConfiguration(num_slices=self.channel_slice_factor)
                )

            self.maxpool = TtMaxPool2d(pool_config, device)
            # Compute maxpool output shape
            kernel_h, kernel_w = pool_config.kernel_size
            stride_h, stride_w = pool_config.stride
            pad_h, pad_w = pool_config.padding
            dil_h, dil_w = pool_config.dilation
            out_h = (pool_config.input_height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
            out_w = (pool_config.input_width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
            self.maxpool_out_shape = (pool_config.batch_size, out_h, out_w, pool_config.channels)
            logger.debug(
                f"MaxPool initialized with TT CNN Builder - config: {pool_config}, out_shape={self.maxpool_out_shape}"
            )
        else:
            # Fallback: create maxpool config based on conv3 output shape
            # MaxPool expects the UNRESHAPED (flattened) input from TT CNN Builder
            logger.warning(
                "MaxPool config not found (PyTorch uses functional maxpool), creating from conv3 output shape"
            )
            from models.tt_cnn.tt.builder import MaxPool2dConfiguration

            # TT CNN conv outputs flattened [B, 1, H*W, C], so use the logical NHWC shape for maxpool config
            _, conv3_out_h, conv3_out_w, conv3_out_c = self.conv3_out_shape
            pool_config = MaxPool2dConfiguration(
                input_height=conv3_out_h,
                input_width=conv3_out_w,
                channels=conv3_out_c,
                batch_size=1,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                slice_strategy=ChannelSliceStrategyConfiguration(num_slices=self.channel_slice_factor)
                if self.channel_slice_factor > 1
                else None,
            )
            self.maxpool = TtMaxPool2d(pool_config, device)
            # Compute maxpool output shape
            kernel_h, kernel_w = pool_config.kernel_size
            stride_h, stride_w = pool_config.stride
            pad_h, pad_w = pool_config.padding
            dil_h, dil_w = pool_config.dilation
            out_h = (pool_config.input_height + 2 * pad_h - dil_h * (kernel_h - 1) - 1) // stride_h + 1
            out_w = (pool_config.input_width + 2 * pad_w - dil_w * (kernel_w - 1) - 1) // stride_w + 1
            self.maxpool_out_shape = (pool_config.batch_size, out_h, out_w, pool_config.channels)

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
        # out_h = floor((in_h + 2*pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1)
        # out_w = floor((in_w + 2*pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1)
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

        # Conv1 + separate ReLU + Reshape
        x = self._conv_relu_block(self.conv1, x, "Conv1", self.conv1_out_shape)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)  # next conv is sliced

        # Conv2 + separate ReLU + Reshape
        x = self._conv_relu_block(self.conv2, x, "Conv2", self.conv2_out_shape)

        # Conv3 + separate ReLU (NO reshape before maxpool - maxpool needs flattened input)
        x = self.conv3(x)
        logger.debug(f"Conv3 - raw conv output shape: {x.shape}, expected: {self.conv3_out_shape}")
        # Don't reshape yet - pass flattened to maxpool
        # x = ttnn.relu(x)  # Separate ReLU
        x = ttnn.move(x)  # Keep in current memory config
        logger.debug(f"Conv3 + separate ReLU - output: {x.shape}")

        # MaxPool (takes flattened input, returns flattened output)
        # Maxpool uses channel slicing, so output goes to DRAM
        x = self.maxpool(x)
        logger.debug(f"MaxPool raw output: {x.shape}, expected: {self.maxpool_out_shape}")
        if list(x.shape) != list(self.maxpool_out_shape):
            logger.debug(f"Reshaping maxpool from {x.shape} to {self.maxpool_out_shape}")
            x = ttnn.reshape(x, self.maxpool_out_shape)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        logger.debug(f"TtStem complete - output: {x.shape}")

        return x

    def _conv_relu_block(self, conv_layer, x: ttnn.Tensor, layer_name: str, output_shape) -> ttnn.Tensor:
        """Helper method for conv + separate relu operations"""
        x = conv_layer(x)
        logger.debug(f"{layer_name} - raw conv output shape: {x.shape}, expected: {output_shape}")
        # TT CNN Builder returns flattened [B, 1, H*W, C], reshape to [B, H, W, C]
        if list(x.shape) != list(output_shape):
            logger.debug(f"Reshaping {layer_name} from {x.shape} to {output_shape}")
            # x = ttnn.reshape(x, output_shape)
        # x = ttnn.relu(x)  # Separate ReLU
        x = ttnn.move(x)  # Keep in current memory config
        logger.debug(f"{layer_name} + separate ReLU - output: {x.shape}")
        return x
