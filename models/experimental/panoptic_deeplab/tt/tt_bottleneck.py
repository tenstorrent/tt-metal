# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from dataclasses import replace

from models.tt_cnn.tt.builder import TtConv2d
from models.common.lightweightmodule import LightweightModule


class TtBottleneck(LightweightModule):
    """
    TTNN implementation of BottleneckBlock with fused Conv+BatchNorm using TT CNN Builder API.

    Based on the model structure, BottleneckBlock contains:
    - conv1: 1x1 conv (channel reduction) with bias
    - conv2: 3x3 conv (spatial convolution, potentially with stride/dilation) with bias
    - conv3: 1x1 conv (channel expansion) with bias
    - shortcut: optional 1x1 conv for residual connection when input/output dimensions differ

    Each with ReLU activation. BatchNorm operations are fused into the Conv weights and biases.
    The final output uses residual addition + ReLU.

    Uses TT CNN Builder's Conv2dConfiguration for layer configuration.
    """

    def __init__(
        self,
        parameters,
        device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat8_b,
        has_shortcut: bool = False,
        stride: int = 1,
        dilation: int = 1,
        shortcut_stride: int = 1,
        block_id: str = "unknown",
        model_configs=None,
    ):
        super().__init__()
        self.device = device
        self.has_shortcut = has_shortcut
        self.block_id = block_id
        self.model_configs = model_configs

        logger.debug(f"Initializing TtBottleneck {block_id} with TT CNN Builder API")

        # Extract parameters (Conv2dConfiguration objects)
        conv1_params = parameters["conv1"]
        conv2_params = parameters["conv2"]
        conv3_params = parameters["conv3"]

        # Initialize conv layers using TT CNN Builder
        # Note: stride and dilation may need to be adjusted for conv2
        self.conv1, self.conv1_out_shape = self._create_conv_layer(
            conv1_params, f"{block_id}.conv1", override_stride=(1, 1), override_padding=(0, 0)
        )

        # Conv2 needs to handle stride and dilation overrides
        self.conv2, self.conv2_out_shape = self._create_conv_layer(
            conv2_params,
            f"{block_id}.conv2",
            override_stride=(stride, stride),
            override_dilation=(dilation, dilation),
            override_padding=(dilation, dilation),  # Padding matches dilation for 3x3 conv
        )

        self.conv3, self.conv3_out_shape = self._create_conv_layer(
            conv3_params, f"{block_id}.conv3", override_stride=(1, 1), override_padding=(0, 0)
        )

        # Initialize shortcut if needed
        if has_shortcut:
            shortcut_params = parameters["shortcut"]
            self.shortcut, self.shortcut_out_shape = self._create_conv_layer(
                shortcut_params,
                f"{block_id}.shortcut",
                override_stride=(shortcut_stride, shortcut_stride),
                override_padding=(0, 0),
            )

        logger.debug(f"TtBottleneck {block_id} initialization complete")

    def _create_conv_layer(
        self, params, conv_path: str, override_stride=None, override_dilation=None, override_padding=None
    ):
        """Helper method to create conv layers using TT CNN Builder with config overrides"""

        # Get base Conv2dConfiguration from preprocessing
        if "conv_config" in params:
            base_config = params["conv_config"]
            logger.debug(f"Using Conv2dConfiguration from preprocessing for {conv_path}")
        else:
            # Fallback: should not happen with new preprocessing
            logger.error(f"Conv2dConfiguration not found for {conv_path}")
            raise ValueError(
                f"Expected 'conv_config' in parameters for {conv_path}. Please use new preprocessing system."
            )

        # Apply architectural overrides (stride, dilation, padding) if specified
        if override_stride is not None or override_dilation is not None or override_padding is not None:
            config_overrides = {}
            if override_stride is not None:
                config_overrides["stride"] = override_stride
            if override_dilation is not None:
                config_overrides["dilation"] = override_dilation
            if override_padding is not None:
                config_overrides["padding"] = override_padding

            base_config = replace(base_config, **config_overrides)
            logger.debug(f"Applied architectural overrides for {conv_path}: {config_overrides}")

        # Apply model-specific overrides from model_configs
        if self.model_configs is not None:
            final_config = self.model_configs.apply_conv_overrides(base_config, conv_path=conv_path)
            logger.debug(f"Applied model config overrides for {conv_path}")
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
            f"Created {conv_path} - in={final_config.in_channels}, out={final_config.out_channels}, stride={final_config.stride}, dilation={final_config.dilation}, out_shape={output_shape}"
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
        logger.debug(f"TtBottleneck {self.block_id} forward pass starting, input shape: {x.shape}")

        # Store input for residual connection
        # For blocks without shortcuts, we must create a deep copy of the input because conv1 will
        # deallocate x (deallocate_activation=True), and if identity is just a reference to x,
        # it will also become deallocated, causing PCC errors when we try to add it later.
        # For blocks with shortcuts (res2.0, res3.0, res4.0, res5.0), conv1 has deallocate_activation=False
        # configured in model_configs, so identity can safely reference x (shortcut will replace it anyway).
        if not self.has_shortcut or "res4" in self.block_id or "res5" in self.block_id:
            # All blocks without shortcuts need a deep copy
            # Efficient strategy based on current memory location:
            # - If already in DRAM: Use ttnn.clone() to create a copy
            # - If sharded or L1: Move to DRAM (this creates a new allocation)
            if x.memory_config().buffer_type == ttnn.BufferType.DRAM:
                # Already in DRAM: clone works and is efficient
                identity = ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            else:
                # Sharded or L1 tensor: moving to DRAM creates new allocation
                identity = ttnn.to_memory_config(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Blocks with shortcuts: conv1 won't deallocate x, so identity can reference it
            # (it will be replaced by shortcut output anyway)
            identity = x

        # Process shortcut if needed (BatchNorm is now fused into shortcut Conv)
        if self.has_shortcut:
            logger.debug(f"TtBottleneck {self.block_id} processing shortcut convolution")
            identity = self.shortcut(identity)
            if "res3" in self.block_id:
                identity = ttnn.to_memory_config(identity, ttnn.DRAM_MEMORY_CONFIG)
            else:
                # Only move if not already in DRAM
                if identity.memory_config().buffer_type != ttnn.BufferType.DRAM:
                    identity = ttnn.move(identity)
            logger.debug(f"TtBottleneck {self.block_id} shortcut processing complete, shape: {identity.shape}")

        # Main path: Conv1 + separate ReLU (BatchNorm fused into Conv1)
        logger.debug(f"TtBottleneck {self.block_id} processing conv1 (1x1 reduction)")
        out = self.conv1(x)
        # TT CNN returns flattened [B, 1, H*W, C], reshape to pre-computed output shape
        if out.memory_config().buffer_type != ttnn.BufferType.DRAM:
            out = ttnn.move(out)
        logger.debug(f"TtBottleneck {self.block_id} conv1 complete, output shape: {out.shape}")

        # dilated convs in res5 experience huge circular buffers if input is L1 sharded
        if "res5" in self.block_id:
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)

        # Conv2 + separate ReLU (BatchNorm fused into Conv2)
        logger.debug(f"TtBottleneck {self.block_id} processing conv2 (3x3 spatial)")
        out = self.conv2(out)
        # TT CNN returns flattened [B, 1, H*W, C], reshape to pre-computed output shape
        if out.memory_config().buffer_type != ttnn.BufferType.DRAM:
            out = ttnn.move(out)
        logger.debug(f"TtBottleneck {self.block_id} conv2 complete, output shape: {out.shape}")

        # Conv3 (no ReLU yet, BatchNorm fused into Conv3)
        logger.debug(f"TtBottleneck {self.block_id} processing conv3 (1x1 expansion)")
        out = self.conv3(out)
        # TT CNN returns flattened [B, 1, H*W, C], reshape to pre-computed output shape
        if out.memory_config().buffer_type != ttnn.BufferType.DRAM:
            out = ttnn.move(out)
        logger.debug(f"TtBottleneck {self.block_id} conv3 complete, output shape: {out.shape}")

        # Residual connection + ReLU
        logger.debug(f"TtBottleneck {self.block_id} adding residual connection and applying final ReLU")
        # Reshard identity to match output memory config when shortcuts use BlockSharded strategy
        # This is needed for res3.0 and res4.0 where shortcut uses BlockSharded but conv3 outputs HeightSharded
        if "res4.0" in self.block_id:
            identity = ttnn.reshard(identity, out.memory_config())
        # Always add residual - shapes should match after reshape
        out = ttnn.add(out, identity)
        out = ttnn.relu(out)
        if out.memory_config().buffer_type != ttnn.BufferType.DRAM:
            out = ttnn.move(out)

        logger.debug(f"TtBottleneck {self.block_id} forward pass complete, final output shape: {out.shape}")
        return out
