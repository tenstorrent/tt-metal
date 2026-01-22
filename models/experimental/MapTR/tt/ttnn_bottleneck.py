# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0


import ttnn

from models.tt_cnn.tt.builder import (
    TtConv2d,
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
)


def create_conv_config(conv_args, conv_pth, activation=None, activation_dtype=ttnn.bfloat16):
    """Create Conv2dConfiguration from model args and weights."""
    return Conv2dConfiguration.from_model_args(
        conv2d_args=conv_args,
        weights=conv_pth.weight,
        bias=conv_pth.bias if hasattr(conv_pth, "bias") else None,
        activation=activation,
        sharding_strategy=AutoShardedStrategyConfiguration(),
        activation_dtype=activation_dtype,
    )


class TtBottleneck:
    """ResNet Bottleneck block"""

    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        is_downsample=False,
        activation_dtype=ttnn.bfloat16,
        **kwargs,
    ):
        """Initialize the Bottleneck block.

        Args:
            conv_args: Convolution arguments from infer_ttnn_module_args
            conv_pth: Weights from custom_preprocessor
            device: TTNN device
            is_downsample: Whether this block has a downsample path
            activation_dtype: Data type for activations (bfloat16 or bfloat8_b)
        """
        self.is_downsample = is_downsample
        self.activation_dtype = activation_dtype

        # Conv1: 1x1 with ReLU
        conv1_config = create_conv_config(
            conv_args.conv1,
            conv_pth.conv1,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )
        self.conv1 = TtConv2d(conv1_config, device)

        # Conv2: 3x3 with ReLU
        conv2_config = create_conv_config(
            conv_args.conv2,
            conv_pth.conv2,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        )
        self.conv2 = TtConv2d(conv2_config, device)

        # Conv3: 1x1 without activation
        conv3_config = create_conv_config(
            conv_args.conv3,
            conv_pth.conv3,
            activation=None,
        )
        self.conv3 = TtConv2d(conv3_config, device)

        if is_downsample:
            ds_config = create_conv_config(
                conv_args.downsample[0],
                conv_pth.downsample,
                activation=None,
                activation_dtype=activation_dtype,
            )
            self.downsample = TtConv2d(ds_config, device)

    def __call__(self, x_identity: ttnn.Tensor) -> ttnn.Tensor:
        """Execute the bottleneck block.

        Args:
            x_identity: Input tensor

        Returns:
            Output tensor after bottleneck processing
        """
        x = self.conv1(x_identity)

        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.conv2(x)
        x = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self.is_downsample:
            x_identity = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)

        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x
