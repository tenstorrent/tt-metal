# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.tt_cnn.tt.builder import (
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    Conv2dConfiguration,
    DeviceDescriptor,
    HeightShardedStrategyConfiguration,
    TtConv2d,
    WidthShardedStrategyConfiguration,
)


class TtMobileNetV2Conv2D:
    """MobileNetV2 Conv2D layer using builder API internally"""

    def __init__(
        self,
        input_params,
        parameters,
        device,
        batch_size,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        reshard_if_not_optimal=True,
        activation_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        activation_function=None,
    ):
        self.input_params = input_params
        self.parameters = parameters
        self.device = device
        self.batch_size = batch_size
        self.groups = groups
        self.dilation = dilation
        self.act_block_h = act_block_h
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.activation_dtype = activation_dtype
        self.shard_layout = shard_layout
        self.activation_function = activation_function

        if self.block_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        if self.width_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        self.device_descriptor = DeviceDescriptor(device, (8, 8))

        self.conv_layer = None
        self.weights, self.bias = parameters

    def _get_sharding_strategy(self):
        """Convert original sharding logic to builder API strategy"""
        if self.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
            return BlockShardedStrategyConfiguration(reshard_if_not_optimal=self.reshard_if_not_optimal)
        elif self.shard_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
            return WidthShardedStrategyConfiguration(reshard_if_not_optimal=self.reshard_if_not_optimal)
        elif self.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            return HeightShardedStrategyConfiguration(
                reshard_if_not_optimal=self.reshard_if_not_optimal,
                act_block_h_override=self.act_blocks if self.act_block_h else 0,
            )
        else:
            return AutoShardedStrategyConfiguration()

    def _create_conv_configuration(self, input_height, input_width, in_channels):
        """Create builder API configuration matching original parameters exactly"""
        # Get sharding strategy
        sharding_strategy = self._get_sharding_strategy()

        # Create configuration that matches original exactly
        config = Conv2dConfiguration(
            input_height=input_height,
            input_width=input_width,
            in_channels=in_channels,
            out_channels=self.input_params[3],
            batch_size=self.batch_size,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            groups=self.groups,
            dilation=(self.dilation, self.dilation),
            weight=self.weights,
            bias=self.bias,
            sharding_strategy=sharding_strategy,
        )

        # Override default values to match original exactly
        object.__setattr__(
            config, "activation", self.activation_function if self.activation_function is not None else ""
        )
        object.__setattr__(config, "activation_dtype", self.activation_dtype)
        object.__setattr__(config, "weights_dtype", ttnn.bfloat8_b)
        object.__setattr__(config, "output_dtype", self.activation_dtype)
        object.__setattr__(config, "output_layout", self.output_layout)
        object.__setattr__(config, "math_fidelity", ttnn.MathFidelity.LoFi)
        object.__setattr__(config, "fp32_dest_acc_en", False)
        object.__setattr__(config, "packer_l1_acc", False)
        object.__setattr__(config, "enable_act_double_buffer", self.enable_act_double_buffer)
        object.__setattr__(config, "enable_weights_double_buffer", True)
        object.__setattr__(
            config,
            "enable_split_reader",
            True if self.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED else self.enable_split_reader,
        )
        object.__setattr__(config, "deallocate_activation", self.deallocate_activation)
        object.__setattr__(config, "reallocate_halo_output", False)

        return config

    def __call__(self, x):
        """Same interface as original - preserves exact functionality"""

        # Get input dimensions (same logic as original)
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
            in_channels = x.shape[3]
        else:
            input_height = int(math.sqrt((x.shape[2] // self.batch_size)))
            input_width = int(math.sqrt((x.shape[2] // self.batch_size)))
            in_channels = x.shape[3]

        # Create conv layer on first call using builder API
        if self.conv_layer is None:
            config = self._create_conv_configuration(input_height, input_width, in_channels)
            self.conv_layer = TtConv2d(config, self.device_descriptor)

        # Call builder API layer
        result = self.conv_layer(x)

        # Calculate output dimensions (preserve original logic)
        kernel_size, stride, padding = self.input_params[0], self.input_params[1], self.input_params[2]
        output_height = (input_height + 2 * padding - kernel_size) // stride + 1
        output_width = (input_width + 2 * padding - kernel_size) // stride + 1

        return result, output_height, output_width


class TtInvertedResidual:
    def __init__(
        self, model_params, device, batchsize, expand_ratio, stride, in_channels, out_channels, id, block_shard=False
    ):
        self.device = device
        self.batchsize = batchsize
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_shard = block_shard
        self.id = id
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.conv1 = None
        if expand_ratio != 1:
            # conv1: expansion layer (1x1 conv to expand channels)
            self.conv1 = TtMobileNetV2Conv2D(
                [1, 1, 0, hidden_dim],
                (model_params[f"fused_conv_{id * 3 - id}_weight"], model_params[f"fused_conv_{id * 3 - id}_bias"]),
                device,
                batchsize,
                block_shard=False if id == 6 and (11 < id <= 16) else self.block_shard,
                deallocate_activation=True if not self.use_res_connect else False,
                enable_act_double_buffer=True,
                activation_function="relu6",
            )

        # conv2: depthwise layer (3x3 depthwise conv)
        self.conv2 = TtMobileNetV2Conv2D(
            [3, stride, 1, hidden_dim],
            (model_params[f"fused_conv_{id * 3 -id +1}_weight"], model_params[f"fused_conv_{id * 3 - id + 1}_bias"]),
            device,
            batchsize,
            groups=hidden_dim,
            block_shard=self.block_shard,
            deallocate_activation=True,
            activation_function="relu6",
            enable_act_double_buffer=True if self.block_shard else False,
        )

        # conv3: projection layer (1x1 conv to project back to output channels)
        self.conv3 = TtMobileNetV2Conv2D(
            [1, 1, 0, out_channels],
            (model_params[f"conv_{id}_weight"], model_params[f"conv_{id}_bias"]),
            device,
            batchsize,
            block_shard=False if (10 <= id <= 16) else self.block_shard,
            deallocate_activation=True,
            enable_act_double_buffer=True,
        )

    def __call__(self, x):
        identity = x
        if self.conv1 is not None:
            x, _, _ = self.conv1(x)  # Builder returns (result, h, w) tuple
        out, _, _ = self.conv2(x)  # Builder returns (result, h, w) tuple
        out, _, _ = self.conv3(out)  # Builder returns (result, h, w) tuple
        if self.use_res_connect:
            if identity.memory_config() != out.memory_config():
                identity = ttnn.to_memory_config(identity, out.memory_config())
            tmp = ttnn.add(identity, out)
            ttnn.deallocate(identity)
            ttnn.deallocate(out)
            out = tmp
        return out
