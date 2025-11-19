# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    TtConv2d as TtConv2dBuilder,
    Conv2dConfiguration,
    HeightShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    AutoShardedStrategyConfiguration,
)


class TtConv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        activation=None,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_blk=False,
        dealloc_act=False,
        act_block_h=None,
    ):
        self.device = device
        self.conv_args = conv

        # Map VAD-v2 sharding to Builder API Strategy
        if is_blk:
            sharding_strategy = BlockShardedStrategyConfiguration(
                reshard_if_not_optimal=True, act_block_h_override=act_block_h if act_block_h else 0
            )
        elif shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            sharding_strategy = HeightShardedStrategyConfiguration(
                reshard_if_not_optimal=True, act_block_h_override=act_block_h if act_block_h else 0
            )
        else:
            sharding_strategy = AutoShardedStrategyConfiguration()

        # Prepare Weights and Bias (convert to TTNN if not already)
        weight = conv_pth.weight
        bias = conv_pth.bias

        # Builder API expects weights in specific format, we might need to adapt
        # But TtConv2dBuilder handles ttnn.Tensor inputs if configured correctly

        # Construct Configuration
        self.config = Conv2dConfiguration(
            input_height=conv.input_height,
            input_width=conv.input_width,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            batch_size=conv.batch_size,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            dilation=(1, 1),  # VAD-v2 seems to use default dilation
            weight=weight,
            bias=bias,
            activation=activation,
            weights_dtype=weights_dtype,
            activation_dtype=activation_dtype,
            sharding_strategy=sharding_strategy,
            deallocate_activation=dealloc_act,
            # math_fidelity=ttnn.MathFidelity.LoFi, # Builder defaults to LoFi
        )

        # Initialize Builder Conv
        self.builder_conv = TtConv2dBuilder(self.config, device)

    def __call__(self, x):
        # VAD-v2 inputs are [Batch, Channels, Height, Width] (NCHW)
        # Builder API/ttnn.conv2d for 1x1 convs expects [Batch, Height, Width, Channels] (NHWC)
        # or at least fails on NCHW with batch > 1

        # Check if we need to permute input
        needs_permutation = (
            self.config.kernel_size == (1, 1)
            and self.config.batch_size > 1
            and x.shape[1] == self.config.in_channels  # Confirm Channels is dim 1
        )

        if needs_permutation:
            # Permute NCHW -> NHWC (0, 2, 3, 1)
            # Note: ttnn.permute might be costly
            x = ttnn.permute(x, (0, 2, 3, 1))

        # Call Builder Conv
        x = self.builder_conv(x)

        # If we permuted input, output is likely NHWC, need to permute back to NCHW
        # to satisfy downstream VAD-v2 layers expecting NCHW
        if needs_permutation:
            x = ttnn.permute(x, (0, 3, 1, 2))  # NHWC -> NCHW

        # TtConv2D (Wrapper) contract returns: tensor, output_height, output_width
        output_height = self.config.input_height // self.config.stride[0]
        output_width = self.config.input_width // self.config.stride[1]

        return x, output_height, output_width
