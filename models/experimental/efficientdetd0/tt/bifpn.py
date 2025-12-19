# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
#
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import Tuple
from models.experimental.efficientdetd0.tt.utils import (
    UpsampleConfiguration,
    TtSeparableConvBlock,
    TtMaxPool2dDynamicSamePadding,
    TtConv2dDynamicSamePadding,
    generate_conv_configuration_from_args,
    generate_maxpool_configuration_from_args,
    generate_upsample_configuration_from_args,
)
from models.tt_cnn.tt.builder import (
    TtUpsample,
    HeightShardedStrategyConfiguration,
)


def compute_fast_attention_weights(weight, epsilon, three_weights):
    # Convert to interleaved if sharded
    w_relu = ttnn.relu(weight)
    w_sum = ttnn.sum(w_relu, dim=0)
    denominator = ttnn.add(w_sum, epsilon)
    weight_0 = ttnn.div(w_relu[0], denominator)
    weight_1 = ttnn.div(w_relu[1], denominator)
    if three_weights:
        weight_2 = ttnn.div(w_relu[2], denominator)
    ttnn.deallocate(w_relu)
    ttnn.deallocate(w_sum)
    ttnn.deallocate(denominator)
    if three_weights:
        return weight_0, weight_1, weight_2
    else:
        return weight_0, weight_1


class TtBiFPN:
    """
    TTNN implementation of BiFPN (Bi-directional Feature Pyramid Network)
    """

    def __init__(
        self,
        device,
        parameters,
        module_args,
        num_channels: int,
        first_time: bool = False,
        epsilon: float = 1e-4,
        attention: bool = True,
        use_p8: bool = False,
        sharding_strategy=HeightShardedStrategyConfiguration(reshard_if_not_optimal=True),
        deallocate_activation: bool = False,
    ):
        """
        Args:
            device: TTNN device
            parameters: Model parameters containing weights for all conv layers
            module_args: Configuration parameters for all operations
            num_channels: Number of channels in BiFPN layers
            first_time: Whether input comes directly from backbone (requires channel reduction)
            epsilon: Small constant for numerical stability in weighted attention
            attention: Whether to use fast weighted attention
            use_p8: Whether to use P8 pyramid level
            sharding_strategy: Memory layout for sharding
            deallocate_activation: Whether to deallocate intermediate activations
        """
        self.device = device
        self.epsilon = epsilon
        self.attention = attention
        self.first_time = first_time
        self.sharding_strategy = sharding_strategy

        # Initialize separable conv blocks for upsampling path
        self.conv6_up = TtSeparableConvBlock(
            device=device,
            parameters=parameters.conv6_up,
            module_args=module_args.conv6_up,
            sharding_strategy=sharding_strategy,
            deallocate_activation=True,
        )
        self.conv5_up = TtSeparableConvBlock(
            device=device,
            parameters=parameters.conv5_up,
            module_args=module_args.conv5_up,
            sharding_strategy=sharding_strategy,
            deallocate_activation=True,
        )
        self.conv4_up = TtSeparableConvBlock(
            device=device,
            parameters=parameters.conv4_up,
            module_args=module_args.conv4_up,
            sharding_strategy=sharding_strategy,
            deallocate_activation=True,
        )
        self.conv3_up = TtSeparableConvBlock(
            device=device,
            parameters=parameters.conv3_up,
            module_args=module_args.conv3_up,
            sharding_strategy=sharding_strategy,
            deallocate_activation=True,
        )

        # Initialize separable conv blocks for downsampling path
        self.conv4_down = TtSeparableConvBlock(
            device,
            parameters=parameters.conv4_down,
            module_args=module_args.conv4_down,
            sharding_strategy=sharding_strategy,
            deallocate_activation=True,
        )
        self.conv5_down = TtSeparableConvBlock(
            device,
            parameters=parameters.conv5_down,
            module_args=module_args.conv5_down,
            sharding_strategy=sharding_strategy,
            deallocate_activation=True,
        )
        self.conv6_down = TtSeparableConvBlock(
            device,
            parameters=parameters.conv6_down,
            module_args=module_args.conv6_down,
            sharding_strategy=sharding_strategy,
            deallocate_activation=True,
        )
        self.conv7_down = TtSeparableConvBlock(
            device,
            parameters=parameters.conv7_down,
            module_args=module_args.conv7_down,
            sharding_strategy=sharding_strategy,
            deallocate_activation=True,
        )

        # Initialize Upsample layers
        self.p6_upsample = TtUpsample(
            configuration=generate_upsample_configuration_from_args(module_args.p6_upsample),
            device=device,
        )
        self.p5_upsample = TtUpsample(
            configuration=generate_upsample_configuration_from_args(module_args.p5_upsample),
            device=device,
        )
        self.p4_upsample = TtUpsample(
            configuration=generate_upsample_configuration_from_args(module_args.p4_upsample),
            device=device,
        )
        self.p3_upsample = TtUpsample(
            configuration=generate_upsample_configuration_from_args(module_args.p3_upsample),
            device=device,
        )

        # Initialize maxpool layers for downsampling
        self.p4_downsample = TtMaxPool2dDynamicSamePadding(
            configuration=generate_maxpool_configuration_from_args(
                maxpool2d_args=module_args.p4_downsample,
                output_layout=ttnn.TILE_LAYOUT,
            ),
            device=device,
        )
        self.p5_downsample = TtMaxPool2dDynamicSamePadding(
            configuration=generate_maxpool_configuration_from_args(
                maxpool2d_args=module_args.p5_downsample,
                output_layout=ttnn.TILE_LAYOUT,
            ),
            device=device,
        )
        self.p6_downsample = TtMaxPool2dDynamicSamePadding(
            configuration=generate_maxpool_configuration_from_args(
                maxpool2d_args=module_args.p6_downsample,
                output_layout=ttnn.TILE_LAYOUT,
            ),
            device=device,
        )
        self.p7_downsample = TtMaxPool2dDynamicSamePadding(
            configuration=generate_maxpool_configuration_from_args(
                maxpool2d_args=module_args.p7_downsample,
                output_layout=ttnn.TILE_LAYOUT,
            ),
            device=device,
        )

        # Initialize channel reduction layers for first_time
        if self.first_time:
            self.p3_down_channel = TtConv2dDynamicSamePadding(
                configuration=generate_conv_configuration_from_args(
                    conv2d_args=module_args.p3_down_channel[0],
                    parameters_dict=parameters.p3_down_channel[0],
                    sharding_strategy=sharding_strategy,
                    deallocate_activation=deallocate_activation,
                ),
                device=device,
            )
            self.p4_down_channel = TtConv2dDynamicSamePadding(
                configuration=generate_conv_configuration_from_args(
                    conv2d_args=module_args.p4_down_channel[0],
                    parameters_dict=parameters.p4_down_channel[0],
                    sharding_strategy=sharding_strategy,
                    deallocate_activation=False,
                ),
                device=device,
            )
            self.p5_down_channel = TtConv2dDynamicSamePadding(
                configuration=generate_conv_configuration_from_args(
                    conv2d_args=module_args.p5_down_channel[0],
                    parameters_dict=parameters.p5_down_channel[0],
                    sharding_strategy=sharding_strategy,
                    deallocate_activation=False,
                ),
                device=device,
            )

            # P5 to P6 conversion (conv + maxpool)
            self.p5_to_p6_conv = TtConv2dDynamicSamePadding(
                configuration=generate_conv_configuration_from_args(
                    conv2d_args=module_args.p5_to_p6[0],
                    parameters_dict=parameters.p5_to_p6[0],
                    sharding_strategy=sharding_strategy,
                    deallocate_activation=False,
                ),
                device=device,
            )
            self.p5_to_p6_pool = TtMaxPool2dDynamicSamePadding(
                configuration=generate_maxpool_configuration_from_args(
                    maxpool2d_args=module_args.p5_to_p6[2],
                    output_layout=ttnn.TILE_LAYOUT,
                ),
                device=device,
            )

            # P6 to P7 conversion (maxpool only)
            self.p6_to_p7 = TtMaxPool2dDynamicSamePadding(
                configuration=generate_maxpool_configuration_from_args(
                    maxpool2d_args=module_args.p6_to_p7[0],
                    output_layout=ttnn.TILE_LAYOUT,
                ),
                device=device,
            )

            # Additional channel reduction for bottom-up path
            self.p4_down_channel_2 = TtConv2dDynamicSamePadding(
                configuration=generate_conv_configuration_from_args(
                    conv2d_args=module_args.p4_down_channel_2[0],
                    parameters_dict=parameters.p4_down_channel_2[0],
                    sharding_strategy=sharding_strategy,
                    deallocate_activation=deallocate_activation,
                ),
                device=device,
            )
            self.p5_down_channel_2 = TtConv2dDynamicSamePadding(
                configuration=generate_conv_configuration_from_args(
                    conv2d_args=module_args.p5_down_channel_2[0],
                    parameters_dict=parameters.p5_down_channel_2[0],
                    sharding_strategy=sharding_strategy,
                    deallocate_activation=deallocate_activation,
                ),
                device=device,
            )

        # Store attention weights as TTNN tensors
        if attention:
            self.p6_w1_weight_0, self.p6_w1_weight_1 = compute_fast_attention_weights(
                parameters.p6_w1, epsilon=self.epsilon, three_weights=False
            )
            self.p5_w1_weight_0, self.p5_w1_weight_1 = compute_fast_attention_weights(
                parameters.p5_w1, epsilon=self.epsilon, three_weights=False
            )
            self.p4_w1_weight_0, self.p4_w1_weight_1 = compute_fast_attention_weights(
                parameters.p4_w1, epsilon=self.epsilon, three_weights=False
            )
            self.p3_w1_weight_0, self.p3_w1_weight_1 = compute_fast_attention_weights(
                parameters.p3_w1, epsilon=self.epsilon, three_weights=False
            )
            self.p4_w2_weight_0, self.p4_w2_weight_1, self.p4_w2_weight_2 = compute_fast_attention_weights(
                parameters.p4_w2, epsilon=self.epsilon, three_weights=True
            )
            self.p5_w2_weight_0, self.p5_w2_weight_1, self.p5_w2_weight_2 = compute_fast_attention_weights(
                parameters.p5_w2, epsilon=self.epsilon, three_weights=True
            )
            self.p6_w2_weight_0, self.p6_w2_weight_1, self.p6_w2_weight_2 = compute_fast_attention_weights(
                parameters.p6_w2, epsilon=self.epsilon, three_weights=True
            )
            self.p7_w2_weight_0, self.p7_w2_weight_1 = compute_fast_attention_weights(
                parameters.p7_w2, epsilon=self.epsilon, three_weights=False
            )

    @staticmethod
    def _pre_upsample_reshape(x, config: UpsampleConfiguration):
        """Convert sharded tensor [1,1,NHW,C] to [N,H,W,C] for Upsample layer."""
        # Convert to interleaved if sharded
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        # Convert to ROW_MAJOR layout for upsample
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Get the actual shape
        unflattened_shape = (config.batch_size, config.input_height, config.input_width, config.channels)
        x = ttnn.reshape(x, unflattened_shape)
        return x

    def _swish(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Swish activation: x * sigmoid(x)"""
        return x * ttnn.sigmoid_accurate(x, True)

    def __call__(self, inputs: Tuple[ttnn.Tensor, ...]) -> Tuple[ttnn.Tensor, ...]:
        """
        Forward pass through BiFPN

        Args:
            inputs: Tuple of feature maps (P3, P4, P5) for first_time=True
                   or (P3, P4, P5, P6, P7) for first_time=False

        Returns:
            Tuple of output feature maps (P3_out, P4_out, P5_out, P6_out, P7_out)
        """
        if self.attention:
            return self._forward_fast_attention(inputs)
        else:
            raise NotImplemented("Only _forward_fast_attention() is supported")

    def _forward_fast_attention(self, inputs: Tuple[ttnn.Tensor, ...]) -> Tuple[ttnn.Tensor, ...]:
        """Forward pass with fast weighted attention"""

        if self.first_time:
            p3, p4, p5 = inputs
            # Generate P6 and P7 from P5
            p6_in = self.p5_to_p6_conv(p5)
            p6_in = self.p5_to_p6_pool(p6_in)
            p7_in = self.p6_to_p7(p6_in)

            # Channel reduction for P3, P4, P5
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        p7_upsampled = self._pre_upsample_reshape(p7_in, self.p6_upsample.configuration)
        p7_upsampled = self.p6_upsample(p7_upsampled)

        p6_in = ttnn.to_memory_config(p6_in, ttnn.DRAM_MEMORY_CONFIG)
        p6_in = ttnn.reshape(p6_in, p7_upsampled.shape)
        p6_up = self.conv6_up(self._swish(self.p6_w1_weight_0 * p6_in + self.p6_w1_weight_1 * p7_upsampled))
        ttnn.deallocate(p7_upsampled)

        p6_upsampled = self._pre_upsample_reshape(p6_up, self.p5_upsample.configuration)
        p6_upsampled = self.p5_upsample(p6_upsampled)

        p5_in = ttnn.to_memory_config(p5_in, ttnn.DRAM_MEMORY_CONFIG)
        p5_in = ttnn.reshape(p5_in, p6_upsampled.shape)
        p5_up = self.conv5_up(self._swish(self.p5_w1_weight_0 * p5_in + self.p5_w1_weight_1 * p6_upsampled))
        ttnn.deallocate(p6_upsampled)

        p5_upsampled = self._pre_upsample_reshape(p5_up, self.p4_upsample.configuration)
        p5_upsampled = self.p4_upsample(p5_upsampled)

        p4_in = ttnn.to_memory_config(p4_in, ttnn.DRAM_MEMORY_CONFIG)
        p4_in = ttnn.reshape(p4_in, p5_upsampled.shape)
        p4_up = self.conv4_up(self._swish(self.p4_w1_weight_0 * p4_in + self.p4_w1_weight_1 * p5_upsampled))
        ttnn.deallocate(p5_upsampled)

        p4_upsampled = self._pre_upsample_reshape(p4_up, self.p3_upsample.configuration)
        p4_upsampled = self.p3_upsample(p4_upsampled)

        p3_in = ttnn.to_memory_config(p3_in, ttnn.DRAM_MEMORY_CONFIG)
        p3_in = ttnn.reshape(p3_in, p4_upsampled.shape)
        p3_out = self.conv3_up(self._swish(self.p3_w1_weight_0 * p3_in + self.p3_w1_weight_1 * p4_upsampled))
        ttnn.deallocate(p3_in)
        ttnn.deallocate(p4_upsampled)

        # Update p4_in and p5_in for bottom-up path if first_time
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p4_in = ttnn.to_memory_config(p4_in, ttnn.DRAM_MEMORY_CONFIG)
            p5_in = self.p5_down_channel_2(p5)
            p5_in = ttnn.to_memory_config(p5_in, ttnn.DRAM_MEMORY_CONFIG)

        # Bottom-up pathway with weighted attention
        p3_downsampled = self.p4_downsample(p3_out)
        p4_up = ttnn.reshape(p4_up, p4_in.shape)
        p3_downsampled = ttnn.reshape(p3_downsampled, p4_in.shape)
        p4_out = self.conv4_down(
            self._swish(
                self.p4_w2_weight_0 * p4_in + self.p4_w2_weight_1 * p4_up + self.p4_w2_weight_2 * p3_downsampled
            )
        )

        ttnn.deallocate(p4_in)
        ttnn.deallocate(p3_downsampled)

        p4_downsampled = self.p5_downsample(p4_out)
        p5_up = ttnn.to_memory_config(p5_up, ttnn.DRAM_MEMORY_CONFIG)
        p5_up = ttnn.reshape(p5_up, p5_in.shape)
        p4_downsampled = ttnn.reshape(p4_downsampled, p5_in.shape)
        p5_out = self.conv5_down(
            self._swish(
                self.p5_w2_weight_0 * p5_in + self.p5_w2_weight_1 * p5_up + self.p5_w2_weight_2 * p4_downsampled
            )
        )

        ttnn.deallocate(p5_in)
        ttnn.deallocate(p4_downsampled)

        # P6_2 = weighted_sum(P6_0, P6_1, downsample(P5_2))
        p5_downsampled = self.p6_downsample(p5_out)
        p6_up = ttnn.to_memory_config(p6_up, ttnn.DRAM_MEMORY_CONFIG)
        p6_up = ttnn.reshape(p6_up, p6_in.shape)
        p5_downsampled = ttnn.reshape(p5_downsampled, p6_in.shape)
        p6_out = self.conv6_down(
            self._swish(
                self.p6_w2_weight_0 * p6_in + self.p6_w2_weight_1 * p6_up + self.p6_w2_weight_2 * p5_downsampled
            )
        )
        ttnn.deallocate(p6_in)
        ttnn.deallocate(p5_downsampled)

        # P7_2 = weighted_sum(P7_0, downsample(P6_2))
        p6_downsampled = self.p7_downsample(p6_out)
        p6_downsampled = ttnn.to_memory_config(p6_downsampled, ttnn.DRAM_MEMORY_CONFIG)
        p6_downsampled = ttnn.reshape(p6_downsampled, p7_in.shape)
        p7_out = self.conv7_down(self._swish(self.p7_w2_weight_0 * p7_in + self.p7_w2_weight_1 * p6_downsampled))
        ttnn.deallocate(p7_in)
        ttnn.deallocate(p6_downsampled)

        return p3_out, p4_out, p5_out, p6_out, p7_out
