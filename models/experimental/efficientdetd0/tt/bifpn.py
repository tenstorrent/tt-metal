import ttnn
from typing import Tuple
from models.experimental.efficientdetd0.tt.utils import (
    SeparableConvBlock,
    MaxPool2dDynamicSamePadding,
    Conv2dDynamicSamePadding,
)


class TtBiFPN:
    """
    TTNN implementation of BiFPN (Bi-directional Feature Pyramid Network)
    """

    def __init__(
        self,
        device: ttnn.Device,
        parameters,
        conv_params,
        num_channels: int,
        first_time: bool = False,
        epsilon: float = 1e-4,
        attention: bool = True,
        use_p8: bool = False,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        deallocate_activation: bool = False,
    ):
        """
        Args:
            device: TTNN device
            parameters: Model parameters containing weights for all conv layers
            conv_params: Configuration parameters for all conv operations
            num_channels: Number of channels in BiFPN layers
            first_time: Whether input comes directly from backbone (requires channel reduction)
            epsilon: Small constant for numerical stability in weighted attention
            attention: Whether to use fast weighted attention
            use_p8: Whether to use P8 pyramid level
            shard_layout: Memory layout for sharding
            deallocate_activation: Whether to deallocate intermediate activations
        """
        self.device = device
        self.epsilon = epsilon
        self.attention = attention
        self.first_time = first_time
        self.shard_layout = shard_layout

        # Initialize separable conv blocks for upsampling path
        self.conv6_up = SeparableConvBlock(
            device, parameters.conv6_up, shard_layout, conv_params.conv6_up, deallocate_activation=True
        )
        self.conv5_up = SeparableConvBlock(
            device, parameters.conv5_up, shard_layout, conv_params.conv5_up, deallocate_activation=True
        )
        self.conv4_up = SeparableConvBlock(
            device, parameters.conv4_up, shard_layout, conv_params.conv4_up, deallocate_activation=True
        )
        self.conv3_up = SeparableConvBlock(
            device, parameters.conv3_up, shard_layout, conv_params.conv3_up, deallocate_activation=True
        )

        # Initialize separable conv blocks for downsampling path
        self.conv4_down = SeparableConvBlock(
            device,
            parameters.conv4_down,
            shard_layout,
            conv_params.conv4_down,
            deallocate_activation=True,
        )
        self.conv5_down = SeparableConvBlock(
            device,
            parameters.conv5_down,
            shard_layout,
            conv_params.conv5_down,
            deallocate_activation=True,
        )
        self.conv6_down = SeparableConvBlock(
            device,
            parameters.conv6_down,
            shard_layout,
            conv_params.conv6_down,
            deallocate_activation=True,
        )
        self.conv7_down = SeparableConvBlock(
            device,
            parameters.conv7_down,
            shard_layout,
            conv_params.conv7_down,
            deallocate_activation=True,
        )

        # Initialize maxpool layers for downsampling
        self.p4_downsample = MaxPool2dDynamicSamePadding(
            device,
            None,
            # shard_layout,
            None,
            conv_params.p4_downsample,
            deallocate_activation=False,
        )
        self.p5_downsample = MaxPool2dDynamicSamePadding(
            device,
            None,
            # shard_layout,
            None,
            conv_params.p5_downsample,
            deallocate_activation=False,
        )
        self.p6_downsample = MaxPool2dDynamicSamePadding(
            device,
            None,
            # shard_layout,
            None,
            conv_params.p6_downsample,
            deallocate_activation=False,
        )
        self.p7_downsample = MaxPool2dDynamicSamePadding(
            device,
            None,
            # shard_layout,
            None,
            conv_params.p7_downsample,
            deallocate_activation=False,
        )

        # Initialize channel reduction layers for first_time
        if self.first_time:
            # import pdb; pdb.set_trace()
            self.p3_down_channel = Conv2dDynamicSamePadding(
                device,
                parameters.p3_down_channel[0],
                shard_layout,
                conv_params.p3_down_channel[0],
                deallocate_activation=deallocate_activation,
            )
            self.p4_down_channel = Conv2dDynamicSamePadding(
                device,
                parameters.p4_down_channel[0],
                shard_layout,
                conv_params.p4_down_channel[0],
                deallocate_activation=False,
            )
            self.p5_down_channel = Conv2dDynamicSamePadding(
                device,
                parameters.p5_down_channel[0],
                shard_layout,
                conv_params.p5_down_channel[0],
                deallocate_activation=False,
            )

            # P5 to P6 conversion (conv + maxpool)
            # import pdb; pdb.set_trace()
            self.p5_to_p6_conv = Conv2dDynamicSamePadding(
                device,
                parameters.p5_to_p6[0],
                shard_layout,
                conv_params.p5_to_p6[0],
                deallocate_activation=False,
            )
            self.p5_to_p6_pool = MaxPool2dDynamicSamePadding(
                device,
                None,
                # shard_layout,
                None,
                conv_params.p5_to_p6[2],
                deallocate_activation=False,
            )

            # P6 to P7 conversion (maxpool only)
            self.p6_to_p7 = MaxPool2dDynamicSamePadding(
                device,
                None,
                # shard_layout,
                None,
                conv_params.p6_to_p7[0],
                deallocate_activation=False,
            )

            # Additional channel reduction for bottom-up path
            self.p4_down_channel_2 = Conv2dDynamicSamePadding(
                device,
                parameters.p4_down_channel_2[0],
                shard_layout,
                conv_params.p4_down_channel_2[0],
                deallocate_activation=deallocate_activation,
            )
            self.p5_down_channel_2 = Conv2dDynamicSamePadding(
                device,
                parameters.p5_down_channel_2[0],
                shard_layout,
                conv_params.p5_down_channel_2[0],
                deallocate_activation=deallocate_activation,
            )

        # Store attention weights as TTNN tensors
        if attention:
            self.p6_w1 = ttnn.from_torch(parameters.p6_w1, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self.p5_w1 = ttnn.from_torch(parameters.p5_w1, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self.p4_w1 = ttnn.from_torch(parameters.p4_w1, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self.p3_w1 = ttnn.from_torch(parameters.p3_w1, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            self.p4_w2 = ttnn.from_torch(parameters.p4_w2, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self.p5_w2 = ttnn.from_torch(parameters.p5_w2, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self.p6_w2 = ttnn.from_torch(parameters.p6_w2, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            self.p7_w2 = ttnn.from_torch(parameters.p7_w2, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

            p7_w2_relu = ttnn.relu(self.p7_w2)
            p7_w2_sum = ttnn.sum(p7_w2_relu, dim=0)
            denominator = ttnn.add(p7_w2_sum, self.epsilon)
            self.p7_weight_0 = ttnn.div(p7_w2_relu[0], denominator)
            self.p7_weight_1 = ttnn.div(p7_w2_relu[1], denominator)
            ttnn.deallocate(p7_w2_relu)
            ttnn.deallocate(p7_w2_sum)
            ttnn.deallocate(denominator)

    def _upsample(self, x, scale_factor):
        # Convert to interleaved if sharded
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        # Convert to ROW_MAJOR layout for upsample
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        # Perform upsample
        x = ttnn.upsample(x, (scale_factor, scale_factor))

        # Convert back to TILE layout if needed
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

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

        # P6_1 = weighted_sum(P6_0, upsample(P7_0))
        p6_w1_tile = ttnn.to_layout(self.p6_w1, ttnn.TILE_LAYOUT)
        p6_w1_relu = ttnn.relu(p6_w1_tile)
        p6_w1_sum = ttnn.sum(p6_w1_relu, dim=0)
        denominator = ttnn.add(p6_w1_sum, self.epsilon)  # Use ttnn.add for tensor + scalar
        weight_0 = ttnn.div(p6_w1_relu[0], denominator)
        weight_1 = ttnn.div(p6_w1_relu[1], denominator)

        p7_upsampled = self._upsample(p7_in, scale_factor=2)
        term1 = ttnn.mul(weight_0, p6_in)
        term2 = ttnn.mul(weight_1, p7_upsampled)
        term2_reshaped = ttnn.reshape(term2, term1.shape)
        p6_weighted = ttnn.add(term1, term2_reshaped)
        p6_up = self.conv6_up(self._swish(p6_weighted))

        # P5_1 = weighted_sum(P5_0, upsample(P6_1))
        p5_w1_tile = ttnn.to_layout(self.p5_w1, ttnn.TILE_LAYOUT)
        p5_w1_relu = ttnn.relu(p5_w1_tile)
        p5_w1_sum = ttnn.sum(p5_w1_relu, dim=0)
        denominator = ttnn.add(p5_w1_sum, self.epsilon)
        weight_0 = ttnn.div(p5_w1_relu[0], denominator)
        weight_1 = ttnn.div(p5_w1_relu[1], denominator)

        p6_upsampled = self._upsample(p6_up, scale_factor=2)
        term1 = ttnn.mul(weight_0, p5_in)
        term2 = ttnn.mul(weight_1, p6_upsampled)
        term2_reshaped = ttnn.reshape(term2, term1.shape)
        p5_weighted = ttnn.add(term1, term2_reshaped)
        p5_up = self.conv5_up(self._swish(p5_weighted))

        # P4_1 = weighted_sum(P4_0, upsample(P5_1))
        p4_w1_tile = ttnn.to_layout(self.p4_w1, ttnn.TILE_LAYOUT)
        p4_w1_relu = ttnn.relu(p4_w1_tile)
        p4_w1_sum = ttnn.sum(p4_w1_relu, dim=0)
        denominator = ttnn.add(p4_w1_sum, self.epsilon)
        weight_0 = ttnn.div(p4_w1_relu[0], denominator)
        weight_1 = ttnn.div(p4_w1_relu[1], denominator)

        p5_upsampled = self._upsample(p5_up, scale_factor=2)
        term1 = ttnn.mul(weight_0, p4_in)
        term2 = ttnn.mul(weight_1, p5_upsampled)
        term2_reshaped = ttnn.reshape(term2, term1.shape)
        p4_weighted = ttnn.add(term1, term2_reshaped)
        p4_up = self.conv4_up(self._swish(p4_weighted))

        # P3_2 = weighted_sum(P3_0, upsample(P4_1))
        p3_w1_tile = ttnn.to_layout(self.p3_w1, ttnn.TILE_LAYOUT)
        p3_w1_relu = ttnn.relu(p3_w1_tile)
        p3_w1_sum = ttnn.sum(p3_w1_relu, dim=0)
        denominator = ttnn.add(p3_w1_sum, self.epsilon)
        weight_0 = ttnn.div(p3_w1_relu[0], denominator)
        weight_1 = ttnn.div(p3_w1_relu[1], denominator)

        p4_upsampled = self._upsample(p4_up, scale_factor=2)
        term1 = ttnn.mul(weight_0, p3_in)
        ttnn.deallocate(p3_in)
        term2 = ttnn.mul(weight_1, p4_upsampled)
        term2_reshaped = ttnn.reshape(term2, term1.shape)
        p3_weighted = ttnn.add(term1, term2_reshaped)
        p3_out = self.conv3_up(self._swish(p3_weighted))

        # Update p4_in and p5_in for bottom-up path if first_time
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p4_in = ttnn.to_memory_config(p4_in, ttnn.DRAM_MEMORY_CONFIG)
            p5_in = self.p5_down_channel_2(p5)
            p5_in = ttnn.to_memory_config(p5_in, ttnn.DRAM_MEMORY_CONFIG)

        # Bottom-up pathway with weighted attention
        # P4_2 = weighted_sum(P4_0, P4_1, downsample(P3_2))
        p4_w2_tile = ttnn.to_layout(self.p4_w2, ttnn.TILE_LAYOUT)
        p4_w2_relu = ttnn.relu(p4_w2_tile)
        p4_w2_sum = ttnn.sum(p4_w2_relu, dim=0)
        denominator = ttnn.add(p4_w2_sum, self.epsilon)
        weight_0 = ttnn.div(p4_w2_relu[0], denominator)
        weight_1 = ttnn.div(p4_w2_relu[1], denominator)
        weight_2 = ttnn.div(p4_w2_relu[2], denominator)

        p3_downsampled = self.p4_downsample(p3_out)
        term1 = ttnn.mul(weight_0, p4_in)
        ttnn.deallocate(p4_in)
        term2 = ttnn.mul(weight_1, p4_up)
        term3 = ttnn.mul(weight_2, p3_downsampled)
        term2_reshaped = ttnn.reshape(term2, term1.shape)
        term3_reshaped = ttnn.reshape(term3, term1.shape)
        p4_weighted = ttnn.add(term1, term2_reshaped)
        p4_weighted = ttnn.add(p4_weighted, term3_reshaped)
        p4_out = self.conv4_down(self._swish(p4_weighted))

        # P5_2 = weighted_sum(P5_0, P5_1, downsample(P4_2))
        p5_w2_tile = ttnn.to_layout(self.p5_w2, ttnn.TILE_LAYOUT)
        p5_w2_relu = ttnn.relu(p5_w2_tile)
        p5_w2_sum = ttnn.sum(p5_w2_relu, dim=0)
        denominator = ttnn.add(p5_w2_sum, self.epsilon)
        weight_0 = ttnn.div(p5_w2_relu[0], denominator)
        weight_1 = ttnn.div(p5_w2_relu[1], denominator)
        weight_2 = ttnn.div(p5_w2_relu[2], denominator)

        p4_downsampled = self.p5_downsample(p4_out)
        term1 = ttnn.mul(weight_0, p5_in)
        ttnn.deallocate(p5_in)
        term2 = ttnn.mul(weight_1, p5_up)
        term3 = ttnn.mul(weight_2, p4_downsampled)

        term2_reshaped = ttnn.reshape(term2, term1.shape)
        term3_reshaped = ttnn.reshape(term3, term1.shape)
        p5_weighted = ttnn.add(term1, term2_reshaped)
        p5_weighted = ttnn.add(p5_weighted, term3_reshaped)
        p5_out = self.conv5_down(self._swish(p5_weighted))

        # P6_2 = weighted_sum(P6_0, P6_1, downsample(P5_2))
        p6_w2_tile = ttnn.to_layout(self.p6_w2, ttnn.TILE_LAYOUT)
        p6_w2_relu = ttnn.relu(p6_w2_tile)
        p6_w2_sum = ttnn.sum(p6_w2_relu, dim=0)
        denominator = ttnn.add(p6_w2_sum, self.epsilon)
        weight_0 = ttnn.div(p6_w2_relu[0], denominator)
        weight_1 = ttnn.div(p6_w2_relu[1], denominator)
        weight_2 = ttnn.div(p6_w2_relu[2], denominator)

        p5_downsampled = self.p6_downsample(p5_out)

        term1 = ttnn.mul(weight_0, p6_in)
        ttnn.deallocate(p6_in)
        term2 = ttnn.mul(weight_1, p6_up)
        term3 = ttnn.mul(weight_2, p5_downsampled)

        term2_reshaped = ttnn.reshape(term2, term1.shape)
        term3_reshaped = ttnn.reshape(term3, term1.shape)
        p6_weighted = ttnn.add(term1, term2_reshaped)
        p6_weighted = ttnn.add(p6_weighted, term3_reshaped)
        p6_out = self.conv6_down(self._swish(p6_weighted))

        # P7_2 = weighted_sum(P7_0, downsample(P6_2))
        # p7_w2_relu = ttnn.relu(self.p7_w2)
        # p7_w2_sum = ttnn.sum(p7_w2_relu, dim=0)
        # denominator = ttnn.add(p7_w2_sum, self.epsilon)
        # weight_0 = ttnn.div(p7_w2_relu[0], denominator)
        # weight_1 = ttnn.div(p7_w2_relu[1], denominator)

        p6_downsampled = self.p7_downsample(p6_out)
        term1 = ttnn.mul(self.p7_weight_0, p7_in)
        term2 = ttnn.mul(self.p7_weight_1, p6_downsampled)
        term2 = ttnn.reshape(term2, term1.shape)
        p7_weighted = ttnn.add(term1, term2)
        p7_out = self.conv7_down(self._swish(p7_weighted))
        ttnn.deallocate(p7_in)
        ttnn.deallocate(p6_downsampled)
        ttnn.deallocate(term1)
        ttnn.deallocate(term2)
        ttnn.deallocate(p7_weighted)

        return p3_out, p4_out, p5_out, p6_out, p7_out
