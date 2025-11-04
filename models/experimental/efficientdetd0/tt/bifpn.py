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
        self.use_p8 = use_p8
        self.attention = attention
        self.first_time = first_time
        self.shard_layout = shard_layout

        # Initialize separable conv blocks for upsampling path
        self.conv6_up = SeparableConvBlock(
            device, parameters.conv6_up, shard_layout, conv_params.conv6_up, deallocate_activation=deallocate_activation
        )
        self.conv5_up = SeparableConvBlock(
            device, parameters.conv5_up, shard_layout, conv_params.conv5_up, deallocate_activation=deallocate_activation
        )
        self.conv4_up = SeparableConvBlock(
            device, parameters.conv4_up, shard_layout, conv_params.conv4_up, deallocate_activation=deallocate_activation
        )
        self.conv3_up = SeparableConvBlock(
            device, parameters.conv3_up, shard_layout, conv_params.conv3_up, deallocate_activation=deallocate_activation
        )

        # Initialize separable conv blocks for downsampling path
        self.conv4_down = SeparableConvBlock(
            device,
            parameters.conv4_down,
            shard_layout,
            conv_params.conv4_down,
            deallocate_activation=deallocate_activation,
        )
        self.conv5_down = SeparableConvBlock(
            device,
            parameters.conv5_down,
            shard_layout,
            conv_params.conv5_down,
            deallocate_activation=deallocate_activation,
        )
        self.conv6_down = SeparableConvBlock(
            device,
            parameters.conv6_down,
            shard_layout,
            conv_params.conv6_down,
            deallocate_activation=deallocate_activation,
        )
        self.conv7_down = SeparableConvBlock(
            device,
            parameters.conv7_down,
            shard_layout,
            conv_params.conv7_down,
            deallocate_activation=deallocate_activation,
        )

        # if use_p8:
        #     self.conv7_up = SeparableConvBlock(
        #         device, parameters.conv7_up, shard_layout, conv_params.conv7_up,
        #         deallocate_activation=deallocate_activation
        #     )
        #     self.conv8_down = SeparableConvBlock(
        #         device, parameters.conv8_down, shard_layout, conv_params.conv8_down,
        #         deallocate_activation=deallocate_activation
        #     )

        # Initialize maxpool layers for downsampling
        self.p4_downsample = MaxPool2dDynamicSamePadding(
            device,
            None,
            # shard_layout,
            None,
            conv_params.p4_downsample,
            deallocate_activation=deallocate_activation,
        )
        self.p5_downsample = MaxPool2dDynamicSamePadding(
            device,
            None,
            # shard_layout,
            None,
            conv_params.p5_downsample,
            deallocate_activation=deallocate_activation,
        )
        self.p6_downsample = MaxPool2dDynamicSamePadding(
            device,
            None,
            # shard_layout,
            None,
            conv_params.p6_downsample,
            deallocate_activation=deallocate_activation,
        )
        self.p7_downsample = MaxPool2dDynamicSamePadding(
            device,
            None,
            # shard_layout,
            None,
            conv_params.p7_downsample,
            deallocate_activation=deallocate_activation,
        )

        # if use_p8:
        #     self.p8_downsample = MaxPool2dDynamicSamePadding(
        #         device, None, shard_layout, conv_params.p8_downsample,
        #         deallocate_activation=deallocate_activation
        #     )

        # Initialize channel reduction layers for first_time
        if self.first_time:
            # import pdb; pdb.set_trace()
            self.p3_down_channel = Conv2dDynamicSamePadding(
                device,
                parameters.p3_down_channel,
                shard_layout,
                conv_params.p3_down_channel,
                deallocate_activation=deallocate_activation,
            )
            self.p4_down_channel = Conv2dDynamicSamePadding(
                device,
                parameters.p4_down_channel,
                shard_layout,
                conv_params.p4_down_channel,
                deallocate_activation=deallocate_activation,
            )
            self.p5_down_channel = Conv2dDynamicSamePadding(
                device,
                parameters.p5_down_channel,
                shard_layout,
                conv_params.p5_down_channel,
                deallocate_activation=deallocate_activation,
            )

            # P5 to P6 conversion (conv + maxpool)
            # import pdb; pdb.set_trace()
            self.p5_to_p6_conv = Conv2dDynamicSamePadding(
                device,
                parameters.p5_to_p6_conv,
                shard_layout,
                conv_params.p5_to_p6_conv,
                deallocate_activation=deallocate_activation,
            )
            self.p5_to_p6_pool = MaxPool2dDynamicSamePadding(
                device,
                None,
                # shard_layout,
                None,
                conv_params.p5_to_p6_pool,
                deallocate_activation=deallocate_activation,
            )

            # P6 to P7 conversion (maxpool only)
            self.p6_to_p7 = MaxPool2dDynamicSamePadding(
                device,
                None,
                # shard_layout,
                None,
                conv_params.p6_to_p7,
                deallocate_activation=deallocate_activation,
            )

            # if use_p8:
            #     self.p7_to_p8 = MaxPool2dDynamicSamePadding(
            #         device, None,  shard_layout, conv_params.p7_to_p8,
            #         deallocate_activation=deallocate_activation
            #     )

            # Additional channel reduction for bottom-up path
            self.p4_down_channel_2 = Conv2dDynamicSamePadding(
                device,
                parameters.p4_down_channel_2,
                shard_layout,
                conv_params.p4_down_channel_2,
                deallocate_activation=deallocate_activation,
            )
            self.p5_down_channel_2 = Conv2dDynamicSamePadding(
                device,
                parameters.p5_down_channel_2,
                shard_layout,
                conv_params.p5_down_channel_2,
                deallocate_activation=deallocate_activation,
            )

        # Store attention weights as TTNN tensors
        if attention:
            self.p6_w1 = ttnn.from_torch(parameters.p6_w1, device=device, dtype=ttnn.bfloat16)
            self.p5_w1 = ttnn.from_torch(parameters.p5_w1, device=device, dtype=ttnn.bfloat16)
            self.p4_w1 = ttnn.from_torch(parameters.p4_w1, device=device, dtype=ttnn.bfloat16)
            self.p3_w1 = ttnn.from_torch(parameters.p3_w1, device=device, dtype=ttnn.bfloat16)

            self.p4_w2 = ttnn.from_torch(parameters.p4_w2, device=device, dtype=ttnn.bfloat16)
            self.p5_w2 = ttnn.from_torch(parameters.p5_w2, device=device, dtype=ttnn.bfloat16)
            self.p6_w2 = ttnn.from_torch(parameters.p6_w2, device=device, dtype=ttnn.bfloat16)
            self.p7_w2 = ttnn.from_torch(parameters.p7_w2, device=device, dtype=ttnn.bfloat16)

    # def _upsample(self, x: ttnn.Tensor, scale_factor: int = 2) -> ttnn.Tensor:
    #     """Upsample using nearest neighbor interpolation"""
    #     return ttnn.upsample(x, (scale_factor, scale_factor))

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
            return self._forward(inputs)

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

        # Top-down pathway with weighted attention

        # P6_1 = weighted_sum(P6_0, upsample(P7_0))
        # p6_w1_tile = ttnn.to_layout(self.p6_w1, ttnn.TILE_LAYOUT)
        # p6_w1_relu = ttnn.relu(p6_w1_tile)
        # p6_w1_sum = ttnn.sum(p6_w1_relu, dim=0)
        # weight_0 = p6_w1_relu[0] / (p6_w1_sum + self.epsilon)
        # weight_1 = p6_w1_relu[1] / (p6_w1_sum + self.epsilon)

        # P6_1 = weighted_sum(P6_0, upsample(P7_0))
        p6_w1_tile = ttnn.to_layout(self.p6_w1, ttnn.TILE_LAYOUT)
        p6_w1_relu = ttnn.relu(p6_w1_tile)
        p6_w1_sum = ttnn.sum(p6_w1_relu, dim=0)
        denominator = ttnn.add(p6_w1_sum, self.epsilon)  # Use ttnn.add for tensor + scalar
        weight_0 = ttnn.div(p6_w1_relu[0], denominator)
        weight_1 = ttnn.div(p6_w1_relu[1], denominator)

        p7_upsampled = self._upsample(p7_in, scale_factor=2)
        # p6_weighted = weight_0 * p6_in + weight_1 * p7_upsampled
        # p6_up = self.conv6_up(self._swish(p6_weighted))
        term1 = ttnn.mul(weight_0, p6_in)
        term2 = ttnn.mul(weight_1, p7_upsampled)
        # term1_reshaped = ttnn.reshape(term1, (1, 1, 64, 64))
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
        # p5_weighted = weight_0 * p5_in + weight_1 * p6_upsampled
        # p5_up = self.conv5_up(self._swish(p5_weighted))
        term1 = ttnn.mul(weight_0, p5_in)
        term2 = ttnn.mul(weight_1, p6_upsampled)
        # term1_reshaped = ttnn.reshape(term1, (1, 1, 256, 64))
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
        # p4_weighted = weight_0 * p4_in + weight_1 * p5_upsampled
        # p4_up = self.conv4_up(self._swish(p4_weighted))
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
        # p3_weighted = weight_0 * p3_in + weight_1 * p4_upsampled
        # p3_out = self.conv3_up(self._swish(p3_weighted))
        term1 = ttnn.mul(weight_0, p3_in)
        term2 = ttnn.mul(weight_1, p4_upsampled)
        term2_reshaped = ttnn.reshape(term2, term1.shape)
        p3_weighted = ttnn.add(term1, term2_reshaped)
        p3_out = self.conv3_up(self._swish(p3_weighted))

        # Update p4_in and p5_in for bottom-up path if first_time
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

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
        term2 = ttnn.mul(weight_1, p4_up)
        term3 = ttnn.mul(weight_2, p3_downsampled)
        term2_reshaped = ttnn.reshape(term2, term1.shape)
        term3_reshaped = ttnn.reshape(term3, term1.shape)
        p4_weighted = ttnn.add(term1, term2_reshaped)
        p4_weighted = ttnn.add(p4_weighted, term3_reshaped)
        # p4_weighted = weight_0 * p4_in + weight_1 * p4_up + weight_2 * p3_downsampled
        # p4_out = self.conv4_down(self._swish(p4_weighted))
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
        term2 = ttnn.mul(weight_1, p5_up)
        term3 = ttnn.mul(weight_2, p4_downsampled)

        term2_reshaped = ttnn.reshape(term2, term1.shape)
        term3_reshaped = ttnn.reshape(term3, term1.shape)
        p5_weighted = ttnn.add(term1, term2_reshaped)
        p5_weighted = ttnn.add(p5_weighted, term3_reshaped)
        # p5_weighted = weight_0 * p5_in + weight_1 * p5_up + weight_2 * p4_downsampled
        # p5_out = self.conv5_down(self._swish(p5_weighted))
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
        term2 = ttnn.mul(weight_1, p6_up)
        term3 = ttnn.mul(weight_2, p5_downsampled)

        term2_reshaped = ttnn.reshape(term2, term1.shape)
        term3_reshaped = ttnn.reshape(term3, term1.shape)
        p6_weighted = ttnn.add(term1, term2_reshaped)
        p6_weighted = ttnn.add(p6_weighted, term3_reshaped)
        # p6_weighted = weight_0 * p6_in + weight_1 * p6_up + weight_2 * p5_downsampled
        # p6_out = self.conv6_down(self._swish(p6_weighted))
        p6_out = self.conv6_down(self._swish(p6_weighted))

        # P7_2 = weighted_sum(P7_0, downsample(P6_2))
        p7_w2_tile = ttnn.to_layout(self.p7_w2, ttnn.TILE_LAYOUT)
        p7_w2_relu = ttnn.relu(p7_w2_tile)
        p7_w2_sum = ttnn.sum(p7_w2_relu, dim=0)
        denominator = ttnn.add(p7_w2_sum, self.epsilon)
        weight_0 = ttnn.div(p7_w2_relu[0], denominator)
        weight_1 = ttnn.div(p7_w2_relu[1], denominator)

        p6_downsampled = self.p7_downsample(p6_out)
        term1 = ttnn.mul(weight_0, p7_in)
        term2 = ttnn.mul(weight_1, p6_downsampled)
        term2_reshaped = ttnn.reshape(term2, term1.shape)
        p7_weighted = ttnn.add(term1, term2_reshaped)
        # p7_weighted = weight_0 * p7_in + weight_1 * p6_downsampled
        # p7_out = self.conv7_down(self._swish(p7_weighted))
        p7_out = self.conv7_down(self._swish(p7_weighted))

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def _forward(self, inputs: Tuple[ttnn.Tensor, ...]) -> Tuple[ttnn.Tensor, ...]:
        """Forward pass without weighted attention (simple addition)"""

        if self.first_time:
            p3, p4, p5 = inputs

            # Generate P6 and P7 from P5
            p6_in = self.p5_to_p6_conv(p5)
            p6_in = self.p5_to_p6_pool(p6_in)
            p7_in = self.p6_to_p7(p6_in)

            if self.use_p8:
                p8_in = self.p7_to_p8(p7_in)

            # Channel reduction for P3, P4, P5
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            if self.use_p8:
                p3_in, p4_in, p5_in, p6_in, p7_in, p8_in = inputs
            else:
                p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # Top-down pathway
        if self.use_p8:
            # P7_1 = P7_0 + upsample(P8_0)
            p7_up = self.conv7_up(self._swish(p7_in + self._upsample(p8_in, scale_factor=2)))
            # P6_1 = P6_0 + upsample(P7_1)
            p6_up = self.conv6_up(self._swish(p6_in + self._upsample(p7_up, scale_factor=2)))
        else:
            # P6_1 = P6_0 + upsample(P7_0)
            p6_up = self.conv6_up(self._swish(p6_in + self._upsample(p7_in, scale_factor=2)))

        # P5_1 = P5_0 + upsample(P6_1)
        p5_up = self.conv5_up(self._swish(p5_in + self._upsample(p6_up, scale_factor=2)))

        # P4_1 = P4_0 + upsample(P5_1)
        p4_up = self.conv4_up(self._swish(p4_in + self._upsample(p5_up, scale_factor=2)))

        # P3_2 = P3_0 + upsample(P4_1)
        p3_out = self.conv3_up(self._swish(p3_in + self._upsample(p4_up, scale_factor=2)))

        # Update p4_in and p5_in for bottom-up path if first_time
        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # Bottom-up pathway
        # P4_2 = P4_0 + P4_1 + downsample(P3_2)
        p4_out = self.conv4_down(self._swish(p4_in + p4_up + self.p4_downsample(p3_out)))

        # P5_2 = P5_0 + P5_1 + downsample(P4_2)
        p5_out = self.conv5_down(self._swish(p5_in + p5_up + self.p5_downsample(p4_out)))

        # P6_2 = P6_0 + P6_1 + downsample(P5_2)
        p6_out = self.conv6_down(self._swish(p6_in + p6_up + self.p6_downsample(p5_out)))

        if self.use_p8:
            # P7_2 = P7_0 + P7_1 + downsample(P6_2)
            p7_out = self.conv7_down(self._swish(p7_in + p7_up + self.p7_downsample(p6_out)))

            # P8_2 = P8_0 + downsample(P7_2)
            p8_out = self.conv8_down(self._swish(p8_in + self.p8_downsample(p7_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out, p8_out
        else:
            # P7_2 = P7_0 + downsample(P6_2)
            p7_out = self.conv7_down(self._swish(p7_in + self.p7_downsample(p6_out)))

            return p3_out, p4_out, p5_out, p6_out, p7_out
