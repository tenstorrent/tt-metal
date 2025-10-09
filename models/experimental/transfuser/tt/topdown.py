# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtTopDown:
    def __init__(
        self, device, torch_model, perception_output_features=512, bev_features_channels=64, bev_upsample_factor=2
    ):
        self.device = device
        self.perception_output_features = perception_output_features
        self.bev_features_channels = bev_features_channels
        self.bev_upsample_factor = bev_upsample_factor

        # Extract and convert PyTorch weights to TT-NN format
        self.c5_conv_weight = self._prepare_conv_weight(torch_model.c5_conv.weight)
        self.c5_conv_bias = self._prepare_conv_bias(torch_model.c5_conv.bias)

        self.up_conv5_weight = self._prepare_conv_weight(torch_model.up_conv5.weight)
        self.up_conv5_bias = self._prepare_conv_bias(torch_model.up_conv5.bias)

        self.up_conv4_weight = self._prepare_conv_weight(torch_model.up_conv4.weight)
        self.up_conv4_bias = self._prepare_conv_bias(torch_model.up_conv4.bias)

        self.up_conv3_weight = self._prepare_conv_weight(torch_model.up_conv3.weight)
        self.up_conv3_bias = self._prepare_conv_bias(torch_model.up_conv3.bias)

    def _prepare_conv_weight(self, torch_weight):
        """Convert PyTorch conv weight to TT-NN format"""
        return ttnn.from_torch(torch_weight, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def _prepare_conv_bias(self, torch_bias):
        """Convert PyTorch conv bias to TT-NN format"""
        if torch_bias is None:
            return None
        bias_reshaped = torch_bias.reshape(1, 1, 1, -1)
        return ttnn.from_torch(bias_reshaped, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    def __call__(self, x):
        # Input should be in NHWC format, flatten for conv2d
        batch_size, input_height, input_width, input_channels = x.shape

        # Flatten input for conv2d: (B, H, W, C) -> (1, 1, B*H*W, C)
        x_flat = ttnn.reshape(x, (1, 1, batch_size * input_height * input_width, input_channels))

        # Lateral conv: 512 -> 64 channels (c5_conv)
        p5_flat = ttnn.conv2d(
            input_tensor=x_flat,
            weight_tensor=self.c5_conv_weight,
            bias_tensor=self.c5_conv_bias,
            device=self.device,
            in_channels=self.perception_output_features,
            out_channels=self.bev_features_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        p5_flat = ttnn.relu(p5_flat)

        # Reshape back to 4D for upsampling with explicit interleaved memory config
        p5 = ttnn.reshape(
            p5_flat,
            (batch_size, input_height, input_width, self.bev_features_channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Use interleaved memory layout
        )

        # CRITICAL: Ensure ROW_MAJOR_LAYOUT before upsampling
        if p5.layout == ttnn.TILE_LAYOUT:
            p5 = ttnn.to_layout(p5, ttnn.ROW_MAJOR_LAYOUT)

        # First upsample stage
        p5_upsampled = ttnn.upsample(p5, (self.bev_upsample_factor, self.bev_upsample_factor), mode="bilinear")
        up_height = input_height * self.bev_upsample_factor
        up_width = input_width * self.bev_upsample_factor

        # Flatten for conv2d
        p5_up_flat = ttnn.reshape(p5_upsampled, (1, 1, batch_size * up_height * up_width, self.bev_features_channels))

        p4_flat = ttnn.conv2d(
            input_tensor=p5_up_flat,
            weight_tensor=self.up_conv5_weight,
            bias_tensor=self.up_conv5_bias,
            device=self.device,
            in_channels=self.bev_features_channels,
            out_channels=self.bev_features_channels,
            batch_size=batch_size,
            input_height=up_height,
            input_width=up_width,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        p4_flat = ttnn.relu(p4_flat)

        # Reshape back to 4D for upsampling with explicit interleaved memory config
        p4_flat = ttnn.sharded_to_interleaved(p4_flat, ttnn.DRAM_MEMORY_CONFIG)
        p4 = ttnn.reshape(
            p4_flat,
            (batch_size, up_height, up_width, self.bev_features_channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Ensure ROW_MAJOR_LAYOUT before upsampling
        if p4.layout == ttnn.TILE_LAYOUT:
            p4 = ttnn.to_layout(p4, ttnn.ROW_MAJOR_LAYOUT)

        # Second upsample stage
        p4_upsampled = ttnn.upsample(p4, (self.bev_upsample_factor, self.bev_upsample_factor), mode="bilinear")
        up_height2 = up_height * self.bev_upsample_factor
        up_width2 = up_width * self.bev_upsample_factor

        # Flatten for conv2d
        p4_up_flat = ttnn.reshape(p4_upsampled, (1, 1, batch_size * up_height2 * up_width2, self.bev_features_channels))

        p3_flat = ttnn.conv2d(
            input_tensor=p4_up_flat,
            weight_tensor=self.up_conv4_weight,
            bias_tensor=self.up_conv4_bias,
            device=self.device,
            in_channels=self.bev_features_channels,
            out_channels=self.bev_features_channels,
            batch_size=batch_size,
            input_height=up_height2,
            input_width=up_width2,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        p3_flat = ttnn.relu(p3_flat)

        # Reshape back to 4D for upsampling with explicit interleaved memory config
        p3 = ttnn.reshape(
            p3_flat,
            (batch_size, up_height2, up_width2, self.bev_features_channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Ensure ROW_MAJOR_LAYOUT before upsampling
        if p3.layout == ttnn.TILE_LAYOUT:
            p3 = ttnn.to_layout(p3, ttnn.ROW_MAJOR_LAYOUT)

        # Third upsample stage
        p3_upsampled = ttnn.upsample(p3, (self.bev_upsample_factor, self.bev_upsample_factor), mode="bilinear")
        up_height3 = up_height2 * self.bev_upsample_factor
        up_width3 = up_width2 * self.bev_upsample_factor

        # Flatten for conv2d
        p3_up_flat = ttnn.reshape(p3_upsampled, (1, 1, batch_size * up_height3 * up_width3, self.bev_features_channels))

        p2_flat = ttnn.conv2d(
            input_tensor=p3_up_flat,
            weight_tensor=self.up_conv3_weight,
            bias_tensor=self.up_conv3_bias,
            device=self.device,
            in_channels=self.bev_features_channels,
            out_channels=self.bev_features_channels,
            batch_size=batch_size,
            input_height=up_height3,
            input_width=up_width3,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
        )
        p2_flat = ttnn.relu(p2_flat)

        # Reshape all outputs back to 4D format with explicit interleaved memory config
        p2 = ttnn.reshape(
            p2_flat,
            (batch_size, up_height3, up_width3, self.bev_features_channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        p3_out = ttnn.reshape(
            p3_flat,
            (batch_size, up_height2, up_width2, self.bev_features_channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        p4_out = ttnn.reshape(
            p4_flat,
            (batch_size, up_height, up_width, self.bev_features_channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        p5_out = ttnn.reshape(
            p5_flat,
            (batch_size, input_height, input_width, self.bev_features_channels),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return p2, p3_out, p4_out, p5_out
