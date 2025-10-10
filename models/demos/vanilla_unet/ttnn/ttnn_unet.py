# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.vanilla_unet.ttnn.common import Conv, ConvTranspose


def torch_to_ttnn(input, device=None):
    input = ttnn.from_torch(input, ttnn.bfloat16, device=device)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


class TtUnet:
    def __init__(
        self,
        device,
        parameters,
        model,
        conv_args,
    ):
        self.model = model
        self.conv_args = conv_args
        self.bn_parameters = parameters["decoder1"]["bn"]

        self.enc1_1 = Conv(
            [1, 1, 1, 1],
            parameters["encoder1"][0],
            act_block_h=32 * 10,
            reshard=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            conv_args=conv_args["encoder1"][0],
            dtype=ttnn.bfloat8_b,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.enc1_2 = Conv(
            [1, 1, 1, 1],
            parameters["encoder1"][1],
            act_block_h=32 * 5,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            conv_args=conv_args["encoder1"][3],
        )

        self.enc2_1 = Conv(
            [1, 1, 1, 1],
            parameters["encoder2"][0],
            reshard=True,
            conv_args=conv_args["encoder2"][0],
            act_block_h=32 * 6,
        )
        self.enc2_2 = Conv(
            [1, 1, 1, 1],
            parameters["encoder2"][1],
            act_block_h=32 * 6,
            auto_shard=True,
            conv_args=conv_args["encoder2"][3],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )

        self.enc3_1 = Conv(
            [1, 1, 1, 1],
            parameters["encoder3"][0],
            auto_shard=True,
            conv_args=conv_args["encoder3"][0],
            dtype=ttnn.bfloat8_b,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.enc3_2 = Conv(
            [1, 1, 1, 1],
            parameters["encoder3"][1],
            act_block_h=32,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            auto_shard=True,
            conv_args=conv_args["encoder3"][3],
        )

        self.enc4_1 = Conv(
            [1, 1, 1, 1],
            parameters["encoder4"][0],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard=True,
            conv_args=conv_args["encoder4"][0],
            dtype=ttnn.bfloat8_b,
            output_layout=ttnn.TILE_LAYOUT,
        )
        self.enc4_2 = Conv(
            [1, 1, 1, 1],
            parameters["encoder4"][1],
            conv_args=conv_args["encoder4"][3],
            dtype=ttnn.bfloat8_b,
            output_layout=ttnn.TILE_LAYOUT,
        )

        self.bottleneck_1 = Conv(
            [1, 1, 1, 1],
            parameters["bottleneck"][0],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            height_sharding=False,
            reshard=True,
            conv_args=conv_args["bottleneck"][0],
        )
        self.bottleneck_2 = Conv(
            [1, 1, 1, 1],
            parameters["bottleneck"][1],
            height_sharding=False,
            reshard=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            conv_args=conv_args["bottleneck"][3],
        )

        self.upconv4 = ConvTranspose([2, 2, 0, 0], parameters["upconv4"], reshard=True, conv_args=conv_args["upconv4"])
        self.dec4_1 = Conv(
            [1, 1, 1, 1],
            parameters["decoder4"][0],
            auto_shard=True,
            conv_args=conv_args["decoder4"][0],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.dec4_2 = Conv(
            [1, 1, 1, 1],
            parameters["decoder4"][1],
            auto_shard=True,
            conv_args=conv_args["decoder4"][3],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )

        self.upconv3 = ConvTranspose(
            [2, 2, 0, 0],
            parameters["upconv3"],
            reshard=True,
            conv_args=conv_args["upconv3"],
        )
        self.dec3_1 = Conv(
            [1, 1, 1, 1],
            parameters["decoder3"][0],
            act_block_h=32,
            reshard=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            conv_args=conv_args["decoder3"][0],
        )
        self.dec3_2 = Conv(
            [1, 1, 1, 1], parameters["decoder3"][1], height_sharding=False, conv_args=conv_args["decoder3"][3]
        )

        self.upconv2 = ConvTranspose(
            [2, 2, 0, 0],
            parameters["upconv2"],
            act_block_h=32,
            reshard=True,
            conv_args=conv_args["upconv2"],
        )
        self.dec2_1 = Conv(
            [1, 1, 1, 1],
            parameters["decoder2"][0],
            act_block_h=32,
            conv_args=conv_args["decoder2"][0],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )
        self.dec2_2 = Conv([1, 1, 1, 1], parameters["decoder2"][1], act_block_h=64, conv_args=conv_args["decoder2"][3])

        self.upconv1 = ConvTranspose(
            [2, 2, 0, 0],
            parameters["upconv1"],
            act_block_h=32,
            output_layout=ttnn.TILE_LAYOUT,
            conv_args=conv_args["upconv1"],
        )
        self.dec1_1 = Conv(
            [1, 1, 1, 1],
            parameters["decoder1"][0],
            auto_shard=True,
            dtype=ttnn.bfloat8_b,
            output_layout=ttnn.TILE_LAYOUT,
            reallocate_halo_output=True,
            conv_args=conv_args["decoder1"][0],
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            act_block_h=32 * 10,
        )
        self.dec1_2 = Conv(
            [1, 1, 1, 1],
            parameters["decoder1"][1],
            act_block_h=32 * 10,
            reshard=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reallocate_halo_output=True,
            conv_args=conv_args["decoder1"][3],
        )

        self.conv = Conv(
            [1, 1, 0, 0],
            parameters["conv"],
            activation=None,
            reshard=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            dtype=ttnn.bfloat8_b,
            output_layout=ttnn.TILE_LAYOUT,
            conv_args=conv_args["conv"],
        )

    def __call__(self, device, input_tensor):
        N, C, H, W = input_tensor.shape
        min_channels = 16  # Padding from image channels (3) to min channels (16)
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(input_tensor, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = input_tensor
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))
        ttnn.deallocate(nchw)
        ttnn.deallocate(input_tensor)
        nhwc = ttnn.reallocate(nhwc)

        enc1 = self.enc1_1(device, nhwc)
        enc1 = self.enc1_2(device, enc1)
        pool_1 = ttnn.max_pool2d(
            input_tensor=enc1,
            batch_size=1,
            input_h=self.conv_args.pool1.input_height,
            input_w=self.conv_args.pool1.input_width,
            channels=enc1.shape[3],
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )

        enc1 = ttnn.to_memory_config(enc1, ttnn.DRAM_MEMORY_CONFIG)

        memory_config = ttnn.create_sharded_memory_config(
            [1216, 32],
            core_grid=device.core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        pool_1 = ttnn.to_memory_config(pool_1, memory_config)

        enc2 = self.enc2_1(device, pool_1)
        ttnn.deallocate(pool_1)
        enc2 = self.enc2_2(device, enc2)

        if enc2.is_sharded:
            pool_in = ttnn.clone(ttnn.sharded_to_interleaved(enc2, ttnn.L1_MEMORY_CONFIG))

        pool_2 = ttnn.max_pool2d(
            input_tensor=pool_in,
            batch_size=1,
            input_h=self.conv_args.pool2.input_height,
            input_w=self.conv_args.pool2.input_width,
            channels=pool_in.shape[3],
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )
        enc2 = ttnn.to_memory_config(enc2, ttnn.DRAM_MEMORY_CONFIG)

        enc3 = self.enc3_1(device, pool_2)
        ttnn.deallocate(pool_2)
        enc3 = self.enc3_2(device, enc3)

        pool_3 = ttnn.max_pool2d(
            input_tensor=enc3,
            batch_size=1,
            input_h=self.conv_args.pool3.input_height,
            input_w=self.conv_args.pool3.input_width,
            channels=enc3.shape[3],
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )

        enc3 = ttnn.to_memory_config(enc3, ttnn.L1_MEMORY_CONFIG)

        enc4 = self.enc4_1(device, pool_3)
        ttnn.deallocate(pool_3)
        enc4 = self.enc4_2(device, enc4)

        pool_4 = ttnn.max_pool2d(
            input_tensor=enc4,
            batch_size=1,
            input_h=self.conv_args.pool4.input_height,
            input_w=self.conv_args.pool4.input_width,
            channels=enc4.shape[3],
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )
        ttnn.deallocate(pool_in)

        bottleneck = self.bottleneck_1(device, pool_4)
        ttnn.deallocate(pool_4)
        bottleneck = self.bottleneck_2(device, bottleneck)

        dec4 = self.upconv4(device, bottleneck)

        if dec4.is_sharded:
            dec4 = ttnn.sharded_to_interleaved(dec4, ttnn.L1_MEMORY_CONFIG)
        if enc4.is_sharded:
            enc4 = ttnn.sharded_to_interleaved(enc4, ttnn.L1_MEMORY_CONFIG)
        dec4 = ttnn.concat([dec4, enc4], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        ttnn.deallocate(enc4)

        dec4 = self.dec4_1(device, dec4)
        dec4 = self.dec4_2(device, dec4)

        dec3 = self.upconv3(device, dec4)
        ttnn.deallocate(dec4)

        if dec3.is_sharded:
            dec3 = ttnn.sharded_to_interleaved(dec3, ttnn.L1_MEMORY_CONFIG)
        if enc3.is_sharded:
            enc3 = ttnn.sharded_to_interleaved(enc3, ttnn.L1_MEMORY_CONFIG)

        dec3 = ttnn.concat([dec3, enc3], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(enc3)
        dec3 = self.dec3_1(device, dec3)
        dec3 = self.dec3_2(device, dec3)

        dec2 = self.upconv2(device, dec3)
        ttnn.deallocate(dec3)

        memory_config = ttnn.create_sharded_memory_config(
            [1216, 64],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        out_memory_config = ttnn.create_sharded_memory_config(
            [1216, 128],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        enc2 = ttnn.interleaved_to_sharded(enc2, memory_config)

        dec2 = ttnn.concat([dec2, enc2], dim=3, memory_config=out_memory_config)

        ttnn.deallocate(enc2)

        dec2 = self.dec2_1(device, dec2)
        dec2 = self.dec2_2(device, dec2)
        dec1 = self.upconv1(device, dec2)
        ttnn.deallocate(dec2)
        dec1 = ttnn.to_layout(dec1, ttnn.ROW_MAJOR_LAYOUT)

        enc1 = ttnn.sharded_to_interleaved(enc1, ttnn.L1_MEMORY_CONFIG)

        memory_config = ttnn.create_sharded_memory_config(
            [4800, 32],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        out_memory_config = ttnn.create_sharded_memory_config(
            [4800, 64],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        enc1 = ttnn.interleaved_to_sharded(enc1, memory_config)
        dec1 = ttnn.concat(
            [dec1, enc1],
            dim=3,
            memory_config=out_memory_config,
        )
        ttnn.deallocate(enc1)

        dec1 = ttnn.to_layout(dec1, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        if dec1.is_sharded:
            dec1 = ttnn.sharded_to_interleaved(dec1, ttnn.L1_MEMORY_CONFIG)
        dec1 = self.dec1_1(device, dec1)

        dec1 = self.dec1_2(device, dec1)
        ttnn_output = self.conv(device, dec1)
        ttnn.deallocate(dec1)
        ttnn_output = ttnn.add(ttnn_output, 0.0, dtype=ttnn.bfloat16)
        ttnn_output = ttnn.sigmoid_accurate(ttnn_output)

        return ttnn_output
