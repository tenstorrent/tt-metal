# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.functional_vanilla_unet.ttnn.common import Conv, ConvTranspose, ConvSplit


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
    ):
        self.model = model
        self.enc1_1 = Conv(
            [1, 1, 1, 1],
            parameters["encoder1"][0],
            act_block_h=64,
        )
        self.enc1_2 = Conv([1, 1, 1, 1], parameters["encoder1"][1], act_block_h=32)

        self.enc2_1 = Conv([1, 1, 1, 1], parameters["encoder2"][0], reshard=True)
        self.enc2_2 = Conv([1, 1, 1, 1], parameters["encoder2"][1], act_block_h=64, auto_shard=True)

        self.enc3_1 = Conv([1, 1, 1, 1], parameters["encoder3"][0], auto_shard=True)
        self.enc3_2 = Conv([1, 1, 1, 1], parameters["encoder3"][1], act_block_h=32, auto_shard=True)

        self.enc4_1 = Conv(
            [1, 1, 1, 1],
            parameters["encoder4"][0],
        )
        self.enc4_2 = Conv(
            [1, 1, 1, 1],
            parameters["encoder4"][1],
        )

        self.bottleneck_1 = Conv(
            [1, 1, 1, 1],
            parameters["bottleneck"][0],
            height_sharding=False,
        )
        self.bottleneck_2 = Conv([1, 1, 1, 1], parameters["bottleneck"][1], height_sharding=False, reshard=True)

        self.upconv4 = ConvTranspose([2, 2, 0, 0], parameters["upconv4"], auto_shard=True)
        self.dec4_1 = Conv([1, 1, 1, 1], parameters["decoder4"][0], auto_shard=True)
        self.dec4_2 = Conv([1, 1, 1, 1], parameters["decoder4"][1], auto_shard=True)

        self.upconv3 = ConvTranspose([2, 2, 0, 0], parameters["upconv3"], auto_shard=True)
        self.dec3_1 = Conv([1, 1, 1, 1], parameters["decoder3"][0], act_block_h=32)
        self.dec3_2 = Conv([1, 1, 1, 1], parameters["decoder3"][1], height_sharding=False)

        self.upconv2 = ConvTranspose([2, 2, 0, 0], parameters["upconv2"], auto_shard=True)
        self.dec2_1 = Conv([1, 1, 1, 1], parameters["decoder2"][0], act_block_h=32)
        self.dec2_2 = Conv([1, 1, 1, 1], parameters["decoder2"][1], act_block_h=64)

        self.upconv1 = model.upconv1
        self.dec1_1 = ConvSplit([1, 1, 1, 1], parameters["decoder1"][0], auto_shard=True)
        self.dec1_2 = Conv([1, 1, 1, 1], parameters["decoder1"][1], act_block_h=32)

        self.conv = Conv([1, 1, 0, 0], parameters["conv"], activation="")

    def __call__(self, device, input_tensor):
        enc1 = self.enc1_1(device, input_tensor)
        enc1 = self.enc1_2(device, enc1)  # 0.9992

        pool_in = ttnn.reshape(enc1, (1, 1, enc1.shape[0] * enc1.shape[1] * enc1.shape[2], enc1.shape[3]))
        pool_1 = ttnn.max_pool2d(
            input_tensor=pool_in,
            batch_size=1,
            input_h=enc1.shape[1],
            input_w=enc1.shape[2],
            channels=enc1.shape[3],
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )  # 0.9993
        pool_1_out_h, pool_1_out_w = int(enc1.shape[1] / 2), int(enc1.shape[2] / 2)
        pool_1 = ttnn.reshape(pool_1, (enc1.shape[0], pool_1_out_h, pool_1_out_w, enc1.shape[3]))

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
        enc2 = self.enc2_2(device, enc2)  # 0.99806

        pool_in = ttnn.reshape(enc2, (1, 1, enc2.shape[0] * enc2.shape[1] * enc2.shape[2], enc2.shape[3]))
        if pool_in.is_sharded:
            pool_in = ttnn.sharded_to_interleaved(pool_in, ttnn.L1_MEMORY_CONFIG)
        pool_2 = ttnn.max_pool2d(
            input_tensor=pool_in,
            batch_size=1,
            input_h=enc2.shape[1],
            input_w=enc2.shape[2],
            channels=enc2.shape[3],
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )
        pool_2_out_h, pool_2_out_w = int(enc2.shape[1] / 2), int(enc2.shape[2] / 2)
        pool_2 = ttnn.reshape(pool_2, (1, pool_2_out_h, pool_2_out_w, enc2.shape[3]))
        enc2 = ttnn.to_memory_config(enc2, ttnn.DRAM_MEMORY_CONFIG)
        if pool_2.is_sharded:
            pool_2 = ttnn.sharded_to_interleaved(pool_2, ttnn.L1_MEMORY_CONFIG)
        enc3 = self.enc3_1(device, pool_2)
        ttnn.deallocate(pool_2)
        enc3 = self.enc3_2(device, enc3)  # 0.991

        pool_in = ttnn.reshape(enc3, (1, 1, enc3.shape[0] * enc3.shape[1] * enc3.shape[2], enc3.shape[3]))

        pool_3 = ttnn.max_pool2d(
            input_tensor=pool_in,
            batch_size=1,
            input_h=enc3.shape[1],
            input_w=enc3.shape[2],
            channels=enc3.shape[3],
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )  # 0.993
        pool_3_out_h, pool_3_out_w = int(enc3.shape[1] / 2), int(enc3.shape[2] / 2)
        pool_3 = ttnn.reshape(pool_3, (1, pool_3_out_h, pool_3_out_w, enc3.shape[3]))
        enc3 = ttnn.to_memory_config(enc3, ttnn.DRAM_MEMORY_CONFIG)

        if pool_3.is_sharded:
            pool_3 = ttnn.sharded_to_interleaved(pool_3, ttnn.L1_MEMORY_CONFIG)

        enc4 = self.enc4_1(device, pool_3)
        ttnn.deallocate(pool_3)
        enc4 = self.enc4_2(device, enc4)  # 0.9949

        pool_in = ttnn.reshape(enc4, (1, 1, enc4.shape[0] * enc4.shape[1] * enc4.shape[2], enc4.shape[3]))

        pool_4 = ttnn.max_pool2d(
            input_tensor=pool_in,
            batch_size=1,
            input_h=enc4.shape[1],
            input_w=enc4.shape[2],
            channels=enc4.shape[3],
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
        )  # 0.9948
        ttnn.deallocate(pool_in)
        pool_4_out_h, pool_4_out_w = int(enc4.shape[1] / 2), int(enc4.shape[2] / 2)
        pool_4 = ttnn.reshape(pool_4, (1, pool_4_out_h, pool_4_out_w, enc4.shape[3]))
        enc4 = ttnn.to_memory_config(enc4, ttnn.DRAM_MEMORY_CONFIG)

        if pool_4.is_sharded:
            pool_4 = ttnn.sharded_to_interleaved(pool_4, ttnn.L1_MEMORY_CONFIG)
        bottleneck = self.bottleneck_1(device, pool_4)
        ttnn.deallocate(pool_4)
        bottleneck = self.bottleneck_2(device, bottleneck)  # 0.997

        if bottleneck.is_sharded:
            bottleneck = ttnn.sharded_to_interleaved(bottleneck, ttnn.L1_MEMORY_CONFIG)
        dec4 = self.upconv4(device, bottleneck)  # 0.9978

        if dec4.is_sharded:
            dec4 = ttnn.sharded_to_interleaved(dec4, ttnn.L1_MEMORY_CONFIG)
        if enc4.is_sharded:
            enc4 = ttnn.sharded_to_interleaved(enc4, ttnn.L1_MEMORY_CONFIG)
        dec4 = ttnn.concat([dec4, enc4], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.996

        ttnn.deallocate(enc4)

        dec4 = self.dec4_1(device, dec4)
        dec4 = self.dec4_2(device, dec4)  # 0.992
        if dec4.is_sharded:
            dec4 = ttnn.sharded_to_interleaved(dec4, ttnn.L1_MEMORY_CONFIG)
        dec3 = self.upconv3(device, dec4)  # 0.993
        ttnn.deallocate(dec4)

        if dec3.is_sharded:
            dec3 = ttnn.sharded_to_interleaved(dec3, ttnn.L1_MEMORY_CONFIG)
        if enc3.is_sharded:
            enc3 = ttnn.sharded_to_interleaved(enc3, ttnn.L1_MEMORY_CONFIG)

        dec3 = ttnn.concat([dec3, enc3], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.993
        ttnn.deallocate(enc3)

        dec3 = self.dec3_1(device, dec3)  # 0.9892
        dec3 = self.dec3_2(device, dec3)  # 0.98766

        if dec3.is_sharded:
            dec3 = ttnn.sharded_to_interleaved(dec3, ttnn.L1_MEMORY_CONFIG)
        dec2 = self.upconv2(device, dec3)  # 0.992
        ttnn.deallocate(dec3)

        if dec2.is_sharded:
            dec2 = ttnn.sharded_to_interleaved(dec2, ttnn.L1_MEMORY_CONFIG)
        if enc2.is_sharded:
            enc2 = ttnn.sharded_to_interleaved(enc2, ttnn.L1_MEMORY_CONFIG)

        dec2 = ttnn.concat([dec2, enc2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.997

        ttnn.deallocate(enc2)

        dec2 = self.dec2_1(device, dec2)
        dec2 = self.dec2_2(device, dec2)  # 0.995

        dec2 = ttnn_to_torch(dec2)
        dec2 = dec2.permute(0, 3, 1, 2)
        dec2 = dec2.to(torch.float)
        dec1 = self.upconv1(dec2)
        dec1 = dec1.permute(0, 2, 3, 1)
        dec1 = torch_to_ttnn(dec1, device)  # 0.9966

        if enc1.is_sharded:
            enc1 = ttnn.sharded_to_interleaved(enc1, ttnn.L1_MEMORY_CONFIG)
        if dec1.is_sharded:
            dec1 = ttnn.sharded_to_interleaved(dec1, ttnn.L1_MEMORY_CONFIG)

        dec1 = ttnn.concat([dec1, enc1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)  # 0.9988

        ttnn.deallocate(enc1)

        dec1 = ttnn_to_torch(dec1)
        dec1 = dec1.permute(0, 3, 1, 2)
        dec1 = dec1.to(torch.float)
        dec1 = self.model.decoder1[:3](dec1)
        dec1 = dec1.permute(0, 2, 3, 1)
        dec1 = torch_to_ttnn(dec1, device)  # 0.9979
        dec1 = self.dec1_2(device, dec1)  # 0.9968

        ttnn_output = self.conv(device, dec1)  # 0.9770
        ttnn.deallocate(dec1)

        ttnn_output = ttnn.sigmoid_accurate(ttnn_output)  # 0.96

        return ttnn_output
