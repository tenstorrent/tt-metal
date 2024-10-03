# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.functional_vanilla_unet.ttnn.common import Conv


def torch_to_ttnn(input, device=None, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    # input = ttnn.to_layout(input, layout)
    # input = ttnn.to_device(input, device)
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
            [1, 480, 640, 3],
            [1, 1, 1, 1],
            parameters["encoder1"][0],
            act_block_h=64,
        )
        self.enc1_2 = Conv(
            [1, 480, 640, 32],
            [1, 1, 1, 1],
            parameters["encoder1"][1],
            act_block_h=32,
        )

        self.enc2_1 = Conv([1, 240, 320, 32], [1, 1, 1, 1], parameters["encoder2"][0], reshard=False)
        self.enc2_2 = Conv([1, 240, 320, 64], [1, 1, 1, 1], parameters["encoder2"][1], act_block_h=64)

        self.enc3_1 = Conv([1, 120, 160, 64], [1, 1, 1, 1], parameters["encoder3"][0])
        self.enc3_2 = Conv([1, 120, 160, 128], [1, 1, 1, 1], parameters["encoder3"][1], act_block_h=32)

        self.enc4_1 = Conv(
            [1, 60, 80, 128], [1, 1, 1, 1], parameters["encoder4"][0], height_sharding=False, act_block_h=1 * 32
        )
        self.enc4_2 = Conv([1, 60, 80, 256], [1, 1, 1, 1], parameters["encoder4"][1], act_block_h=32)

        self.bottleneck_1 = Conv([1, 30, 40, 256], [1, 1, 1, 1], parameters["bottleneck"][0], height_sharding=False)
        self.bottleneck_2 = Conv(
            [1, 30, 40, 512], [1, 1, 1, 1], parameters["bottleneck"][1], height_sharding=False, act_block_h=32
        )

        self.upconv4 = model.upconv4
        self.dec4_1 = Conv(
            [1, 60, 80, 512], [1, 1, 1, 1], parameters["decoder4"][0], height_sharding=False, act_block_h=32
        )
        self.dec4_2 = Conv([1, 60, 80, 256], [1, 1, 1, 1], parameters["decoder4"][1], act_block_h=32)

        self.upconv3 = model.upconv3
        self.dec3_1 = Conv(
            [1, 120, 160, 256], [1, 1, 1, 1], parameters["decoder3"][0], height_sharding=False, act_block_h=1 * 32
        )
        self.dec3_2 = Conv(
            [1, 120, 160, 128], [1, 1, 1, 1], parameters["decoder3"][1], height_sharding=False, act_block_h=1 * 32
        )

        # self.upconv2 = model.upconv2
        # self.dec2_1 = Conv(
        #     [1, 240, 320, 128], [1, 1, 1, 1], parameters["decoder2"][0], height_sharding=False, act_block_h=1 * 32
        # )
        # self.dec2_2 = Conv([1, 240, 320, 64], [1, 1, 1, 1], parameters["decoder2"][1], act_block_h=64)

        # self.upconv1 = model.upconv1
        # self.dec1_1 = Conv([1, 1, 1, 1], parameters["decoder1"][0], reshard=True)
        # self.dec1_2 = Conv([1, 1, 1, 1], parameters["decoder1"][1], act_block_h=32)

        # self.conv = Conv([1, 1, 0, 0], parameters["conv"], activation="")

    def __call__(self, device, input_tensor):
        enc1 = self.enc1_1(device, input_tensor)
        enc1 = self.enc1_2(device, enc1)

        pool_1 = ttnn.max_pool2d(
            input_tensor=enc1,
            batch_size=1,
            input_h=480,
            input_w=640,
            channels=32,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            device=device,
        )
        pool_1 = ttnn.reshape(pool_1, (1, 240, 320, 32))
        pool_1 = ttnn.sharded_to_interleaved(
            pool_1,
            ttnn.L1_MEMORY_CONFIG,
        )
        pool_1 = ttnn.from_device(pool_1)

        enc2 = self.enc2_1(device, pool_1)
        enc2 = self.enc2_2(device, enc2)
        pool_2 = ttnn.max_pool2d(
            input_tensor=enc2,
            batch_size=1,
            input_h=240,
            input_w=320,
            channels=64,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            device=device,
        )
        pool_2 = ttnn.reshape(pool_2, (1, 120, 160, 64))
        pool_2 = ttnn.sharded_to_interleaved(
            pool_2,
            ttnn.L1_MEMORY_CONFIG,
        )
        pool_2 = ttnn.from_device(pool_2)

        enc3 = self.enc3_1(device, pool_2)
        enc3 = self.enc3_2(device, enc3)
        pool_3 = ttnn.max_pool2d(
            input_tensor=enc3,
            batch_size=1,
            input_h=120,
            input_w=160,
            channels=128,
            kernel_size=[2, 2],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            device=device,
        )
        # pool_3 = ttnn.reshape(pool_3, (1, 60, 80, 128))
        pool_3 = ttnn.sharded_to_interleaved(
            pool_3,
            ttnn.L1_MEMORY_CONFIG,
        )
        # pool_3 = ttnn.from_device(pool_3)

        enc4 = self.enc4_1(device, pool_3)
        enc4 = self.enc4_2(device, enc4)

        enc4_duplicate = enc4
        enc4_duplicate = ttnn.sharded_to_interleaved(
            enc4_duplicate,
            ttnn.L1_MEMORY_CONFIG,
        )
        enc4_duplicate = ttnn_to_torch(enc4_duplicate)

        # pool4 = ttnn.max_pool2d(
        #     input_tensor=enc4,
        #     batch_size=1,
        #     input_h=60,
        #     input_w=80,
        #     channels=256,
        #     kernel_size=[2, 2],
        #     stride=[2, 2],
        #     padding=[0, 0],
        #     dilation=[1, 1],
        #     device=device,
        # )

        enc4_duplicate = enc4_duplicate.reshape(1, 60, 80, 256)
        enc4_duplicate = enc4_duplicate.permute(0, 3, 1, 2)
        pool_4 = self.model.pool4(enc4_duplicate)
        pool_4 = pool_4.permute(0, 2, 3, 1)
        pool_4 = torch_to_ttnn(pool_4, device=device)

        bottleneck = self.bottleneck_1(device, pool_4)
        bottleneck = self.bottleneck_2(device, bottleneck)

        bottleneck = ttnn.sharded_to_interleaved(
            bottleneck,
            ttnn.L1_MEMORY_CONFIG,
        )
        bottleneck = ttnn_to_torch(bottleneck)
        bottleneck = bottleneck.reshape(1, 30, 40, 512)
        bottleneck = bottleneck.permute(0, 3, 1, 2)
        bottleneck = bottleneck.to(torch.float)
        dec4 = self.upconv4(bottleneck)
        dec4 = dec4.permute(0, 2, 3, 1)
        dec4 = torch_to_ttnn(dec4)

        dec4 = ttnn.to_layout(dec4, layout=ttnn.TILE_LAYOUT)
        dec4 = ttnn.to_device(dec4, device=device)

        # enc4 = ttnn.to_layout(enc4, layout=ttnn.TILE_LAYOUT)
        # enc4 = ttnn.to_device(enc4, device=device)
        enc4 = ttnn.sharded_to_interleaved(
            enc4,
            ttnn.L1_MEMORY_CONFIG,
        )
        enc4 = ttnn.to_layout(enc4, layout=ttnn.ROW_MAJOR_LAYOUT)
        enc4 = ttnn.reshape(enc4, (1, 60, 80, 256))
        enc4 = ttnn.to_layout(enc4, layout=ttnn.TILE_LAYOUT)

        dec4 = ttnn.concat([dec4, enc4], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(enc4)

        # dec4 = ttnn.from_device(dec4)
        # dec4 = ttnn.to_layout(dec4, layout=ttnn.ROW_MAJOR_LAYOUT)
        # # PCC is 0.99 until this point of dec4

        dec4 = self.dec4_1(device, dec4)
        dec4 = self.dec4_2(device, dec4)

        dec4 = ttnn.sharded_to_interleaved(
            dec4,
            ttnn.L1_MEMORY_CONFIG,
        )
        dec4 = ttnn_to_torch(dec4)
        dec4 = dec4.reshape(1, 60, 80, 256)
        dec4 = dec4.permute(0, 3, 1, 2)
        dec4 = dec4.to(torch.float)
        dec3 = self.upconv3(dec4)
        dec3 = dec3.permute(0, 2, 3, 1)
        dec3 = torch_to_ttnn(dec3)

        enc3 = ttnn.sharded_to_interleaved(
            enc3,
            ttnn.L1_MEMORY_CONFIG,
        )
        enc3 = ttnn.to_layout(enc3, layout=ttnn.ROW_MAJOR_LAYOUT)
        enc3 = ttnn.reshape(enc3, (1, 120, 160, 128))
        enc3 = ttnn.to_layout(enc3, layout=ttnn.TILE_LAYOUT)

        dec3 = ttnn.to_layout(dec3, layout=ttnn.TILE_LAYOUT)
        dec3 = ttnn.to_device(dec3, device=device)

        dec3 = ttnn.concat([dec3, enc3], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(enc3)
        # dec3 = ttnn.from_device(dec3)
        # dec3 = ttnn.to_layout(dec3, layout=ttnn.ROW_MAJOR_LAYOUT)
        # print("-----dec3--------", dec3.shape)
        # dec3 = self.dec3_1(device, dec3)
        # dec3 = self.dec3_2(device, dec3)

        # dec3 = ttnn_to_torch(dec3)
        # dec3 = dec3.reshape(1, 120, 160, 128)
        # dec3 = dec3.permute(0, 3, 1, 2)
        # dec3 = dec3.to(torch.float)
        # dec2 = self.upconv2(dec3)
        # dec2 = dec2.permute(0, 2, 3, 1)
        # dec2 = torch_to_ttnn(dec2)

        # enc2 = ttnn.sharded_to_interleaved(
        #     enc2,
        #     ttnn.L1_MEMORY_CONFIG,
        # )
        # enc2 = ttnn.to_layout(enc2, layout=ttnn.ROW_MAJOR_LAYOUT)
        # enc2 = ttnn.reshape(enc2, (1, 240, 320, 64))
        # enc2 = ttnn.to_layout(enc2, layout=ttnn.TILE_LAYOUT)

        # dec2 = ttnn.to_layout(dec2, layout=ttnn.TILE_LAYOUT)
        # dec2 = ttnn.to_device(dec2, device=device)

        # dec2 = ttnn.concat([dec2, enc2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        # ttnn.deallocate(enc2)
        # dec2 = ttnn.from_device(dec2)
        # dec2 = ttnn.to_layout(dec2, layout=ttnn.ROW_MAJOR_LAYOUT)

        # dec2 = self.dec2_1(device, dec2)
        # dec2 = self.dec2_2(device, dec2)

        # dec2 = ttnn_to_torch(dec2)
        # dec2 = dec2.permute(0, 3, 1, 2)
        # dec2 = dec2.to(torch.float)
        # dec1 = self.upconv1(dec2)
        # dec1 = dec1.permute(0, 2, 3, 1)
        # dec1 = torch_to_ttnn(dec1)

        # enc1 = ttnn.to_layout(enc1, ttnn.TILE_LAYOUT)

        # dec1 = ttnn.to_layout(dec1, layout=ttnn.TILE_LAYOUT)
        # dec1 = ttnn.to_device(dec1, device=device)
        # dec1 = ttnn.concat([dec1, enc1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        # dec1 = ttnn.from_device(dec1)
        # dec1 = ttnn.to_layout(dec1, layout=ttnn.ROW_MAJOR_LAYOUT)

        # dec1 = ttnn_to_torch(dec1)
        # dec1 = dec1.permute(0, 3, 1, 2)
        # dec1 = dec1.to(torch.float)
        # dec1 = self.model.decoder1[:3](dec1)
        # dec1 = dec1.permute(0, 2, 3, 1)
        # dec1 = torch_to_ttnn(dec1)
        # dec1 = self.dec1_2(device, dec1)

        # output_tensor = self.conv(device, dec1)
        # output_tensor = torch.sigmoid(output_tensor)  # If kept in ttnn gives 0.0 , Will check and report it.
        return dec3
