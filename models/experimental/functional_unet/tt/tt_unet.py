# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
import tt_lib
import tt_lib.fallback_ops
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from tt_lib import tensor as ttl_tensor, device as ttl_device
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


def ttnn_to_torch(input):
    # input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def permute_conv_weights(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    # print("Before permuting weight: ", weight.shape)
    # weight = torch.permute(weight, (2, 3, 0, 1))
    # print("After permuting weight: ", weight.shape)
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


class TtUnet:
    def __init__(
        self,
        device,
        parameters,
        state_dict,
    ) -> None:
        self.enc1_1 = parameters.encoder1_c1
        self.enc1_2 = parameters.encoder1_c2

        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}
        self.pool1 = ttnn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            dtype=ttnn.bfloat16,
            device=device,
            batch_size=1,
            input_height=480,
            input_width=640,
            reader_patterns_cache=self.max_pool_reader_patterns_cache,
            deallocate_activation=False,
            parallel_config_override=max_pool_parallel_config_override,
            channels=32,
        )
        self.enc2_1 = parameters.encoder2_c1
        self.enc2_2 = parameters.encoder2_c2
        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}
        self.pool2 = ttnn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            dtype=ttnn.bfloat16,
            device=device,
            batch_size=1,
            input_height=240,
            input_width=320,
            reader_patterns_cache=self.max_pool_reader_patterns_cache,
            deallocate_activation=False,
            parallel_config_override=max_pool_parallel_config_override,
            channels=64,
        )
        self.enc3_1 = parameters.encoder3_c1
        self.enc3_2 = parameters.encoder3_c2
        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}
        self.pool3 = ttnn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            dtype=ttnn.bfloat16,
            device=device,
            batch_size=1,
            input_height=120,
            input_width=160,
            reader_patterns_cache=self.max_pool_reader_patterns_cache,
            deallocate_activation=False,
            parallel_config_override=max_pool_parallel_config_override,
            channels=128,
        )
        self.enc4_1 = parameters.encoder4_c1
        self.enc4_2 = parameters.encoder4_c2
        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}
        self.pool4 = ttnn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2),
            padding=(0, 0),
            dilation=(1, 1),
            dtype=ttnn.bfloat16,
            device=device,
            batch_size=1,
            input_height=60,
            input_width=80,
            reader_patterns_cache=self.max_pool_reader_patterns_cache,
            deallocate_activation=False,
            parallel_config_override=max_pool_parallel_config_override,
            channels=256,
        )
        self.bnc1_1 = parameters.bottleneck_c1
        # print("parameters",parameters["bottleneck_c2"])

        # parameters["bottleneck_c2"]["bias"]=torch.reshape(parameters["bottleneck_c2"]["bias"],(1,1,1,-1))
        self.bnc1_2_weight = torch_to_tt_tensor_rm(parameters["bottleneck_c2"]["weight"], device, put_on_device=False)
        self.bnc1_2_bias = torch_to_tt_tensor_rm(parameters["bottleneck_c2"]["bias"], device, put_on_device=False)

        # print("bnc1_2_weight",self.bnc1_2_weight.shape)
        # print("self.bnc1_2_bias",self.bnc1_2_bias.shape)

        self.bnc1_2 = tt_lib.fallback_ops.Conv2d(
            weights=self.bnc1_2_weight,
            biases=self.bnc1_2_bias,
            in_channels=30,
            out_channels=40,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv4.weight = torch.nn.Parameter(state_dict["upconv4.weight"])
        self.upconv4.bias = torch.nn.Parameter(state_dict["upconv4.bias"])

        self.dc4_1 = parameters.decoder4_c1
        self.dc4_2 = parameters.decoder4_c2

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3.weight = torch.nn.Parameter(state_dict["upconv3.weight"])
        self.upconv3.bias = torch.nn.Parameter(state_dict["upconv3.bias"])

        self.dc3_1 = parameters.decoder3_c1
        self.dc3_2 = parameters.decoder3_c2

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv2.weight = torch.nn.Parameter(state_dict["upconv2.weight"])
        self.upconv2.bias = torch.nn.Parameter(state_dict["upconv2.bias"])

        self.dc2_1 = parameters.decoder2_c1
        self.dc2_2 = parameters.decoder2_c2

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv1.weight = torch.nn.Parameter(state_dict["upconv1.weight"])
        self.upconv1.bias = torch.nn.Parameter(state_dict["upconv1.bias"])

        # self.dc1_1 = parameters.decoder1_c1
        self.dc1_1_weight = torch_to_tt_tensor_rm(parameters["decoder1_c1"]["weight"], device, put_on_device=False)
        self.dc1_1_bias = torch_to_tt_tensor_rm(parameters["decoder1_c1"]["bias"], device, put_on_device=False)
        self.dc1_1 = tt_lib.fallback_ops.Conv2d(
            weights=self.dc1_1_weight,
            biases=self.dc1_1_bias,
            in_channels=480,
            out_channels=640,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.dc1_2 = parameters.decoder1_c2

        # print("conv parameters: ", parameters.conv)
        # self.conv = parameters.conv
        self.conv_weight = torch_to_tt_tensor_rm(parameters["conv"]["weight"], device, put_on_device=False)
        self.conv_bias = torch_to_tt_tensor_rm(parameters["conv"]["bias"], device, put_on_device=False)
        self.conv = tt_lib.fallback_ops.Conv2d(
            weights=self.conv_weight,
            biases=self.conv_bias,
            in_channels=480,
            out_channels=640,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        )

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.enc1_1.conv.input_sharded_memory_config)
        # print("input_tensor",input_tensor.memory_config())
        output_tensor_enc1_1 = self.enc1_1(input_tensor)
        output_tensor_enc1_2 = self.enc1_2(output_tensor_enc1_1)

        output_tensor_pool_1 = self.pool1(output_tensor_enc1_2)
        output_tensor_enc2_1 = self.enc2_1(output_tensor_pool_1)
        output_tensor_enc2_2 = self.enc2_2(output_tensor_enc2_1)

        output_tensor_pool_2 = self.pool2(output_tensor_enc2_2)
        output_tensor_enc3_1 = self.enc3_1(output_tensor_pool_2)
        output_tensor_enc3_2 = self.enc3_2(output_tensor_enc3_1)

        output_tensor_pool_3 = self.pool3(output_tensor_enc3_2)
        output_tensor_enc4_1 = self.enc4_1(output_tensor_pool_3)

        output_tensor_enc4_2 = self.enc4_2(output_tensor_enc4_1)
        # print("self.enc4_2.conv.input_sharded_memory_config:", self.enc4_2.conv.input_sharded_memory_config)
        # print("output_tensor_pool_4",output_tensor_enc4_2.shape)
        # print("output_tensor_enc4_2",output_tensor_enc4_2.memory_config())
        output_tensor_pool_4 = self.pool4(output_tensor_enc4_2)
        # print("output_tensor_enc4_2",output_tensor_enc4_2.memory_config())

        output_tensor_bnc1_1 = self.bnc1_1(output_tensor_pool_4)
        output_tensor_bnc1_1 = ttnn_to_torch(output_tensor_bnc1_1)

        # print("device: ",device)

        output_tensor_bnc1_1 = output_tensor_bnc1_1.reshape(1, 30, 40, 512)
        output_tensor_bnc1_1 = torch.permute(output_tensor_bnc1_1, (0, 3, 1, 2))
        output_tensor_bnc1_1 = torch_to_tt_tensor_rm(output_tensor_bnc1_1, device, put_on_device=True)
        output_tensor_bnc1_2 = self.bnc1_2(output_tensor_bnc1_1)
        output_tensor_bnc1_2 = tt_to_torch_tensor(output_tensor_bnc1_2)
        output_tensor_bnc1_2 = ttnn.from_torch(
            output_tensor_bnc1_2, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        output_tensor_bnc1_2 = ttnn.relu(output_tensor_bnc1_2)
        output_tensor_bnc1_2 = ttnn_to_torch(output_tensor_bnc1_2)
        output_tensor_bnc1_2 = output_tensor_bnc1_2.to(dtype=torch.float)

        # print("BNC1_2 done! ")
        output_tensor_dc_4 = self.upconv4(output_tensor_bnc1_2)

        output_tensor_dc_4 = torch.permute(output_tensor_dc_4, (0, 2, 3, 1))

        output_tensor_dc_4 = output_tensor_dc_4.reshape(
            output_tensor_dc_4.shape[0],
            1,
            output_tensor_dc_4.shape[1] * output_tensor_dc_4.shape[2],
            output_tensor_dc_4.shape[3],
        )
        output_tensor_dc_4 = ttnn.from_torch(
            output_tensor_dc_4, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        output_tensor_dc_4 = ttnn.concat([output_tensor_dc_4, output_tensor_enc4_2], dim=3)
        ttnn.deallocate(output_tensor_enc4_2)
        output_tensor_dc_4 = tt_lib.tensor.interleaved_to_sharded(
            output_tensor_dc_4, self.dc4_1.conv.input_sharded_memory_config
        )
        output_tensor_dc4_1 = self.dc4_1(output_tensor_dc_4)
        output_tensor_dc4_2 = self.dc4_2(output_tensor_dc4_1)
        # output_tensor_dc4_2=ttnn.to_memory_config(output_tensor_dc4_2, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        output_tensor_dc4_2_temp = ttnn_to_torch(output_tensor_dc4_2)
        ttnn.deallocate(output_tensor_dc4_2)
        output_tensor_dc4_2 = output_tensor_dc4_2_temp.reshape(1, 60, 80, 256)
        output_tensor_dc4_2 = torch.permute(output_tensor_dc4_2, (0, 3, 1, 2))
        output_tensor_dc_3 = self.upconv3(output_tensor_dc4_2)
        output_tensor_dc_3 = torch.permute(output_tensor_dc_3, (0, 2, 3, 1))

        output_tensor_dc_3 = output_tensor_dc_3.reshape(
            output_tensor_dc_3.shape[0],
            1,
            output_tensor_dc_3.shape[1] * output_tensor_dc_3.shape[2],
            output_tensor_dc_3.shape[3],
        )
        output_tensor_dc_3 = ttnn.from_torch(
            output_tensor_dc_3, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        output_tensor_dc_3 = ttnn.concat([output_tensor_dc_3, output_tensor_enc3_2], dim=3)
        ttnn.deallocate(output_tensor_enc3_2)
        output_tensor_dc_3 = tt_lib.tensor.interleaved_to_sharded(
            output_tensor_dc_3, self.dc3_1.conv.input_sharded_memory_config
        )
        output_tensor_dc3_1 = self.dc3_1(output_tensor_dc_3)
        output_tensor_dc3_2 = self.dc3_2(output_tensor_dc3_1)
        # ttnn.deallocate(output_tensor_dc3_1)

        output_tensor_dc3_2 = ttnn_to_torch(output_tensor_dc3_2)
        output_tensor_dc3_2 = output_tensor_dc3_2.reshape(1, 120, 160, 128)
        output_tensor_dc3_2 = torch.permute(output_tensor_dc3_2, (0, 3, 1, 2))
        output_tensor_dc_2 = self.upconv2(output_tensor_dc3_2)
        output_tensor_dc_2 = torch.permute(output_tensor_dc_2, (0, 2, 3, 1))

        output_tensor_dc_2 = output_tensor_dc_2.reshape(
            output_tensor_dc_2.shape[0],
            1,
            output_tensor_dc_2.shape[1] * output_tensor_dc_2.shape[2],
            output_tensor_dc_2.shape[3],
        )
        output_tensor_dc_2 = ttnn.from_torch(
            output_tensor_dc_2, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        output_tensor_dc_2 = ttnn.concat([output_tensor_dc_2, output_tensor_enc2_2], dim=3)
        ttnn.deallocate(output_tensor_enc2_2)
        output_tensor_dc_2 = tt_lib.tensor.interleaved_to_sharded(
            output_tensor_dc_2, self.dc2_1.conv.input_sharded_memory_config
        )
        output_tensor_dc2_1 = self.dc2_1(output_tensor_dc_2)
        output_tensor_dc2_2 = self.dc2_2(output_tensor_dc2_1)

        output_tensor_dc2_2_temp = ttnn_to_torch(output_tensor_dc2_2)
        ttnn.deallocate(output_tensor_dc2_2)
        output_tensor_dc2_2 = output_tensor_dc2_2_temp.reshape(1, 240, 320, 64)
        output_tensor_dc2_2 = torch.permute(output_tensor_dc2_2, (0, 3, 1, 2))
        output_tensor_dc_1 = self.upconv1(output_tensor_dc2_2)
        output_tensor_dc_1 = torch.permute(output_tensor_dc_1, (0, 2, 3, 1))

        output_tensor_dc_1 = output_tensor_dc_1.reshape(
            output_tensor_dc_1.shape[0],
            1,
            output_tensor_dc_1.shape[1] * output_tensor_dc_1.shape[2],
            output_tensor_dc_1.shape[3],
        )
        output_tensor_dc_1 = ttnn.from_torch(
            output_tensor_dc_1, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

        output_tensor_dc_1 = ttnn.concat([output_tensor_dc_1, output_tensor_enc1_2], dim=3)
        ttnn.deallocate(output_tensor_enc1_2)
        # output_tensor_dc_1=ttnn.to_memory_config(output_tensor_dc_1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # output_tensor_dc_1 = tt_lib.tensor.interleaved_to_sharded(output_tensor_dc_1, self.dc1_1.conv.input_sharded_memory_config)

        output_tensor_dc1_1 = ttnn.to_torch(output_tensor_dc_1)
        output_tensor_dc1_1 = output_tensor_dc1_1.reshape(1, 480, 640, 64)
        output_tensor_dc1_1 = torch.permute(output_tensor_dc1_1, (0, 3, 1, 2))
        # print("shape of output_tensor_dc1_1:",output_tensor_dc1_1.shape)
        output_tensor_dc1_1 = torch_to_tt_tensor_rm(output_tensor_dc1_1, device, put_on_device=True)
        output_tensor_dc1_1 = self.dc1_1(output_tensor_dc1_1)
        output_tensor_dc1_1 = tt_to_torch_tensor(output_tensor_dc1_1)
        # output_tensor_dc1_1=output_tensor_dc1_1.to(dtype=torch.float)
        output_tensor_dc1_1 = torch.permute(output_tensor_dc1_1, (0, 2, 3, 1))

        output_tensor_dc1_1 = output_tensor_dc1_1.reshape(
            output_tensor_dc1_1.shape[0],
            1,
            output_tensor_dc1_1.shape[1] * output_tensor_dc1_1.shape[2],
            output_tensor_dc1_1.shape[3],
        )
        output_tensor_dc1_1 = ttnn.from_torch(
            output_tensor_dc1_1, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        output_tensor_dc1_1 = ttnn.relu(output_tensor_dc1_1)
        output_tensor_dc1_1 = tt_lib.tensor.interleaved_to_sharded(
            output_tensor_dc1_1, self.dc1_2.conv.input_sharded_memory_config
        )

        output_tensor_dc1_2 = self.dc1_2(output_tensor_dc1_1)
        # print("shape after dc1_2: ", output_tensor_dc1_2.shape)

        output_tensor_dc1_2 = ttnn.to_torch(output_tensor_dc1_2)
        output_tensor_dc1_2 = output_tensor_dc1_2.reshape(1, 480, 640, 32)
        output_tensor_dc1_2 = torch.permute(output_tensor_dc1_2, (0, 3, 1, 2))
        output_tensor_dc1_2 = torch_to_tt_tensor_rm(output_tensor_dc1_2, device, put_on_device=True)
        output_tensor_conv = self.conv(output_tensor_dc1_2)
        # print("shape after last conv: ", output_tensor_conv.shape)
        output_tensor_conv = tt_to_torch_tensor(output_tensor_conv)
        output_tensor_conv = torch.permute(output_tensor_conv, (0, 2, 3, 1))

        output_tensor_conv = output_tensor_conv.reshape(
            output_tensor_conv.shape[0],
            1,
            output_tensor_conv.shape[1] * output_tensor_conv.shape[2],
            output_tensor_conv.shape[3],
        )
        # output_tensor_conv = ttnn.from_torch(
        #    output_tensor_conv, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        # )

        # output_tensor_conv = tt_lib.tensor.sharded_to_interleaved(output_tensor_conv, ttnn.L1_MEMORY_CONFIG)
        # output_tensor = ttnn.sigmoid(output_tensor_conv)
        output_tensor = torch.sigmoid(output_tensor_conv)
        output_tensor = ttnn.from_torch(output_tensor, device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        return ttnn.from_device(output_tensor)
