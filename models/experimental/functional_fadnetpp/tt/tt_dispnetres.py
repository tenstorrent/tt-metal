# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.experimental.functional_fadnetpp.tt.ttnn_resblock import TtResBlock
import tt_lib
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


class TtDispNetRes(nn.Module):
    def __init__(
        self, parameters, device, in_planes, resBlock=True, input_channel=3, encoder_ratio=16, decoder_ratio=16
    ):
        super(TtDispNetRes, self).__init__()
        self.input_channel = input_channel
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio
        self.resBlock = resBlock
        self.res_scale = 7
        self.parameters = parameters
        self.device = device

        # self.conv1 = parameters.conv1
        self.conv1_weight = torch_to_tt_tensor_rm(parameters["conv1"]["weight"], device, put_on_device=False)
        self.conv1_bias = torch_to_tt_tensor_rm(parameters["conv1"]["bias"], device, put_on_device=False)
        self.conv1 = tt_lib.fallback_ops.Conv2d(
            weights=self.conv1_weight,
            biases=self.conv1_bias,
            in_channels=11,
            out_channels=32,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=True,
        )
        # self.conv1 = nn.Conv2d(11, 32, kernel_size=7, stride=2, padding=(7 - 1) // 2, bias=True)

        # self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        if resBlock:
            self.conv2 = TtResBlock(parameters.conv2, self.basicE, self.basicE * 2, stride=2)
            self.conv3 = TtResBlock(parameters.conv3, self.basicE * 2, self.basicE * 4, stride=2)
            self.conv3_1 = TtResBlock(parameters.conv3_1, self.basicE * 4, self.basicE * 4)
            self.conv4 = TtResBlock(parameters.conv4, self.basicE * 4, self.basicE * 8, stride=2)
            self.conv4_1 = TtResBlock(parameters.conv4_1, self.basicE * 8, self.basicE * 8)

            self.conv5_resblock1_weight = torch_to_tt_tensor_rm(
                parameters["conv5"]["resblock_1_conv1"]["weight"], device, put_on_device=False
            )
            self.conv5_resblock1_bias = torch_to_tt_tensor_rm(
                parameters["conv5"]["resblock_1_conv1"]["bias"], device, put_on_device=False
            )
            self.conv5_resblock1 = tt_lib.fallback_ops.Conv2d(
                weights=self.conv5_resblock1_weight,
                biases=self.conv5_resblock1_bias,
                in_channels=256,
                out_channels=512,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )
            self.conv5_resblock2 = parameters.conv5.resblock_2_conv2
            self.conv5_resblock_sc = parameters.conv5.resblock_sc_conv
            self.conv5_1 = TtResBlock(parameters.conv5_1, self.basicE * 16, self.basicE * 16)
            self.conv6 = TtResBlock(parameters.conv6, self.basicE * 16, self.basicE * 32, stride=2)
            self.conv6_1 = TtResBlock(parameters.conv6_1, self.basicE * 32, self.basicE * 32)
        else:
            self.conv2 = parameters.conv2
            self.conv3 = parameters.conv3
            self.conv3_1 = parameters.conv3_1
            self.conv4 = parameters.conv4
            self.conv4_1 = parameters.conv4_1
            self.conv5 = parameters.conv5
            self.conv5_1 = parameters.conv5_1
            self.conv6 = parameters.conv6
            self.conv6_1 = parameters.conv6_1
        self.pred_res6 = parameters.pred_res6

        self.iconv5 = nn.ConvTranspose2d((self.basicD + self.basicE) * 16 + 1, self.basicD * 16, 3, 1, 1)
        self.iconv5.weight = parameters.iconv5["weight"]
        self.iconv5.bias = parameters.iconv5["bias"]
        self.iconv4 = nn.ConvTranspose2d((self.basicD + self.basicE) * 8 + 1, self.basicD * 8, 3, 1, 1)
        self.iconv4.weight = parameters.iconv4["weight"]
        self.iconv4.bias = parameters.iconv4["bias"]
        self.iconv3 = nn.ConvTranspose2d((self.basicD + self.basicE) * 4 + 1, self.basicD * 4, 3, 1, 1)
        self.iconv3.weight = parameters.iconv3["weight"]
        self.iconv3.bias = parameters.iconv3["bias"]
        self.iconv2 = nn.ConvTranspose2d((self.basicD + self.basicE) * 2 + 1, self.basicD * 2, 3, 1, 1)
        self.iconv2.weight = parameters.iconv2["weight"]
        self.iconv2.bias = parameters.iconv2["bias"]
        self.iconv1 = nn.ConvTranspose2d((self.basicD + self.basicE) * 1 + 1, self.basicD, 3, 1, 1)
        self.iconv1.weight = parameters.iconv1["weight"]
        self.iconv1.bias = parameters.iconv1["bias"]
        self.iconv0 = nn.ConvTranspose2d(self.basicD + self.input_channel + 1, self.basicD, 3, 1, 1)
        self.iconv0.weight = parameters.iconv0["weight"]
        self.iconv0.bias = parameters.iconv0["bias"]

        self.upconv5 = nn.ConvTranspose2d(
            self.basicE * 32, self.basicD * 16, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upconv5.weight = parameters.upconv5["weight"]
        self.upconv4 = nn.ConvTranspose2d(
            self.basicD * 16, self.basicD * 8, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upconv4.weight = parameters.upconv4["weight"]
        self.upconv3 = nn.ConvTranspose2d(
            self.basicD * 8, self.basicD * 4, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upconv3.weight = parameters.upconv3["weight"]
        self.upconv2 = nn.ConvTranspose2d(
            self.basicD * 4, self.basicD * 2, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upconv2.weight = parameters.upconv2["weight"]
        self.upconv1 = nn.ConvTranspose2d(self.basicD * 2, self.basicD, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv1.weight = parameters.upconv1["weight"]
        self.upconv0 = nn.ConvTranspose2d(self.basicD, self.basicD, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv0.weight = parameters.upconv0["weight"]

        # self.pred_res5 = parameters.pred_res5
        self.pred_res5_weight = torch_to_tt_tensor_rm(parameters["pred_res5"]["weight"], device, put_on_device=False)

        self.pred_res4 = parameters.pred_res4
        self.pred_res3 = parameters.pred_res3
        self.pred_res2 = parameters.pred_res2
        self.pred_res1 = parameters.pred_res1
        # self.pred_res0 = parameters.pred_res0
        self.pred_res0_weight = torch_to_tt_tensor_rm(parameters["pred_res0"]["weight"], device, put_on_device=False)

        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow6to5.weight = parameters.upflow6to5["weight"]
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow5to4.weight = parameters.upflow5to4["weight"]
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow4to3.weight = parameters.upflow4to3["weight"]
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow3to2.weight = parameters.upflow3to2["weight"]
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow2to1.weight = parameters.upflow2to1["weight"]
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upflow1to0.weight = parameters.upflow1to0["weight"]

    def __call__(self, device, input_tensor, flows, get_features=False):
        input_features = input_tensor
        if type(flows) == tuple:
            base_flow = flows
        else:
            base_flow = [fallback_ops.interpolate(flows, scale_factor=2 ** (-i)) for i in range(7)]

        # input_features = tt_lib.tensor.interleaved_to_sharded(
        #     input_features, self.conv1.conv.input_sharded_memory_config
        # )
        input_features = ttnn.to_torch(input_features)
        input_features = input_features.reshape(1, 960, 576, 11)
        input_features = torch.permute(input_features, (0, 3, 1, 2))
        input_features = torch_to_tt_tensor_rm(input_features, device, put_on_device=True)
        conv1 = self.conv1(input_features)
        conv1 = tt_to_torch_tensor(conv1)

        conv1 = torch.permute(conv1, (0, 2, 3, 1))
        conv1 = conv1.reshape(conv1.shape[0], 1, conv1.shape[1] * conv1.shape[2], conv1.shape[3])
        conv1 = ttnn.from_torch(conv1, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        conv1 = ttnn.leaky_relu(conv1, slope=0.1, memory_config=ttnn.L1_MEMORY_CONFIG)

        input_features = tt_to_torch_tensor(input_features)
        input_features = torch.permute(input_features, (0, 2, 3, 1))
        input_features = input_features.reshape(
            input_features.shape[0], 1, input_features.shape[1] * input_features.shape[2], input_features.shape[3]
        )
        input_features = ttnn.from_torch(
            input_features, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )

        if self.resBlock:
            conv2 = self.conv2(device, conv1)
            conv2 = ttnn.to_device(conv2, device)
            conv3a = self.conv3(device, conv2)
            conv3b = self.conv3_1(device, conv3a)
            conv4a = self.conv4(device, conv3b)
            conv4b = self.conv4_1(device, conv4a)

            conv4b = conv4b.to(device)
            conv4b = tt_lib.tensor.interleaved_to_sharded(
                conv4b, self.conv5_resblock_sc.conv.input_sharded_memory_config
            )
            conv5_residual = conv4b
            conv5_residual = self.conv5_resblock_sc(conv4b)
            conv5_output_tensor_h = conv4b

            conv5_output_tensor_h = ttnn.to_torch(conv5_output_tensor_h)
            conv5_output_tensor_h = conv5_output_tensor_h.reshape(1, 60, 36, 256)
            conv5_output_tensor_h = torch.permute(conv5_output_tensor_h, (0, 3, 1, 2))
            conv5_output_tensor_h = torch_to_tt_tensor_rm(conv5_output_tensor_h, device, put_on_device=True)
            conv5_res1 = self.conv5_resblock1(conv5_output_tensor_h)
            conv5_res1 = tt_to_torch_tensor(conv5_res1)
            conv5_res1 = torch.permute(conv5_res1, (0, 2, 3, 1))
            conv5_res1 = conv5_res1.reshape(
                conv5_res1.shape[0], 1, conv5_res1.shape[1] * conv5_res1.shape[2], conv5_res1.shape[3]
            )
            conv5_res1 = ttnn.from_torch(conv5_res1, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            )
            conv5_residual = tt_lib.tensor.sharded_to_interleaved(conv5_residual, memory_config)
            conv5_res1 = tt_lib.tensor.interleaved_to_sharded(
                conv5_res1, self.conv5_resblock2.conv.input_sharded_memory_config
            )
            conv5_residual = tt_lib.tensor.interleaved_to_sharded(
                conv5_residual, self.conv5_resblock2.conv.input_sharded_memory_config
            )
            conv5_output_tensor_h = self.conv5_resblock2(conv5_res1)
            conv5_output_tensor_h += conv5_residual
            conv5a = ttnn.relu(conv5_output_tensor_h)

            conv5b = self.conv5_1(device, conv5a)
            conv6a = self.conv6(device, conv5b)
            conv6b = self.conv6_1(device, conv6a)
        else:
            conv2 = self.conv2(conv1)
            conv2 = ttnn.to_torch(conv2)
            conv2 = self.leaky_relu(conv2)
            conv2 = ttnn.from_torch(conv2, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            conv3a = self.conv3(conv2)
            conv3a = ttnn.to_torch(conv3a)
            conv3a = self.leaky_relu(conv3a)
            conv3a = ttnn.from_torch(conv3a, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            conv3b = self.conv3_1(conv3a)
            conv3b = ttnn.to_torch(conv3b)
            conv3b = self.leaky_relu(conv3b)
            conv3b = ttnn.from_torch(conv3b, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            conv4a = self.conv4(conv3b)
            conv4a = ttnn.to_torch(conv4a)
            conv4a = self.leaky_relu(conv4a)
            conv4a = ttnn.from_torch(conv4a, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            conv4b = self.conv4_1(conv4a)
            conv4b = ttnn.to_torch(conv4b)
            conv4b = self.leaky_relu(conv4b)
            conv4b = ttnn.from_torch(conv4b, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            conv5a = self.conv5(conv4b)
            conv5a = ttnn.to_torch(conv5a)
            conv5a = self.leaky_relu(conv5a)
            conv5a = ttnn.from_torch(conv5a, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            conv5b = self.conv5_1(conv5a)
            conv5b = ttnn.to_torch(conv5b)
            conv5b = self.leaky_relu(conv5b)
            conv5b = ttnn.from_torch(conv5b, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            conv6a = self.conv6(conv5b)
            conv6a = ttnn.to_torch(conv6a)
            conv6a = self.leaky_relu(conv6a)
            conv6a = ttnn.from_torch(conv6a, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

            conv6b = self.conv6_1(conv6a)
            conv6b = ttnn.to_torch(conv6b)
            conv6b = self.leaky_relu(conv6b)
            conv6b = ttnn.from_torch(conv6b, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        conv6b = conv6b.to(device)
        conv6b = tt_lib.tensor.interleaved_to_sharded(conv6b, self.pred_res6.conv.input_sharded_memory_config)
        pr6_res = self.pred_res6(conv6b)
        pr6 = pr6_res + base_flow[6]
        ttnn.deallocate(pr6_res)

        conv6b = ttnn.to_torch(conv6b)
        conv6b = conv6b.reshape(1, 15, 9, 1024)
        conv6b = torch.permute(conv6b, (0, 3, 1, 2))

        pr6 = ttnn.to_torch(pr6)
        pr6 = pr6.reshape(1, 15, 9, 1)
        pr6 = torch.permute(pr6, (0, 3, 1, 2))

        conv5b = ttnn.to_torch(conv5b)
        conv5b = conv5b.reshape(1, 30, 18, 512)
        conv5b = torch.permute(conv5b, (0, 3, 1, 2))

        upconv5 = self.upconv5(conv6b)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        conv5b = torch.permute(conv5b, (0, 2, 3, 1))
        conv5b = conv5b.reshape(conv5b.shape[0], 1, conv5b.shape[1] * conv5b.shape[2], conv5b.shape[3])
        conv5b = ttnn.from_torch(conv5b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(conv5b)
        conv6b = torch.permute(conv6b, (0, 2, 3, 1))
        conv6b = conv6b.reshape(conv6b.shape[0], 1, conv6b.shape[1] * conv6b.shape[2], conv6b.shape[3])
        conv6b = ttnn.from_torch(conv6b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(conv6b)

        iconv5 = torch_to_tt_tensor_rm(iconv5, device, put_on_device=False)
        pr5_res = tt_lib.fallback_ops.conv2d(input=iconv5, weight=self.pred_res5_weight, stride=(1, 1), padding=(1, 1))
        pr5_res = tt_to_torch_tensor(pr5_res)
        pr5_res = torch.permute(pr5_res, (0, 2, 3, 1))
        pr5_res = pr5_res.reshape(pr5_res.shape[0], 1, pr5_res.shape[1] * pr5_res.shape[2], pr5_res.shape[3])
        pr5_res = ttnn.from_torch(pr5_res, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        pr5 = pr5_res + base_flow[5]
        iconv5 = ttnn.to_torch(iconv5)
        iconv5 = iconv5.reshape(1, 30, 18, 512)
        iconv5 = torch.permute(iconv5, (0, 3, 1, 2))
        iconv5 = iconv5.to(torch.float32)

        pr5 = ttnn.to_torch(pr5)
        pr5 = pr5.reshape(1, 30, 18, 1)
        pr5 = torch.permute(pr5, (0, 3, 1, 2))

        conv4b = ttnn.to_torch(conv4b)
        conv4b = conv4b.reshape(1, 60, 36, 256)
        conv4b = torch.permute(conv4b, (0, 3, 1, 2))

        upconv4 = self.upconv4(iconv5)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)

        iconv5 = torch.permute(iconv5, (0, 2, 3, 1))
        iconv5 = iconv5.reshape(iconv5.shape[0], 1, iconv5.shape[1] * iconv5.shape[2], iconv5.shape[3])
        iconv5 = ttnn.from_torch(iconv5, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(iconv5)

        conv4b = torch.permute(conv4b, (0, 2, 3, 1))
        conv4b = conv4b.reshape(conv4b.shape[0], 1, conv4b.shape[1] * conv4b.shape[2], conv4b.shape[3])
        conv4b = ttnn.from_torch(conv4b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(conv4b)

        iconv4 = torch.permute(iconv4, (0, 2, 3, 1))
        iconv4 = iconv4.reshape(iconv4.shape[0], 1, iconv4.shape[1] * iconv4.shape[2], iconv4.shape[3])
        iconv4 = ttnn.from_torch(iconv4, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        iconv4 = tt_lib.tensor.interleaved_to_sharded(iconv4, self.pred_res4.conv.input_sharded_memory_config)
        pr4_res = self.pred_res4(iconv4)
        pr4 = pr4_res + base_flow[4]
        ttnn.deallocate(pr4_res)

        iconv4 = ttnn.to_torch(iconv4)
        iconv4 = iconv4.reshape(1, 60, 36, 256)
        iconv4 = torch.permute(iconv4, (0, 3, 1, 2))

        pr4 = ttnn.to_torch(pr4)
        pr4 = pr4.reshape(1, 60, 36, 1)
        pr4 = torch.permute(pr4, (0, 3, 1, 2))

        conv3b = ttnn.to_torch(conv3b)
        conv3b = conv3b.reshape(1, 120, 72, 128)
        conv3b = torch.permute(conv3b, (0, 3, 1, 2))

        upconv3 = self.upconv3(iconv4)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)
        iconv3 = torch.permute(iconv3, (0, 2, 3, 1))
        iconv3 = iconv3.reshape(iconv3.shape[0], 1, iconv3.shape[1] * iconv3.shape[2], iconv3.shape[3])
        iconv3 = ttnn.from_torch(iconv3, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        conv3b = torch.permute(conv3b, (0, 2, 3, 1))
        conv3b = conv3b.reshape(conv3b.shape[0], 1, conv3b.shape[1] * conv3b.shape[2], conv3b.shape[3])
        conv3b = ttnn.from_torch(conv3b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(conv3b)

        iconv4 = torch.permute(iconv4, (0, 2, 3, 1))
        iconv4 = iconv4.reshape(iconv4.shape[0], 1, iconv4.shape[1] * iconv4.shape[2], iconv4.shape[3])
        iconv4 = ttnn.from_torch(iconv4, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(iconv4)

        iconv3 = tt_lib.tensor.interleaved_to_sharded(iconv3, self.pred_res3.conv.input_sharded_memory_config)
        pr3_res = self.pred_res3(iconv3)
        pr3 = pr3_res + base_flow[3]
        ttnn.deallocate(pr3_res)

        iconv3 = ttnn.to_torch(iconv3)
        iconv3 = iconv3.reshape(1, 120, 72, 128)
        iconv3 = torch.permute(iconv3, (0, 3, 1, 2))

        pr3 = ttnn.to_torch(pr3)
        pr3 = pr3.reshape(1, 120, 72, 1)
        pr3 = torch.permute(pr3, (0, 3, 1, 2))

        conv2 = ttnn.to_torch(conv2)
        conv2 = conv2.reshape(1, 240, 144, 64)
        conv2 = torch.permute(conv2, (0, 3, 1, 2))

        upconv2 = self.upconv2(iconv3)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2), 1)
        iconv2 = self.iconv2(concat2)

        conv2 = torch.permute(conv2, (0, 2, 3, 1))
        conv2 = conv2.reshape(conv2.shape[0], 1, conv2.shape[1] * conv2.shape[2], conv2.shape[3])
        conv2 = ttnn.from_torch(conv2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(conv2)

        iconv3 = torch.permute(iconv3, (0, 2, 3, 1))
        iconv3 = iconv3.reshape(iconv3.shape[0], 1, iconv3.shape[1] * iconv3.shape[2], iconv3.shape[3])
        iconv3 = ttnn.from_torch(iconv3, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(iconv3)

        iconv2 = torch.permute(iconv2, (0, 2, 3, 1))
        iconv2 = iconv2.reshape(iconv2.shape[0], 1, iconv2.shape[1] * iconv2.shape[2], iconv2.shape[3])
        iconv2 = ttnn.from_torch(iconv2, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        iconv2 = tt_lib.tensor.interleaved_to_sharded(iconv2, self.pred_res2.conv.input_sharded_memory_config)
        pr2_res = self.pred_res2(iconv2)
        pr2 = pr2_res + base_flow[2]

        iconv2 = ttnn.to_torch(iconv2)
        iconv2 = iconv2.reshape(1, 240, 144, 64)
        iconv2 = torch.permute(iconv2, (0, 3, 1, 2))

        pr2 = ttnn.to_torch(pr2)
        pr2 = pr2.reshape(1, 240, 144, 1)
        pr2 = torch.permute(pr2, (0, 3, 1, 2))

        conv1 = ttnn.to_torch(conv1)
        conv1 = conv1.reshape(1, 480, 288, 32)
        conv1 = torch.permute(conv1, (0, 3, 1, 2))

        upconv1 = self.upconv1(iconv2)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1), 1)
        iconv1 = self.iconv1(concat1)

        conv1 = torch.permute(conv1, (0, 2, 3, 1))
        conv1 = conv1.reshape(conv1.shape[0], 1, conv1.shape[1] * conv1.shape[2], conv1.shape[3])
        conv1 = ttnn.from_torch(conv1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(conv1)

        iconv2 = torch.permute(iconv2, (0, 2, 3, 1))
        iconv2 = iconv2.reshape(iconv2.shape[0], 1, iconv2.shape[1] * iconv2.shape[2], iconv2.shape[3])
        iconv2 = ttnn.from_torch(iconv2, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(iconv2)

        iconv1 = torch.permute(iconv1, (0, 2, 3, 1))
        iconv1 = iconv1.reshape(iconv1.shape[0], 1, iconv1.shape[1] * iconv1.shape[2], iconv1.shape[3])
        iconv1 = ttnn.from_torch(iconv1, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        iconv1 = tt_lib.tensor.interleaved_to_sharded(iconv1, self.pred_res1.conv.input_sharded_memory_config)
        pr1_res = self.pred_res1(iconv1)
        pr1 = pr1_res + base_flow[1]
        ttnn.deallocate(pr1_res)

        iconv1 = ttnn.to_torch(iconv1)
        iconv1 = iconv1.reshape(1, 480, 288, 32)
        iconv1 = torch.permute(iconv1, (0, 3, 1, 2))

        pr1 = ttnn.to_torch(pr1)
        pr1 = pr1.reshape(1, 480, 288, 1)
        pr1 = torch.permute(pr1, (0, 3, 1, 2))

        upconv0 = self.upconv0(iconv1)
        upflow1 = self.upflow1to0(pr1)
        input_features = ttnn.to_torch(input_features)
        input_features = input_features.reshape(1, 960, 576, 11)
        input_features = torch.permute(input_features, (0, 3, 1, 2))
        concat0 = torch.cat((upconv0, upflow1, input_features[:, : self.input_channel, :, :]), 1)
        iconv0 = self.iconv0(concat0)

        iconv1 = torch.permute(iconv1, (0, 2, 3, 1))
        iconv1 = iconv1.reshape(iconv1.shape[0], 1, iconv1.shape[1] * iconv1.shape[2], iconv1.shape[3])
        iconv1 = ttnn.from_torch(iconv1, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        ttnn.deallocate(iconv1)

        # predict flow residual
        # iconv0 = torch.permute(iconv0, (0, 2, 3, 1))
        # iconv0 = iconv0.reshape(iconv0.shape[0], 1, iconv0.shape[1] * iconv0.shape[2], iconv0.shape[3])
        # iconv0 = ttnn.from_torch(iconv0, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        # iconv0 = tt_lib.tensor.interleaved_to_sharded(iconv0, self.pred_res0.conv.input_sharded_memory_config)
        iconv0 = torch_to_tt_tensor_rm(iconv0, device, put_on_device=False)
        # pr0_res = self.pred_res0(iconv0)
        pr0_res = tt_lib.fallback_ops.conv2d(input=iconv0, weight=self.pred_res0_weight, stride=(1, 1), padding=(1, 1))
        pr0_res = tt_to_torch_tensor(pr0_res)
        pr0_res = torch.permute(pr0_res, (0, 2, 3, 1))
        pr0_res = pr0_res.reshape(pr0_res.shape[0], 1, pr0_res.shape[1] * pr0_res.shape[2], pr0_res.shape[3])
        pr0_res = ttnn.from_torch(pr0_res, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        iconv0 = tt_to_torch_tensor(iconv0)
        iconv0 = torch.permute(iconv0, (0, 2, 3, 1))
        iconv0 = iconv0.reshape(iconv0.shape[0], 1, iconv0.shape[1] * iconv0.shape[2], iconv0.shape[3])
        iconv0 = ttnn.from_torch(iconv0, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        pr0 = pr0_res + base_flow[0]
        ttnn.deallocate(pr0_res)

        # apply ReLU
        pr0 = ttnn.to_torch(pr0)
        pr0 = pr0.reshape(1, 960, 576, 1)
        pr0 = torch.permute(pr0, (0, 3, 1, 2))

        pr0 = ttnn.from_torch(pr0, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        pr1 = ttnn.from_torch(pr1, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        pr2 = ttnn.from_torch(pr2, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        pr3 = ttnn.from_torch(pr3, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        pr4 = ttnn.from_torch(pr4, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        pr5 = ttnn.from_torch(pr5, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        pr6 = ttnn.from_torch(pr6, device=self.device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        pr0 = ttnn.relu(pr0, memory_config=ttnn.L1_MEMORY_CONFIG)
        pr1 = ttnn.relu(pr1, memory_config=ttnn.L1_MEMORY_CONFIG)
        pr2 = ttnn.relu(pr2, memory_config=ttnn.L1_MEMORY_CONFIG)
        pr3 = ttnn.relu(pr3, memory_config=ttnn.L1_MEMORY_CONFIG)
        pr4 = ttnn.relu(pr4, memory_config=ttnn.L1_MEMORY_CONFIG)
        pr5 = ttnn.relu(pr5, memory_config=ttnn.L1_MEMORY_CONFIG)
        pr6 = ttnn.relu(pr6, memory_config=ttnn.L1_MEMORY_CONFIG)

        if get_features:
            return pr0, pr1, pr2, pr3, pr4, pr5, pr6, iconv0
        else:
            return pr0, pr1, pr2, pr3, pr4, pr5, pr6
