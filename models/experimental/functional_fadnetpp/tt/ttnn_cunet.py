# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from models.experimental.functional_fadnetpp.tt.ttnn_resblock import TtResBlock
import ttnn
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
from tt_lib.fallback_ops import fallback_ops
import torch.nn.functional as F


class ttCUNet:
    def build_corr(img_left, img_right, max_disp=40):
        B, C, H, W = img_left.shape
        volume = img_left.new_zeros([B, max_disp, H, W])
        for i in range(max_disp):
            if i > 0:
                volume[:, i, :, i:] = (img_left[:, :, :, i:] * img_right[:, :, :, :-i]).mean(dim=1)
            else:
                volume[:, i, :, :] = (img_left[:, :, :, :] * img_right[:, :, :, :]).mean(dim=1)

        volume = volume.contiguous()
        return volume

    def output_preprocessing(self, output_tensor, height, width, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = torch.reshape(
            output_tensor,
            [
                output_tensor.shape[0],
                output_tensor.shape[1],
                height,
                width,
            ],
        )
        return output_tensor

    def input_preprocessing(self, input_tensor, device):
        input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
        input_tensor = torch.reshape(
            input_tensor,
            (input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]),
        )
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        return input_tensor

    def __init__(self, parameters, model, resBlock=True) -> None:
        self.resBlock = resBlock
        self.kernel_size = 3
        self.stride = 1
        self.dilation = 1
        self.max_out = 128
        self.basicE = 32

        self.bn_d1 = model.bn_d1

        self.bn_dy = model.bn_dy

        self.relu = nn.ReLU(inplace=False)
        self.max_in = 400 // 8 + 16 + 32
        if self.resBlock:
            self.conv_redir = TtResBlock(parameters.conv_redir, 128, 32, stride=1)
            self.conv_dy = parameters.conv_dy
            self.conv_d2 = parameters.conv_d2
            self.conv_dy_1 = parameters.conv_dy_1
            self.conv4 = TtResBlock(parameters.conv4, self.basicE * 4, self.basicE * 8, stride=2)
            self.conv4_1 = TtResBlock(parameters.conv4_1, self.basicE * 8, self.basicE * 8)
            self.conv5 = TtResBlock(parameters.conv5, self.basicE * 8, self.basicE * 16, stride=2)
            self.conv5_1 = TtResBlock(parameters.conv5_1, self.basicE * 16, self.basicE * 16)
            self.conv6 = TtResBlock(parameters.conv6, self.basicE * 16, self.basicE * 32, stride=2)
            self.conv6_1 = TtResBlock(parameters.conv6_1, self.basicE * 32, self.basicE * 32)
        else:
            self.conv_redir = parameters.conv_redir
            self.conv_dy = parameters.conv_dy
            self.conv_d2 = parameters.conv_d2
            self.conv_dy_1 = parameters.conv_dy_1
            self.conv4 = parameters.conv4
            self.conv4_1 = parameters.conv4_1
            self.conv5 = parameters.conv5
            self.conv5_1 = parameters.conv5_1
            self.conv6 = parameters.conv6
            self.conv6_1 = parameters.conv6_1
        self.pred_flow6 = parameters.pred_flow6

        self.iconv5 = nn.ConvTranspose2d(1025, 512, 3, 1, 1)
        self.iconv5.weight = parameters.iconv5["weight"]
        self.iconv5.bias = parameters.iconv5["bias"]
        self.iconv4 = nn.ConvTranspose2d(513, 256, 3, 1, 1)
        self.iconv4.weight = parameters.iconv4["weight"]
        self.iconv4.bias = parameters.iconv4["bias"]
        self.iconv3 = nn.ConvTranspose2d(257, 128, 3, 1, 1)
        self.iconv3.weight = parameters.iconv3["weight"]
        self.iconv3.bias = parameters.iconv3["bias"]
        self.iconv2 = nn.ConvTranspose2d(129, 64, 3, 1, 1)
        self.iconv2.weight = parameters.iconv2["weight"]
        self.iconv2.bias = parameters.iconv2["bias"]
        self.iconv1 = nn.ConvTranspose2d(65, 32, 3, 1, 1)
        self.iconv1.weight = parameters.iconv1["weight"]
        self.iconv1.bias = parameters.iconv1["bias"]
        self.iconv0 = nn.ConvTranspose2d(35, 32, 3, 1, 1)
        self.iconv0.weight = parameters.iconv0["weight"]
        self.iconv0.bias = parameters.iconv0["bias"]

        self.upconv5 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv5.weight = parameters.upconv5["weight"]
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv4.weight = parameters.upconv4["weight"]
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv3.weight = parameters.upconv3["weight"]
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv2.weight = parameters.upconv2["weight"]
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv1.weight = parameters.upconv1["weight"]
        self.upconv0 = nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.upconv0.weight = parameters.upconv0["weight"]

        self.pred_flow5 = parameters.pred_flow5
        self.pred_flow4 = parameters.pred_flow4
        self.pred_flow3 = parameters.pred_flow3
        self.pred_flow2 = parameters.pred_flow2
        self.pred_flow1 = parameters.pred_flow1
        self.pred_flow0 = fallback_ops.Conv2d(parameters.pred_flow0["weight"], None, 32, 1, 3, 1, padding=1, bias=False)

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

    def __call__(self, input, device, conv1_l, conv2_l, conv3a_l, corr_volume, get_features=False):
        img_left = input
        corr_volume = ttnn.to_device(corr_volume, device)
        corr_volume = ttnn.to_memory_config(corr_volume, memory_config=ttnn.L1_MEMORY_CONFIG)
        out_corr = ttnn.leaky_relu(corr_volume, slope=0.1)
        out_corr = self.output_preprocessing(out_corr, 120, 72, device)
        if self.resBlock:
            out_conv3a_redir = self.conv_redir(device, conv3a_l)
            out_conv3a_redir = self.output_preprocessing(out_conv3a_redir, 120, 72, device)
            in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

            def get_active_filter(in_channel, out_channel=None):
                if out_channel:
                    return self.conv_dy.weight[:out_channel, :in_channel, :, :]
                else:
                    return self.conv_dy.weight[:, :in_channel, :, :]

            in_channel = in_conv3b.size(1)
            filters = get_active_filter(in_channel).contiguous()

            def get_same_padding(kernel_size):
                if isinstance(kernel_size, tuple):
                    assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
                    p1 = get_same_padding(kernel_size[0])
                    p2 = get_same_padding(kernel_size[1])
                    return p1, p2
                assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
                assert kernel_size % 2 > 0, "kernel size should be odd number"
                return kernel_size // 2

            padding = get_same_padding(self.kernel_size)
            y = F.conv2d(in_conv3b, filters, None, self.stride, padding, self.dilation, 1)

            bn_d1 = self.bn_d1(y)
            relu_d1 = self.relu(bn_d1)
            relu_d1 = torch.permute(relu_d1, (0, 2, 3, 1))
            relu_d1 = torch.reshape(
                relu_d1,
                (relu_d1.shape[0], 1, relu_d1.shape[1] * relu_d1.shape[2], relu_d1.shape[3]),
            )
            relu_d1 = ttnn.from_torch(
                relu_d1,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
            )
            relu_d1 = relu_d1.to(device, self.conv_d2.conv.input_sharded_memory_config)
            conv_d2 = self.conv_d2(relu_d1)

            if self.stride != 1 or self.max_out != self.max_in:

                def get_active_filter(n_channel, out_channel=None):
                    if out_channel:
                        return self.conv_dy_1.weight[:out_channel, :in_channel, :, :]
                    else:
                        return self.conv_dy_1.weight[:, :in_channel, :, :]

                in_channel = in_conv3b.size(1)
                filters = get_active_filter(in_channel).contiguous()

                def get_same_padding(kernel_size):
                    if isinstance(kernel_size, tuple):
                        assert len(kernel_size) == 2, "invalid kernel size: %s" % kernel_size
                        p1 = get_same_padding(kernel_size[0])
                        p2 = get_same_padding(kernel_size[1])
                        return p1, p2
                    assert isinstance(kernel_size, int), "kernel size should be either `int` or `tuple`"
                    assert kernel_size % 2 > 0, "kernel size should be odd number"
                    return kernel_size // 2

                padding = get_same_padding(self.kernel_size)
                z = F.conv2d(in_conv3b, filters, None, self.stride, padding, self.dilation, 1)

                bn_dy = self.bn_dy(z)
                bn_dy = torch.permute(bn_dy, (0, 2, 3, 1))
                bn_dy = torch.reshape(
                    bn_dy,
                    (bn_dy.shape[0], 1, bn_dy.shape[1] * bn_dy.shape[2], bn_dy.shape[3]),
                )
                bn_dy = ttnn.from_torch(
                    bn_dy,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                bn_dy += conv_d2
                conv3b = bn_dy
            else:
                bn_dy = conv_d2
                conv3b = bn_dy

            conv4a = self.conv4(device, conv3b)
            conv4b = self.conv4_1(device, conv4a)
            conv5a = self.conv5(device, conv4b)
            conv5b = self.conv5_1(device, conv5a)
            conv6a = self.conv6(device, conv5b)
            conv6b = self.conv6_1(device, conv6a)

        else:
            out_conv3a_redir = self.conv_redir(conv3a_l)
            in_conv3b = torch.cat((out_conv3a_redir, out_corr), 1)

            conv3b = self.conv3_1(in_conv3b)
            conv4a = self.conv4(conv3b)
            conv4b = self.conv4_1(conv4a)
            conv5a = self.conv5(conv4b)
            conv5b = self.conv5_1(conv5a)
            conv6a = self.conv6(conv5b)
            conv6b = self.conv6_1(conv6a)

        conv6b = self.output_preprocessing(conv6b, 15, 9, device)

        upconv5 = self.upconv5(conv6b)
        upconv5 = self.input_preprocessing(upconv5, device)
        upconv5 = ttnn.to_device(upconv5, device)
        upconv5 = ttnn.to_memory_config(upconv5, memory_config=ttnn.L1_MEMORY_CONFIG)
        upconv5 = ttnn.leaky_relu(upconv5, slope=0.1)
        upconv5 = self.output_preprocessing(upconv5, 30, 18, device)

        conv6b = self.input_preprocessing(conv6b, device)
        conv6b = conv6b.to(device, self.pred_flow6.conv.input_sharded_memory_config)

        pr6 = self.pred_flow6(conv6b)
        pr6 = self.output_preprocessing(pr6, 15, 9, device)

        upflow6 = self.upflow6to5(pr6)

        conv5b = self.output_preprocessing(conv5b, 30, 18, device)

        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)

        iconv5 = self.iconv5(concat5)
        iconv5 = self.input_preprocessing(iconv5, device)
        iconv5 = iconv5.to(device, self.pred_flow5.conv.input_sharded_memory_config)

        pr5 = self.pred_flow5(iconv5)

        iconv5 = self.output_preprocessing(iconv5, 30, 18, device)
        upconv4 = self.upconv4(iconv5)
        upconv4 = self.input_preprocessing(upconv4, device)
        upconv4 = ttnn.to_device(upconv4, device)
        upconv4 = ttnn.to_memory_config(upconv4, memory_config=ttnn.L1_MEMORY_CONFIG)
        upconv4 = ttnn.leaky_relu(upconv4, slope=0.1)
        upconv4 = self.output_preprocessing(upconv4, 60, 36, device)

        pr5 = self.output_preprocessing(pr5, 30, 18, device)
        upflow5 = self.upflow5to4(pr5)
        conv4b = self.output_preprocessing(conv4b, 60, 36, device)

        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)

        iconv4 = self.iconv4(concat4)
        upconv3 = self.upconv3(iconv4)
        upconv3 = self.input_preprocessing(upconv3, device)
        upconv3 = ttnn.to_device(upconv3, device)
        upconv3 = ttnn.to_memory_config(upconv3, memory_config=ttnn.L1_MEMORY_CONFIG)
        upconv3 = ttnn.leaky_relu(upconv3, slope=0.1)
        upconv3 = self.output_preprocessing(upconv3, 120, 72, device)
        iconv4 = self.input_preprocessing(iconv4, device)
        iconv4 = iconv4.to(device, self.pred_flow4.conv.input_sharded_memory_config)

        pr4 = self.pred_flow4(iconv4)

        pr4 = self.output_preprocessing(pr4, 60, 36, device)

        upflow4 = self.upflow4to3(pr4)
        conv3b = self.output_preprocessing(conv3b, 120, 72, device)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)
        upconv2 = self.upconv2(iconv3)
        upconv2 = self.input_preprocessing(upconv2, device)
        upconv2 = ttnn.to_device(upconv2, device)
        upconv2 = ttnn.to_memory_config(upconv2, memory_config=ttnn.L1_MEMORY_CONFIG)
        upconv2 = ttnn.leaky_relu(upconv2, slope=0.1)
        upconv2 = self.output_preprocessing(upconv2, 240, 144, device)
        iconv3 = self.input_preprocessing(iconv3, device)
        iconv3 = iconv3.to(device, self.pred_flow3.conv.input_sharded_memory_config)
        pr3 = self.pred_flow3(iconv3)
        pr3 = self.output_preprocessing(pr3, 120, 72, device)

        upflow3 = self.upflow3to2(pr3)
        conv2_l = self.output_preprocessing(conv2_l, 240, 144, device)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)
        upconv1 = self.upconv1(iconv2)
        upconv1 = self.input_preprocessing(upconv1, device)
        upconv1 = ttnn.to_device(upconv1, device)
        upconv1 = ttnn.to_memory_config(upconv1, memory_config=ttnn.L1_MEMORY_CONFIG)
        upconv1 = ttnn.leaky_relu(upconv1, slope=0.1)
        upconv1 = self.output_preprocessing(upconv1, 480, 288, device)
        iconv2 = self.input_preprocessing(iconv2, device)
        iconv2 = iconv2.to(device, self.pred_flow2.conv.input_sharded_memory_config)
        pr2 = self.pred_flow2(iconv2)
        pr2 = self.output_preprocessing(pr2, 240, 144, device)

        upflow2 = self.upflow2to1(pr2)
        conv1_l = self.output_preprocessing(conv1_l, 480, 288, device)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)
        upconv0 = self.upconv0(iconv1)
        upconv0 = self.input_preprocessing(upconv0, device)
        upconv0 = ttnn.to_device(upconv0, device)
        upconv0 = ttnn.to_memory_config(upconv0, memory_config=ttnn.L1_MEMORY_CONFIG)
        upconv0 = ttnn.leaky_relu(upconv0, slope=0.1)
        upconv0 = self.output_preprocessing(upconv0, 960, 576, device)
        iconv1 = self.input_preprocessing(iconv1, device)
        iconv1 = iconv1.to(device, self.pred_flow1.conv.input_sharded_memory_config)
        pr1 = self.pred_flow1(iconv1)
        pr1 = self.output_preprocessing(pr1, 480, 288, device)

        upflow1 = self.upflow1to0(pr1)
        img_left = self.output_preprocessing(img_left, 960, 576, device)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)
        # predict flow
        iconv0 = torch_to_tt_tensor_rm(iconv0, device, put_on_device=True)
        pr0 = self.pred_flow0(iconv0)
        pr0 = ttnn.to_torch(pr0)
        pr0 = self.relu(pr0)

        pr0 = self.input_preprocessing(pr0, device)
        pr1 = self.input_preprocessing(pr1, device)
        pr2 = self.input_preprocessing(pr2, device)
        pr3 = self.input_preprocessing(pr3, device)
        pr4 = self.input_preprocessing(pr4, device)
        pr5 = self.input_preprocessing(pr5, device)
        pr6 = self.input_preprocessing(pr6, device)
        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)

        # can be chosen outside
        if get_features:
            features = (iconv5, iconv4, iconv3, iconv2, iconv1, iconv0)
            return disps, features
        else:
            return disps
