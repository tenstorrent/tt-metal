# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.functional_fadnetpp.reference.resblock import ResBlock


class CUNet(nn.Module):
    def __init__(self, resBlock=True, maxdisp=192, input_channel=3, encoder_ratio=16, decoder_ratio=16):
        super(CUNet, self).__init__()
        self.input_channel = input_channel
        self.maxdisp = maxdisp
        self.relu = nn.ReLU(inplace=False)
        self.basicC = 2
        self.eratio = encoder_ratio
        self.dratio = decoder_ratio
        self.basicE = self.basicC * self.eratio
        self.basicD = self.basicC * self.dratio
        self.resBlock = resBlock
        self.disp_width = maxdisp // 8 + 16

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        if resBlock:
            self.conv_redir = ResBlock(self.basicE * 4, self.basicE, stride=1)

            self.stride = 1
            self.max_in = 400 // 8 + 16 + self.basicE
            self.max_out = self.basicE * 4

            self.max_in_channels = self.max_in
            self.max_out_channels = self.max_out
            self.kernel_size = 3
            self.stride = self.stride
            self.dilation = 1

            self.conv_dy = nn.Conv2d(
                self.max_in_channels,
                self.max_out_channels,
                self.kernel_size,
                stride=self.stride,
                bias=False,
            )

            self.active_out_channel = self.max_out_channels

            self.bn_d1 = nn.BatchNorm2d(self.max_out)
            self.conv_d2 = nn.Conv2d(self.max_out, self.max_out, kernel_size=3, padding=1)
            self.bn_d2 = nn.BatchNorm2d(self.max_out)

            if self.stride != 1 or self.max_out != self.max_in:
                self.max_in_channels = self.max_in
                self.max_out_channels = self.max_out
                self.kernel_size = 3
                self.stride = self.stride
                self.dilation = 1

                self.conv_dy_1 = nn.Conv2d(
                    self.max_in_channels,
                    self.max_out_channels,
                    self.kernel_size,
                    stride=self.stride,
                    bias=False,
                )
                self.bn_dy = nn.BatchNorm2d(self.max_out)
            else:
                self.shortcut = None

            self.conv4 = ResBlock(self.basicE * 4, self.basicE * 8, stride=2)
            self.conv4_1 = ResBlock(self.basicE * 8, self.basicE * 8)
            self.conv5 = ResBlock(self.basicE * 8, self.basicE * 16, stride=2)
            self.conv5_1 = ResBlock(self.basicE * 16, self.basicE * 16)
            self.conv6 = ResBlock(self.basicE * 16, self.basicE * 32, stride=2)
            self.conv6_1 = ResBlock(self.basicE * 32, self.basicE * 32)
        else:
            self.conv_redir = nn.Conv2d(self.basicE * 4, self.basicE, 3, 1, bias=True)
            self.conv3_1 = nn.Conv2d(self.disp_width + self.basicE, self.basicE * 4, 3, 1, bias=True)
            self.conv4 = nn.Conv2d(self.basicE * 4, self.basicE * 8, 3, 2, bias=True)
            self.conv4_1 = nn.Conv2d(self.basicE * 8, self.basicE * 8, 3, 1, bias=True)
            self.conv5 = nn.Conv2d(self.basicE * 8, self.basicE * 16, 3, 1, bias=True)
            self.conv5_1 = nn.Conv2d(self.basicE * 16, self.basicE * 16, 3, 1, bias=True)
            self.conv6 = nn.Conv2d(self.basicE * 16, self.basicE * 32, 3, 1, bias=True)
            self.conv6_1 = nn.Conv2d(self.basicE * 32, self.basicE * 32, 3, 1, bias=True)

        self.pred_flow6 = nn.Conv2d(self.basicE * 32, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.iconv5 = nn.ConvTranspose2d((self.basicD + self.basicE) * 16 + 1, self.basicD * 16, 3, 1, 1)
        self.iconv4 = nn.ConvTranspose2d((self.basicD + self.basicE) * 8 + 1, self.basicD * 8, 3, 1, 1)
        self.iconv3 = nn.ConvTranspose2d((self.basicD + self.basicE) * 4 + 1, self.basicD * 4, 3, 1, 1)
        self.iconv2 = nn.ConvTranspose2d((self.basicD + self.basicE) * 2 + 1, self.basicD * 2, 3, 1, 1)
        self.iconv1 = nn.ConvTranspose2d((self.basicD + self.basicE) * 1 + 1, self.basicD, 3, 1, 1)
        self.iconv0 = nn.ConvTranspose2d(self.basicD + 3 + 1, self.basicD, 3, 1, 1)

        # expand and produce disparity
        self.upconv5 = nn.ConvTranspose2d(
            self.basicE * 32, self.basicD * 16, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upflow6to5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow5 = nn.Conv2d(self.basicE * 16, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv4 = nn.ConvTranspose2d(
            self.basicD * 16, self.basicD * 8, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upflow5to4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow4 = nn.Conv2d(self.basicE * 8, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv3 = nn.ConvTranspose2d(
            self.basicD * 8, self.basicD * 4, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upflow4to3 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow3 = nn.Conv2d(self.basicE * 4, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv2 = nn.ConvTranspose2d(
            self.basicD * 4, self.basicD * 2, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upflow3to2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow2 = nn.Conv2d(self.basicE * 2, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv1 = nn.ConvTranspose2d(self.basicD * 2, self.basicD, kernel_size=4, stride=2, padding=1, bias=False)
        self.upflow2to1 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow1 = nn.Conv2d(self.basicE, 1, kernel_size=3, stride=1, padding=1, bias=False)

        self.upconv0 = nn.ConvTranspose2d(self.basicD, self.basicD, kernel_size=4, stride=2, padding=1, bias=False)
        self.upflow1to0 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.pred_flow0 = nn.Conv2d(self.basicE, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, inputs, conv1_l, conv2_l, conv3a_l, corr_volume, get_features=False):
        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim=1)
        img_left = imgs[0]

        out_corr = self.corr_activation(corr_volume)
        if self.resBlock:
            out_conv3a_redir = self.conv_redir(conv3a_l)
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
            temp_in = torch.rand(1, 98, 120, 72)
            temp = self.conv_dy(temp_in)
            bn_d1 = self.bn_d1(y)
            relu_d1 = self.relu(bn_d1)
            conv_d2 = self.conv_d2(relu_d1)
            bn_d2 = self.bn_d2(conv_d2)

            if self.stride != 1 or self.max_out != self.max_in:

                def get_active_filter(in_channel, out_channel=None):
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
                temp = self.conv_dy_1(temp_in)
                bn_dy = self.bn_dy(z)
                bn_dy += bn_d2
                conv3b = bn_dy
            else:
                bn_dy = bn_d2
                conv3b = bn_dy
            conv4a = self.conv4(conv3b)
            conv4b = self.conv4_1(conv4a)
            conv5a = self.conv5(conv4b)
            conv5b = self.conv5_1(conv5a)
            conv6a = self.conv6(conv5b)
            conv6b = self.conv6_1(conv6a)

        else:
            out_conv3a_redir = self.conv_redir(conv3a_l)
            out_conv3a_redir_lr = self.corr_activation(out_conv3a_redir)
            in_conv3b = torch.cat((out_conv3a_redir_lr, out_corr), 1)

            conv3b_c = self.conv3_1(in_conv3b)
            conv3b = self.corr_activation(conv3b_c)
            conv4a_c = self.conv4(conv3b)
            conv4a = self.corr_activation(conv4a_c)
            conv4b_c = self.conv4_1(conv4a)
            conv4b = self.corr_activation(conv4b_c)
            conv5a_c = self.conv5(conv4b)
            conv5a = self.corr_activation(conv5a_c)
            conv5b_c = self.conv5_1(conv5a)
            conv5b = self.corr_activation(conv5b_c)
            conv6a_c = self.conv6(conv5b)
            conv6a = self.corr_activation(conv6a_c)
            conv6b_c = self.conv6_1(conv6a)
            conv6b = self.corr_activation(conv6b_c)

        pr6 = self.pred_flow6(conv6b)
        upconv5_c = self.upconv5(conv6b)
        upconv5 = self.corr_activation(upconv5_c)
        upflow6 = self.upflow6to5(pr6)
        concat5 = torch.cat((upconv5, upflow6, conv5b), 1)
        iconv5 = self.iconv5(concat5)

        pr5 = self.pred_flow5(iconv5)
        upconv4_c = self.upconv4(iconv5)
        upconv4 = self.corr_activation(upconv4_c)
        upflow5 = self.upflow5to4(pr5)
        concat4 = torch.cat((upconv4, upflow5, conv4b), 1)
        iconv4 = self.iconv4(concat4)

        pr4 = self.pred_flow4(iconv4)
        upconv3_c = self.upconv3(iconv4)
        upconv3 = self.corr_activation(upconv3_c)
        upflow4 = self.upflow4to3(pr4)
        concat3 = torch.cat((upconv3, upflow4, conv3b), 1)
        iconv3 = self.iconv3(concat3)

        pr3 = self.pred_flow3(iconv3)
        upconv2_c = self.upconv2(iconv3)
        upconv2 = self.corr_activation(upconv2_c)
        upflow3 = self.upflow3to2(pr3)
        concat2 = torch.cat((upconv2, upflow3, conv2_l), 1)
        iconv2 = self.iconv2(concat2)

        pr2 = self.pred_flow2(iconv2)
        upconv1_c = self.upconv1(iconv2)
        upconv1 = self.corr_activation(upconv1_c)
        upflow2 = self.upflow2to1(pr2)
        concat1 = torch.cat((upconv1, upflow2, conv1_l), 1)
        iconv1 = self.iconv1(concat1)

        pr1 = self.pred_flow1(iconv1)
        upconv0_c = self.upconv0(iconv1)
        upconv0 = self.corr_activation(upconv0_c)
        upflow1 = self.upflow1to0(pr1)
        concat0 = torch.cat((upconv0, upflow1, img_left), 1)
        iconv0 = self.iconv0(concat0)

        # predict flow
        pr0 = self.pred_flow0(iconv0)
        pr0 = self.relu(pr0)
        disps = (pr0, pr1, pr2, pr3, pr4, pr5, pr6)
        if get_features:
            features = (iconv5, iconv4, iconv3, iconv2, iconv1, iconv0)
            return disps, features
        else:
            return disps
