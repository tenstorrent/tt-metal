# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from models.experimental.functional_fadnetpp.reference.resblock import ResBlock


class ExtractNet(nn.Module):
    def __init__(self, resBlock=True, maxdisp=192, input_channel=6, encoder_ratio=16, decoder_ratio=16):
        super().__init__()
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

        self.conv1a = nn.Conv2d(3, self.basicE, 7, 2, padding=(7 - 1) // 2, bias=True)
        self.conv1b = nn.Conv2d(3, self.basicE, 7, 2, padding=(7 - 1) // 2, bias=True)
        if resBlock:
            self.conv2 = ResBlock(self.basicE, self.basicE * 2, stride=2)
            self.conv3 = ResBlock(self.basicE * 2, self.basicE * 4, stride=2)
        else:
            self.conv2 = nn.Conv2d(self.basicE, self.basicE * 2, 3, 2, padding=(3 - 1) // 2, bias=True)
            self.conv3 = (nn.Conv2d(self.basicE * 2, self.basicE * 4, 3, 2, padding=(3 - 1) // 2, bias=True),)

    def forward(self, inputs):
        # split left image and right image
        imgs = torch.chunk(inputs, 2, dim=1)
        img_left = imgs[0]
        img_right = imgs[1]
        conv1a = self.conv1a(img_left)
        conv1_l = self.relu(conv1a)
        conv1b = self.conv1b(img_right)
        conv1_r = self.relu(conv1b)

        if self.resBlock:
            conv2_l = self.conv2(conv1_l)
            conv3_l = self.conv3(conv2_l)
            conv2_r = self.conv2(conv1_r)
            conv3_r = self.conv3(conv2_r)
        else:
            conv2a = self.conv2(conv1_l)
            conv2_l = self.relu(conv2a)
            conv3a = self.conv3(conv2_l)
            conv3_l = self.relu(conv3a)
            conv2b = self.conv2(conv1_r)
            conv2_r = self.relu(conv2b)
            conv3b = self.conv3(conv2_r)
            conv3_r = self.relu(conv3b)
        return conv1_l, conv2_l, conv3_l, conv3_r
