# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from torchvision import models


# Conv Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


# Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv_block = ConvBlock(out_channels * 2, out_channels)  # Skip connection doubles channels

    def forward(self, x, skip_features):
        x = self.up(x)
        x = torch.cat([x, skip_features], dim=1)  # Concatenate skip features
        x = self.conv_block(x)
        return x


# U-Net with VGG19 Encoder
class UNetVGG19(nn.Module):
    def __init__(self, input_shape=(1, 3, 256, 256)):
        super(UNetVGG19, self).__init__()

        # Load pre-trained VGG19
        vgg19 = models.vgg19(pretrained=True).features
        # Encoder layers (skip connections)
        self.s1 = vgg19[:4]  # block1_conv2
        self.s2 = vgg19[4:9]  # block2_conv2
        self.s3 = vgg19[9:18]  # block3_conv4
        self.s4 = vgg19[18:27]  # block4_conv4

        # Bridge (block5_conv4)
        self.b1 = vgg19[27:36]  # block5_conv4

        # Decoder
        self.d1 = DecoderBlock(512, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        # Output
        self.out = nn.Conv2d(64, 1, kernel_size=1, padding="same")
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        s1 = self.s1(x)
        s2 = self.s2(s1)
        s3 = self.s3(s2)
        s4 = self.s4(s3)

        # Bridge
        b1 = self.b1(s4)

        # Decoder
        d1 = self.d1(b1, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # Output
        outputs = self.out(d4)
        outputs = self.sigmoid(outputs)

        return outputs
