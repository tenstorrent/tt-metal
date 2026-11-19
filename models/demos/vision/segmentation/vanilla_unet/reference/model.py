# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        # The number of encoder and decoder levels the forward pass runs (num_levels_from_env).
        self.depth = 4
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 16, features * 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 16),
            nn.ReLU(inplace=True),
        )

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv2d(features * 16, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 8, features * 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 8),
            nn.ReLU(inplace=True),
        )
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv2d(features * 8, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 4, features * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 4),
            nn.ReLU(inplace=True),
        )
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv2d(features * 4, features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(features * 2, features * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features * 2),
            nn.ReLU(inplace=True),
        )
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(features * 2, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features).eval(),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for level in range(1, self.depth + 1):
            x = getattr(self, f"encoder{level}")(x if level == 1 else getattr(self, f"pool{level - 1}")(x))
            skips.append(x)
        # Below the full depth the next encoder stands in for the bottleneck: it has the bottleneck's
        # shapes at that depth.
        bottleneck = self.bottleneck if self.depth == 4 else getattr(self, f"encoder{self.depth + 1}")
        x = bottleneck(getattr(self, f"pool{self.depth}")(x))

        for level in range(self.depth, 0, -1):
            x = getattr(self, f"upconv{level}")(x)
            x = torch.cat((x, skips[level - 1]), dim=1)
            x = getattr(self, f"decoder{level}")(x)

        return self.conv(x)
