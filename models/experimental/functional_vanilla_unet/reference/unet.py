# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
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
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=features),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))
