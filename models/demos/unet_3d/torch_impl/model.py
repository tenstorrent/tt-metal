# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from .bottleneck import BottleneckTch
from .decoder import DecoderTch
from .encoder import EncoderTch


class UNet3DTch(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=32, num_levels=3):
        super(UNet3DTch, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        # Encoder path
        c = base_channels
        for level in range(num_levels):
            in_ch = in_channels if level == 0 else c
            self.encoders.append(EncoderTch(in_ch, c, c * 2))
            c *= 2

        # Bottleneck
        self.bottleneck = BottleneckTch(c, c * 2)

        # Decoder path
        for level in range(num_levels):
            self.decoders.append(DecoderTch(c * 3, c))
            c //= 2

        # Final convolution
        self.final_conv = nn.Conv3d(c * 2, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for decoder in self.decoders:
            skip = skip_connections.pop()
            x = decoder(x, skip)

        # Final convolution
        x = self.final_conv(x)
        return x
