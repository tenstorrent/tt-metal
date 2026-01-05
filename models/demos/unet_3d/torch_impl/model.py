# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from .decoder import DecoderTch
from .encoder import EncoderTch


class UNet3DTch(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_channels=32,
        num_levels=3,
        num_groups=8,
        scale_factor=2,
    ):
        super(UNet3DTch, self).__init__()
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # Encoder path
        c = base_channels
        for level in range(num_levels):
            in_ch = in_channels if level == 0 else c
            self.encoders.append(
                EncoderTch(
                    is_bottleneck=False,
                    in_channels=in_ch,
                    hid_channels=c,
                    out_channels=c * 2,
                    num_groups=num_groups,
                    scale_factor=scale_factor,
                )
            )
            c *= 2

        # Bottleneck
        self.bottleneck = EncoderTch(
            is_bottleneck=True,
            in_channels=c,
            hid_channels=c,
            out_channels=c * 2,
            num_groups=num_groups,
        )
        # Decoder path
        for level in range(num_levels):
            self.decoders.append(DecoderTch(c * 3, c, num_groups=num_groups))
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

        x = torch.nn.functional.sigmoid(x)
        return x
