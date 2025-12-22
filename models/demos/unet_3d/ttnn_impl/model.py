# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from .bottleneck import Bottleneck
from .conv3d import Conv3D
from .decoder import Decoder
from .encoder import Encoder


class UNet3D:
    def __init__(self, device, in_channels: int, out_channels: int, base_channels: int = 32, num_levels: int = 3):
        self.device = device
        self.encoders = []
        self.decoders = []

        # Encoder path
        c = base_channels
        for level in range(num_levels):
            in_ch = in_channels if level == 0 else c
            encoder = Encoder(device, in_ch, c, c * 2)
            self.encoders.append(encoder)
            c *= 2

        # Bottleneck
        self.bottleneck = Bottleneck(device, c, c * 2)
        # Decoder path
        for level in range(num_levels):
            decoder = Decoder(device, c * 3, c)
            self.decoders.append(decoder)
            c //= 2

        self.final_conv = Conv3D(device, c * 2, out_channels, kernel_size=1)

    def init_params(self, device, params_dict: dict[str, torch.Tensor]):
        # Initialize encoder parameters
        for idx, encoder in enumerate(self.encoders):
            encoder_prefix = f"encoders.{idx}"
            encoder.init_params(device, params_dict, encoder_prefix)

        # Initialize bottleneck parameters
        self.bottleneck.init_params(device, params_dict, "bottleneck")

        # Initialize decoder parameters
        for idx, decoder in enumerate(self.decoders):
            decoder_prefix = f"decoders.{idx}"
            decoder.init_params(device, params_dict, decoder_prefix)

        self.final_conv.init_params(device, params_dict, "final_conv")

    def __call__(self, x) -> ttnn.Tensor:
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

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = self.final_conv(x)

        return x
