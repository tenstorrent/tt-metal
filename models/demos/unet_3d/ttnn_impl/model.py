# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

from .decoder import Decoder
from .encoder import Encoder
from .group_norm3d import determine_expected_group_norm_sharded_config_and_grid_size, is_height_sharded_gn_from_dims
from .linear import FinalConv


class UNet3D:
    def __init__(
        self,
        device,
        in_channels,
        out_channels,
        base_channels=32,
        num_levels=3,
        num_groups=8,
        scale_factor=2,
    ):
        self.encoders = []
        self.decoders = []
        self.device = device

        # Encoder path
        c = base_channels
        for level in range(num_levels):
            in_ch = in_channels if level == 0 else c
            self.encoders.append(
                Encoder(
                    device,
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
        self.bottleneck = Encoder(
            device,
            is_bottleneck=True,
            in_channels=c,
            hid_channels=c,
            out_channels=c * 2,
            num_groups=num_groups,
        )

        # Decoder path
        for level in range(num_levels):
            self.decoders.append(Decoder(device, c * 3, c))
            c //= 2

        self.final_conv = FinalConv(c * 2, out_channels)

    def load_state_dict(self, params_dict: dict[str, torch.Tensor]):
        for idx, encoder in enumerate(self.encoders):
            encoder_prefix = f"encoders.{idx}"
            encoder.load_state_dict(self.device, params_dict, encoder_prefix)

        self.bottleneck.load_state_dict(self.device, params_dict, "bottleneck")

        for idx, decoder in enumerate(self.decoders):
            decoder_prefix = f"decoders.{idx}"
            decoder.load_state_dict(self.device, params_dict, decoder_prefix)

        self.final_conv.load_state_dict(self.device, params_dict, "final_conv")

    def initial_mem_config(self, device, groups, N, D, H, W, C):
        is_height_sharded = is_height_sharded_gn_from_dims(N, D, H, W, C)
        C_padded = (C + 31) // 32 * 32
        groups_padded = (C_padded // C) * groups
        sharded_mem_config, grid_size = determine_expected_group_norm_sharded_config_and_grid_size(
            device=device,
            num_channels=C_padded,
            num_groups=groups_padded,
            input_nhw=N * H * W * D,
            is_height_sharded=is_height_sharded,
            is_row_major=True,
        )

        return sharded_mem_config

    def __call__(self, x0) -> ttnn.Tensor:
        x = ttnn.permute(x0, (0, 2, 3, 4, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x0)
        skip_connections = []
        # Encoder path
        for encoder in self.encoders:
            x, skip = encoder(x, self.device)
            skip_connections.append(skip)

        # Bottleneck
        x = self.bottleneck(x, self.device)

        # Decoder path
        for decoder in self.decoders:
            skip = skip_connections.pop()
            x = decoder(x, skip, self.device)

        x = self.final_conv(x, self.device)

        return ttnn.permute(x, (0, 4, 1, 2, 3))
