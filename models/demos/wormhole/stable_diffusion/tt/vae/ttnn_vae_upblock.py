# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_resnet import ResnetBlock
from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae_upsample import UpsampleBlock


class UpDecoderBlock:
    def __init__(
        self,
        torch_upblock,
        device,
        in_channels,
        input_height,
        input_width,
        out_channels,
        output_height,
        output_width,
        resnet_norm_blocks=[(1, 1), (1, 1), (1, 1)],
    ):
        self.device = device
        self.resnets = []
        for i in range(3):
            self.resnets.append(
                ResnetBlock(
                    torch_upblock.resnets[i],
                    device,
                    in_channels if i == 0 else out_channels,
                    input_height,
                    input_width,
                    out_channels,
                    resnet_norm_blocks[i][0],
                    resnet_norm_blocks[i][1],
                )
            )

        self.upsample = None
        if torch_upblock.upsamplers:
            self.upsample = UpsampleBlock(
                torch_upblock.upsamplers[0],
                device,
                out_channels,
                input_height,
                input_width,
                out_channels,
                output_height,
                output_width,
            )

    def __call__(self, hidden_states):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)
            ttnn.ReadDeviceProfiler(self.device)

        if self.upsample:
            hidden_states = self.upsample(hidden_states)

        return hidden_states
