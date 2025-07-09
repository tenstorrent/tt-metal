# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_upsample2d import TtUpsample2D


class TtUpDecoderBlock2D(nn.Module):
    def __init__(
        self, device, state_dict, module_path, model_config, has_upsample=False, conv_shortcut=False, gn_fallback=False
    ):
        super().__init__()

        num_layers = 3
        self.attentions = []
        self.resnets = []

        for i in range(num_layers):
            self.resnets.append(
                TtResnetBlock2D(
                    device,
                    state_dict,
                    f"{module_path}.resnets.{i}",
                    model_config,
                    conv_shortcut=conv_shortcut and (i == 0),
                    gn_fallback=gn_fallback,
                )
            )

        self.upsamplers = (
            TtUpsample2D(device, state_dict, f"{module_path}.upsamplers.0", model_config, (1, 1), (1, 1), (1, 1), 1)
            if has_upsample
            else None
        )

    def forward(self, input_tensor, input_shape):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        for resnet in self.resnets:
            hidden_states = ttnn.reshape(hidden_states, (1, 1, B * H * W, C))
            hidden_states, [C, H, W] = resnet.forward(hidden_states, [B, C, H, W])

        ttnn.deallocate(input_tensor)
        if self.upsamplers is not None:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, (B, H, W, C))
            hidden_states = ttnn.move(hidden_states)
            hidden_states, [C, H, W] = self.upsamplers.forward(hidden_states)

        return hidden_states, [C, H, W]
