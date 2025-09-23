# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.vae.tt.tt_downsample2d import TtDownsample2D


class TtDownEncoderBlock2D(LightweightModule):
    def __init__(self, device, state_dict, module_path, model_config, has_downsample=False, has_shortcut=False):
        super().__init__()

        num_layers = 2
        self.resnets = []

        for i in range(num_layers):
            self.resnets.append(
                TtResnetBlock2D(
                    device,
                    state_dict,
                    f"{module_path}.resnets.{i}",
                    model_config=model_config,
                    conv_shortcut=(i == 0) and has_shortcut,
                )
            )

        self.downsamplers = (
            TtDownsample2D(
                device,
                state_dict,
                f"{module_path}.downsamplers.0",
                (2, 2),
                (0, 1, 0, 1),
                (1, 1),
                1,
                model_config=model_config,
            )
            if has_downsample
            else None
        )

    def forward(self, hidden_states, input_shape):
        B, C, H, W = input_shape

        for resnet in self.resnets:
            hidden_states, [C, H, W] = resnet.forward(hidden_states, [B, C, H, W])

        if self.downsamplers is not None:
            hidden_states, [C, H, W] = self.downsamplers.forward(hidden_states, [B, C, H, W])
        return hidden_states, [C, H, W]
