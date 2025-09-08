# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.tt_transformer2dmodel import TtTransformer2DModel
from models.experimental.stable_diffusion_xl_refiner.tt.tt_resnetblock2d import TtResnetBlock2D


class TtCrossAttnMidBlock2D(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        num_layers_resn = 2  # there is only one attention layer
        self.resnets = []
        self.device = device

        self.attention = TtTransformer2DModel(
            device,
            state_dict,
            f"{module_path}.attentions.0",
        )

        for i in range(num_layers_resn):
            self.resnets.append(
                TtResnetBlock2D(
                    device,
                    state_dict,
                    f"{module_path}.resnets.{i}",
                )
            )

    def forward(self, input_tensor, input_shape, temb=None, encoder_hidden_states=None):
        B, C, H, W = input_shape
        hidden_states = input_tensor

        hidden_states, [C, H, W] = self.resnets[0].forward(hidden_states, temb, [B, C, H, W])

        hidden_states = self.attention.forward(hidden_states, [B, C, H, W], encoder_hidden_states=encoder_hidden_states)

        hidden_states, [C, H, W] = self.resnets[1].forward(hidden_states, temb, [B, C, H, W])

        return hidden_states, [C, H, W]
