# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.tt_transformer2dmodel import TtTransformer2DModel
from models.experimental.stable_diffusion_xl_refiner.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_downsample2d import TtDownsample2D


class TtCrossAttnDownBlock2D(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
        has_downsample=True,
    ):
        super().__init__()

        num_layers = 2
        self.attentions = []
        self.resnets = []
        self.device = device

        for i in range(num_layers):
            self.attentions.append(
                TtTransformer2DModel(
                    device,
                    state_dict,
                    f"{module_path}.attentions.{i}",
                )
            )

        for i in range(num_layers):
            self.resnets.append(
                TtResnetBlock2D(
                    device,
                    state_dict,
                    f"{module_path}.resnets.{i}",
                )
            )

        self.downsamplers = (
            TtDownsample2D(
                device,
                state_dict,
                f"{module_path}.downsamplers.0",
            )
            if has_downsample
            else None
        )

    def forward(self, input_tensor, input_shape, temb=None, encoder_hidden_states=None):
        B, C, H, W = input_shape
        residuals = ()

        hidden_states = input_tensor
        tt_blocks = list(zip(self.resnets, self.attentions))
        for resnet, attn in tt_blocks:
            hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])
            hidden_states = attn.forward(hidden_states, [B, C, H, W], encoder_hidden_states=encoder_hidden_states)
            # Create a copy of the tensor to avoid aliasing issues with hidden_states
            residual = ttnn.clone(hidden_states)
            residuals = residuals + (residual,)

        if self.downsamplers is not None:
            hidden_states, [C, H, W] = self.downsamplers.forward(hidden_states, [B, C, H, W])
            # Create a copy of the tensor to avoid aliasing issues with hidden_states
            residual = ttnn.clone(hidden_states)
            residuals = residuals + (residual,)

        return hidden_states, [C, H, W], residuals
