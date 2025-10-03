# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_refiner.tt.tt_config import get_upblock_config
from models.experimental.stable_diffusion_xl_refiner.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_refiner.tt.tt_upsample2d import TtUpsample2D


class TtUpBlock2D(LightweightModule):
    def __init__(
        self,
        device,
        state_dict,
        module_path,
    ):
        super().__init__()

        self.device = device
        self.module_path = module_path

        # Configuration setup
        self.block_config = get_upblock_config(module_path)

        self._initialize_components(state_dict)

    def _initialize_components(self, state_dict):
        # Order of layers:
        # 1. resnets (list of three)
        # 2. upsample (optional)

        self.resnets = []
        for i in range(self.block_config.num_resnets):
            resnet_module_path = f"{self.module_path}.resnets.{i}"

            self.resnets.append(
                TtResnetBlock2D(
                    device=self.device,
                    state_dict=state_dict,
                    module_path=resnet_module_path,
                )
            )

        if self.block_config.has_upsample:
            self.upsample = TtUpsample2D(
                device=self.device,
                state_dict=state_dict,
                module_path=f"{self.module_path}.upsamplers.0",
            )
        else:
            self.upsample = None

    def forward(self, hidden_states, input_shape, res_hidden_states_tuple, temb):
        B, C, H, W = input_shape

        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
            hidden_states = ttnn.concat([hidden_states, res_hidden_states], dim=3)
            C = list(hidden_states.shape)[3]

            hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])

        if self.upsample is not None:
            hidden_states = ttnn.reshape(hidden_states, (B, H, W, C))
            hidden_states, [C, H, W] = self.upsample.forward(hidden_states)

        return hidden_states, [C, H, W]
