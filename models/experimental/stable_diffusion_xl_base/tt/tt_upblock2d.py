# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.stable_diffusion_xl_base.tt.tt_resnetblock2d import TtResnetBlock2D
from models.experimental.stable_diffusion_xl_base.tt.tt_upsample2d import TtUpsample2D


class TtUpBlock2D(LightweightModule):
    def __init__(
        self, device, state_dict, module_path, model_config, debug_mode=False, dram_groupnorm=False, has_upsample=False
    ):
        super().__init__()

        num_layers = 3
        self.resnets = []
        self.cached_hidden_states = {}
        self.cached_shapes = {}

        for i in range(num_layers):
            self.resnets.append(
                TtResnetBlock2D(
                    device,
                    state_dict,
                    f"{module_path}.resnets.{i}",
                    model_config,
                    conv_shortcut=True,
                    debug_mode=debug_mode,
                    use_negative_mask="up_blocks.0" not in module_path,
                    dram_groupnorm=dram_groupnorm and i == 0,
                )
            )

        self.upsamplers = (
            TtUpsample2D(
                device,
                state_dict,
                f"{module_path}.upsamplers.0",
                (1, 1),
                (1, 1),
                (1, 1),
                1,
                model_config=model_config,
                debug_mode=debug_mode,
            )
            if has_upsample
            else None
        )

    def forward(
        self, hidden_states, res_hidden_states_tuple, input_shape, temb, cache=False, use_cache=False, slice_id=0
    ):
        B, C, H, W = input_shape

        if use_cache:
            B, C, H, W = self.cached_shapes[slice_id]
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = ttnn.concat([self.cached_hidden_states[slice_id], res_hidden_states], dim=3)
            C = list(hidden_states.shape)[3]

            hidden_states, [C, H, W] = self.resnets[-1].forward(hidden_states, temb, [B, C, H, W])
        else:
            for i, resnet in enumerate(self.resnets):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]

                hidden_states = ttnn.concat([hidden_states, res_hidden_states], dim=3)
                C = list(hidden_states.shape)[3]

                hidden_states, [C, H, W] = resnet.forward(hidden_states, temb, [B, C, H, W])

                if (i == len(self.resnets) - 2) and (cache):
                    if slice_id in self.cached_hidden_states.keys():
                        ttnn.deallocate(self.cached_hidden_states[slice_id])
                    self.cached_hidden_states[slice_id] = ttnn.clone(hidden_states)
                    self.cached_shapes[slice_id] = [B, C, H, W]

        if self.upsamplers is not None:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = ttnn.reshape(hidden_states, [B, H, W, C])
            hidden_states, [C, H, W] = self.upsamplers.forward(hidden_states)

        return hidden_states, [C, H, W]
