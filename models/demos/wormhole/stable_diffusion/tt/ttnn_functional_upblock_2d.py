# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_resnetblock2d import resnetBlock2D
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upsample_2d import upsample2d


class upblock_2d:
    def __init__(
        self, device, parameters, reader_patterns_cache, batch_size, input_height, input_width, compute_kernel_config
    ):
        self.device = device
        self.parameters = parameters
        self.resnets = [
            resnetBlock2D(
                device, resnet, reader_patterns_cache, batch_size, input_height, input_width, compute_kernel_config
            )
            for resnet in parameters.resnets
        ]
        self.upsample_2d = upsample2d(
            device,
            parameters.upsamplers[0],
            reader_patterns_cache,
            batch_size,
            input_height,
            input_width,
            compute_kernel_config,
        )

        self.output_height = self.upsample_2d.output_height
        self.output_width = self.upsample_2d.output_width

    def __call__(
        self,
        hidden_states,
        res_hidden_states_tuple,
        in_channels,
        prev_output_channel,
        out_channels,
        temb_channels,
        num_layers,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="silu",
        resnet_groups=32,
        resnet_pre_norm=True,
        output_scale_factor=1.0,
        add_upsample=True,
        state_dict=None,
        base_address=None,
        temb=None,
        upsample_size=None,
    ):
        for i, resnet in enumerate(self.resnets):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]

            if isinstance(res_hidden_states, (ttnn.Tensor,)):
                on_dev_res_hidden_states = res_hidden_states
            else:
                on_dev_res_hidden_states = ttnn.from_torch(
                    res_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
                )

            if ttnn.is_sharded(hidden_states) and hidden_states.layout == ttnn.ROW_MAJOR_LAYOUT:
                hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
            elif ttnn.is_sharded(hidden_states):
                hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)
            if ttnn.is_sharded(on_dev_res_hidden_states) and on_dev_res_hidden_states.layout == ttnn.ROW_MAJOR_LAYOUT:
                on_dev_res_hidden_states = ttnn.to_layout(
                    on_dev_res_hidden_states,
                    ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            elif ttnn.is_sharded(on_dev_res_hidden_states):
                on_dev_res_hidden_states = ttnn.to_memory_config(on_dev_res_hidden_states, ttnn.L1_MEMORY_CONFIG)
            if hidden_states.dtype != ttnn.bfloat8_b:
                hidden_states = ttnn.clone(
                    hidden_states, memory_config=ttnn.get_memory_config(hidden_states), dtype=ttnn.bfloat8_b
                )
            hidden_states = ttnn.concat(
                [hidden_states, on_dev_res_hidden_states], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            hidden_states = resnet(
                hidden_states,
                temb=temb,
                temb_channels=temb_channels,
                time_embedding_norm=resnet_time_scale_shift,
                in_channels=resnet_in_channels + res_skip_channels,
                out_channels=out_channels,
                use_in_shortcut=None,
                groups=resnet_groups,
                output_scale_factor=output_scale_factor,
            )

        if add_upsample:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = self.upsample_2d(
                hidden_states,
                in_channels,
                out_channels,
            )

        return hidden_states
