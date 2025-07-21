# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_resnetblock2d_new_conv import resnetBlock2D
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upsample_2d_new_conv import upsample2d


class upblock_2d:
    def __init__(self, device, parameters, batch_size, input_height, input_width, compute_kernel_config):
        self.device = device
        self.parameters = parameters
        self.resnets = [
            resnetBlock2D(device, resnet, batch_size, input_height, input_width, compute_kernel_config)
            for resnet in parameters.resnets
        ]
        self.upsample_2d = upsample2d(
            device,
            parameters.upsamplers[0],
            batch_size,
            input_height,
            input_width,
            compute_kernel_config,
        )

        self.output_height = self.upsample_2d.output_height
        self.output_width = self.upsample_2d.output_width
        logger.info(f"Upblock Input = {input_height}x{input_width} Output = {self.output_height}x{self.output_width}")

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

            if hidden_states.is_sharded():
                hidden_states = ttnn.sharded_to_interleaved(hidden_states, ttnn.L1_MEMORY_CONFIG, hidden_states.dtype)

            if on_dev_res_hidden_states.is_sharded():
                on_dev_res_hidden_states = ttnn.sharded_to_interleaved(
                    on_dev_res_hidden_states, ttnn.L1_MEMORY_CONFIG, hidden_states.dtype
                )

            hidden_states = ttnn.concat(
                [hidden_states, on_dev_res_hidden_states], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            old_out = resnet(
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
            # new_out = self.new_resnets[i](
            #     hidden_states,
            #     temb=temb,
            #     temb_channels=temb_channels,
            #     time_embedding_norm=resnet_time_scale_shift,
            #     in_channels=resnet_in_channels + res_skip_channels,
            #     out_channels=out_channels,
            #     use_in_shortcut=None,
            #     groups=resnet_groups,
            #     output_scale_factor=output_scale_factor,
            # )
            # breakpoint()
            hidden_states = old_out
        if add_upsample:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
            hidden_states = self.upsample_2d(
                hidden_states,
                in_channels,
                out_channels,
            )

        return hidden_states
