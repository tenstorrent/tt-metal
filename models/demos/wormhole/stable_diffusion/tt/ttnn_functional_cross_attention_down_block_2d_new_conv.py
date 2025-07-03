# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_downsample_2d_new_conv import downsample_2d
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_resnetblock2d_new_conv import resnetBlock2D
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_transformer_2d_new_conv import transformer_2d_model


class cross_attention_down_block_2d:
    def __init__(self, device, parameters, batch_size, input_height, input_width, compute_kernel_config):
        self.device = device
        self.parameters = parameters
        self.resnets = [
            resnetBlock2D(device, resnet, batch_size, input_height, input_width, compute_kernel_config)
            for i, resnet in enumerate(parameters.resnets)
        ]
        self.attentions = [
            transformer_2d_model(device, attention, batch_size, input_height, input_width, compute_kernel_config)
            for attention in parameters.attentions
        ]
        self.downsample_2d = downsample_2d(
            device,
            parameters.downsamplers[0],
            batch_size,
            input_height,
            input_width,
            compute_kernel_config,
        )

        self.output_height = self.downsample_2d.output_height
        self.output_width = self.downsample_2d.output_width
        logger.info(
            f"Cross Attention Down Block Input = {input_height}x{input_width} Output = {self.output_height}x{self.output_width}"
        )

    def __call__(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        add_downsample=True,
        attention_mask=None,
        cross_attention_kwargs={},
        config=None,
        num_layers=2,
        in_channels: int = None,
        out_channels: int = None,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        dual_cross_attention=False,
        temb_channels=1280,
        groups=32,
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=False,
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        downsample_padding: int = 1,
        cross_attention_dim: int = 768,
        attn_num_head_channels: int = 8,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
        resnet_time_scale_shift: str = "default",
    ):
        output_states = ()

        for index, (resnet, attn) in enumerate(zip(self.resnets, self.attentions)):
            in_channels = in_channels if index == 0 else out_channels
            use_in_shortcut = True if "conv_shortcut" in resnet.parameters else False
            hidden_states = resnet(
                hidden_states,
                temb=temb,
                in_channels=in_channels,
                out_channels=out_channels,
                temb_channels=temb_channels,
                use_in_shortcut=use_in_shortcut,
                eps=resnet_eps,
                groups=resnet_groups,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
            if not dual_cross_attention:
                hidden_states = attn(
                    hidden_states,
                    config,
                    encoder_hidden_states,
                    attention_head_dim=out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )

            output_states += (ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG),)

        if add_downsample is not None:
            hidden_states = self.downsample_2d(
                in_channels=out_channels,
                out_channels=out_channels,
                hidden_states=hidden_states,
                padding=downsample_padding,
                use_conv=True,
            )
            hidden_states = ttnn.reallocate(hidden_states)
            output_states += (ttnn.to_memory_config(hidden_states, ttnn.DRAM_MEMORY_CONFIG),)
        return hidden_states, output_states
