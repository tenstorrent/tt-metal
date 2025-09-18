# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger

import ttnn
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_resnetblock2d_new_conv import resnetBlock2D
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_transformer_2d_new_conv import transformer_2d_model
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_upsample_2d_new_conv import upsample2d
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_utility_functions import dealloc_input


class cross_attention_upblock2d:
    def __init__(self, device, parameters, batch_size, input_height, input_width, compute_kernel_config):
        self.device = device
        self.parameters = parameters
        self.resnets = [
            resnetBlock2D(device, resnet, batch_size, input_height, input_width, compute_kernel_config)
            for resnet in parameters.resnets
        ]
        self.attentions = [
            transformer_2d_model(device, attention, batch_size, input_height, input_width, compute_kernel_config)
            for attention in parameters.attentions
        ]

        self.output_height = self.attentions[-1].output_height
        self.output_width = self.attentions[-1].output_width

        if "upsamplers" in parameters:
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
        logger.info(
            f"Cross Attention UpBlock Input = {input_height}x{input_width} Output = {self.output_height}x{self.output_width}"
        )

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
        temb=None,
        upsample_size=None,
        config=None,
        encoder_hidden_states=None,
        timestep=None,
        class_labels=None,
        cross_attention_kwargs=None,
        return_dict=True,
        num_attention_heads=16,
        attention_head_dim=88,
        num_layers_transformer=1,
        norm_num_groups=32,
        num_vector_embeds=None,
        patch_size=None,
        num_embeds_ada_norm=None,
        use_linear_projection=False,
        norm_type="layer_norm",
        attention_mask=None,
        dual_cross_attention=False,
        upcast_attention: bool = False,
        cross_attention_dim=1280,
        attn_num_head_channels=1,
        only_cross_attention: bool = False,
        index=-1,
    ):
        for i, (resnet, attention) in enumerate(zip(self.resnets, self.attentions)):
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
                hidden_states = dealloc_input(
                    ttnn.to_layout,
                    hidden_states,
                    ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            elif ttnn.is_sharded(hidden_states):
                hidden_states = dealloc_input(ttnn.to_memory_config, hidden_states, ttnn.L1_MEMORY_CONFIG)
            if ttnn.is_sharded(on_dev_res_hidden_states) and on_dev_res_hidden_states.layout == ttnn.ROW_MAJOR_LAYOUT:
                on_dev_res_hidden_states = ttnn.to_layout(
                    on_dev_res_hidden_states,
                    ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            elif ttnn.is_sharded(on_dev_res_hidden_states):
                on_dev_res_hidden_states = ttnn.to_memory_config(on_dev_res_hidden_states, ttnn.L1_MEMORY_CONFIG)
            if hidden_states.dtype != ttnn.bfloat8_b:
                hidden_states = dealloc_input(
                    ttnn.clone, hidden_states, memory_config=ttnn.get_memory_config(hidden_states), dtype=ttnn.bfloat8_b
                )
            hidden_states = dealloc_input(ttnn.concat, [hidden_states, on_dev_res_hidden_states], dim=3)
            ttnn.deallocate(on_dev_res_hidden_states)
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
                eps=resnet_eps,
                pre_norm=resnet_pre_norm,
                non_linearity=resnet_act_fn,
                index=index,
            )
            if not dual_cross_attention:
                hidden_states = attention(
                    hidden_states=hidden_states,
                    config=config,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep=timestep,
                    class_labels=class_labels,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=return_dict,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=num_layers_transformer,
                    patch_size=patch_size,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    use_linear_projection=use_linear_projection,
                    norm_type=norm_type,
                    upcast_attention=upcast_attention,
                    cross_attention_dim=cross_attention_dim,
                    output_bfloat16=(not add_upsample) and (i == len(self.resnets) - 1),
                )
            else:
                assert False, "We do not support Dual Transformer2DModel"

        if add_upsample:
            assert "upsamplers" in self.parameters
            hidden_states = self.upsample_2d(
                hidden_states,
                out_channels,
                out_channels,
            )

        return hidden_states
