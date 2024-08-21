# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_resnetblock2d import resnetBlock2D
from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_transformer_2d import transformer_2d_model


class unet_mid_block_2d_cross_attn:
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
        self.attentions = [
            transformer_2d_model(
                device, attention, reader_patterns_cache, batch_size, input_height, input_width, compute_kernel_config
            )
            for attention in parameters.attentions
        ]

        self.output_height = self.resnets[-1].output_height
        self.output_width = self.resnets[-1].output_width

    def __call__(
        self,
        hidden_states,
        temb,
        encoder_hidden_states,
        attention_mask,
        cross_attention_kwargs,
        config,
        in_channels,
        temb_channels,
        dropout=0.0,
        num_layers=1,
        resnet_eps=1e-6,
        resnet_time_scale_shift="default",
        resnet_act_fn="swish",
        resnet_groups=32,
        resnet_pre_norm=True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        has_cross_attention = True

        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        hidden_states = self.resnets[0](
            input_tensor=hidden_states,
            temb=temb,
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            groups=resnet_groups,
            time_embedding_norm=resnet_time_scale_shift,
            output_scale_factor=output_scale_factor,
            non_linearity=resnet_act_fn,
            pre_norm=resnet_pre_norm,
            eps=resnet_eps,
            use_in_shortcut=None,
        )

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if not dual_cross_attention:
                hidden_states = attn(
                    hidden_states=hidden_states,
                    config=config,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    num_attention_heads=attn_num_head_channels,
                    attention_head_dim=in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    norm_num_groups=resnet_groups,
                    patch_size=None,
                    num_embeds_ada_norm=None,
                    use_linear_projection=use_linear_projection,
                    norm_type="layer_norm",
                    eps=1e-5,
                    cross_attention_dim=cross_attention_dim,
                    upcast_attention=upcast_attention,
                )
            else:
                assert False, "We do not support Dual Transformer"

            hidden_states = resnet(
                input_tensor=hidden_states,
                temb=temb,
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                groups=resnet_groups,
                time_embedding_norm=resnet_time_scale_shift,
                output_scale_factor=output_scale_factor,
                non_linearity=resnet_act_fn,
                pre_norm=resnet_pre_norm,
                eps=resnet_eps,
                use_in_shortcut=None,
            )

        return hidden_states
