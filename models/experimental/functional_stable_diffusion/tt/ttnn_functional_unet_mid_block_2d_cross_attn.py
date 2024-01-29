# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_resnetblock2d import resnetBlock2D
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_transformer_2d import transformer_2d_model


def unet_mid_block_2d_cross_attn(
    hidden_states,
    temb,
    encoder_hidden_states,
    attention_mask,
    cross_attention_kwargs,
    parameters,
    config,
    in_channels,
    temb_channels,
    device,
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

    hidden_states = resnetBlock2D(
        input_tensor=hidden_states,
        temb=temb,
        in_channels=in_channels,
        out_channels=in_channels,
        parameters=parameters.resnets[0],
        device=device,
        temb_channels=temb_channels,
        groups=resnet_groups,
        time_embedding_norm=resnet_time_scale_shift,
        output_scale_factor=output_scale_factor,
        non_linearity=resnet_act_fn,
        pre_norm=resnet_pre_norm,
        eps=resnet_eps,
        use_in_shortcut=None,
    )

    for attn, resnet in zip(parameters.attentions, parameters.resnets[1:]):
        if not dual_cross_attention:
            hidden_states = transformer_2d_model(
                hidden_states=hidden_states,
                parameters=attn,
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
                device=device,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
            )
        else:
            assert False, "We do not support Dual Transformer"

        hidden_states = resnetBlock2D(
            input_tensor=hidden_states,
            temb=temb,
            in_channels=in_channels,
            out_channels=in_channels,
            parameters=resnet,
            device=device,
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
