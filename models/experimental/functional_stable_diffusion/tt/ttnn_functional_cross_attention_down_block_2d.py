# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_resnetblock2d import resnetBlock2D
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_transformer_2d import transformer_2d_model
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_downsample_2d import downsample_2d


def cross_attention_down_block_2d(
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
    *,
    parameters,
    device,
    reader_patterns_cache=None,
):
    output_states = ()

    for index, (resnet, attn) in enumerate(zip(parameters.resnets, parameters.attentions)):
        in_channels = in_channels if index == 0 else out_channels
        use_in_shortcut = True if "conv_shortcut" in resnet else False
        hidden_states = resnetBlock2D(
            hidden_states,
            temb=temb,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            parameters=resnet,
            device=device,
            use_in_shortcut=use_in_shortcut,
            reader_patterns_cache=reader_patterns_cache,
            eps=resnet_eps,
            groups=resnet_groups,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )
        if not dual_cross_attention:
            hidden_states = transformer_2d_model(
                hidden_states,
                attn,
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
                device=device,
                reader_patterns_cache=reader_patterns_cache,
            )

        output_states += (hidden_states,)

    if add_downsample is not None:
        hidden_states = downsample_2d(
            in_channels=out_channels,
            out_channels=out_channels,
            hidden_states=hidden_states,
            padding=downsample_padding,
            device=device,
            parameters=parameters.downsamplers[0],
            use_conv=True,
            reader_patterns_cache=reader_patterns_cache,
        )
        output_states += (hidden_states,)
    return hidden_states, output_states
