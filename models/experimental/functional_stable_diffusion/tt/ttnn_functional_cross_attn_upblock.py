# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_upsample_2d import upsample2d
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_resnetblock2d import resnetBlock2D
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_transformer_2d import transformer_2d_model


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


def cross_attention_upblock2d(
    hidden_states,
    res_hidden_states_tuple,
    parameters,
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
    device=None,
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
):
    for i in range(num_layers):
        res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
        resnet_in_channels = prev_output_channel if i == 0 else out_channels

        res_hidden_states = res_hidden_states_tuple[-1]
        res_hidden_states_tuple = res_hidden_states_tuple[:-1]

        if isinstance(res_hidden_states, (ttnn.Tensor,)):
            on_dev_res_hidden_states = res_hidden_states
        else:
            on_dev_res_hidden_states = ttnn.from_torch(
                res_hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

        hidden_states = ttnn.concat([hidden_states, on_dev_res_hidden_states], dim=1)
        hidden_states = resnetBlock2D(
            hidden_states,
            temb=temb,
            temb_channels=temb_channels,
            time_embedding_norm=resnet_time_scale_shift,
            in_channels=resnet_in_channels + res_skip_channels,
            out_channels=out_channels,
            use_in_shortcut=None,
            groups=resnet_groups,
            output_scale_factor=output_scale_factor,
            parameters=parameters.resnets[i],
            eps=resnet_eps,
            pre_norm=resnet_pre_norm,
            non_linearity=resnet_act_fn,
            device=device,
        )
        if not dual_cross_attention:
            hidden_states = transformer_2d_model(
                hidden_states=hidden_states,
                parameters=parameters.attentions[i],
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
                device=device,
                upcast_attention=upcast_attention,
                cross_attention_dim=cross_attention_dim,
            )
        else:
            assert False, "We do not support Dual Transformer2DModel"

    if add_upsample:
        hidden_states = upsample2d(device, hidden_states, parameters.upsamplers[0], out_channels, out_channels)

    return hidden_states
