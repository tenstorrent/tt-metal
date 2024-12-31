# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
from models.demos.stable_diffusion.tt.resnetblock2d_utils import (
    run_conv_with_split,
)
from models.demos.stable_diffusion.tt.ttnn_resnetblock2d import ResnetBlock2D
from models.demos.stable_diffusion.tt.ttnn_transformer2d import sd_transformer_2d


def sd_downsample_2(input_tensor, parameters, device):
    tt_output_tensor_on_device = run_conv_with_split(
        device,
        input_tensor,
        batch_size=1,
        parameters=parameters,
        kernel_size=3,
        stride=2,
        pad=1,
        weights_dtype=ttnn.bfloat8_b,
        split_factor=4,
        ttnn_weight=parameters.conv.weight,
        ttnn_bias=parameters.conv.bias,
    )
    return tt_output_tensor_on_device


def sd_cross_attention_down_blocks2d(
    hidden_states,
    temb=None,
    encoder_hidden_states=None,
    timestep=None,
    attention_mask=None,
    cross_attention_kwargs=None,
    encoder_attention_mask=None,
    additional_residuals=None,
    config=None,
    conv_shortcut=True,
    use_torch_conv=False,
    class_labels=None,
    add_downsample=False,
    return_dict=None,
    attention_head_dim=None,
    num_layers=None,
    norm_num_groups=32,
    transformer_layers_per_block=10,
    device=None,
    parameters=None,
):
    output_states = ()
    for index, (resnet, attn) in enumerate(zip(parameters.resnets, parameters.attentions)):
        hidden_states = ttnn.to_layout(hidden_states, layout=ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ResnetBlock2D(
            config,
            input_tensor=hidden_states,
            temb=temb,
            parameters=resnet,
            device=device,
            use_torch_conv=use_torch_conv,
        )

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG)
        hidden_states = sd_transformer_2d(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            parameters=attn,
            device=device,
            timestep=timestep,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=return_dict,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            transformer_layers_per_block=transformer_layers_per_block,
            norm_num_groups=norm_num_groups,
            attention_mask=attention_mask,
            config=config,
            eps=1e-06,
        )
        output_states = output_states + (hidden_states,)
    if add_downsample:
        hidden_states = sd_downsample_2(hidden_states, parameters.downsamplers[0], device)
        output_states = output_states + (hidden_states,)
    return hidden_states, output_states
