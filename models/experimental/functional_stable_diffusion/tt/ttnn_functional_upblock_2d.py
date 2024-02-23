# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from models.experimental.functional_stable_diffusion.tt.ttnn_functional_resnetblock2d import resnetBlock2D
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_upsample_2d import upsample2d


def upblock_2d(
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
    state_dict=None,
    base_address=None,
    temb=None,
    upsample_size=None,
    reader_patterns_cache=None,
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
            device=device,
            reader_patterns_cache=reader_patterns_cache,
        )

    if add_upsample:
        hidden_states = upsample2d(
            device,
            hidden_states,
            parameters.upsamplers[0],
            in_channels,
            out_channels,
            reader_patterns_cache=reader_patterns_cache,
        )

    return hidden_states
