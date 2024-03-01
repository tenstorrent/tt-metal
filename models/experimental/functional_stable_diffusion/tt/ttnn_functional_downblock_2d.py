# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_resnetblock2d import resnetBlock2D
from models.experimental.functional_stable_diffusion.tt.ttnn_functional_downsample_2d import downsample_2d


def downblock2d(
    temb,
    hidden_states,
    device,
    in_channels,
    out_channels,
    temb_channels,
    dropout=0.0,
    num_layers=1,
    resnet_eps=1e-6,
    resnet_time_scale_shift="default",
    resnet_act_fn="swish",
    resnet_groups=32,
    resnet_pre_norm=True,
    output_scale_factor=1.0,
    add_downsample=False,
    downsample_padding=1,
    parameters=None,
    reader_patterns_cache: Optional[dict] = None,
    dtype: Optional[ttnn.DataType] = None,
    compute_kernel_config=None,
):
    output_states = ()
    for i in range(num_layers):
        in_channels = in_channels if i == 0 else out_channels
        resnet = resnetBlock2D(
            device=device,
            input_tensor=hidden_states,
            temb=temb,
            in_channels=in_channels,
            parameters=parameters.resnets[i],
            temb_channels=temb_channels,
            groups=resnet_groups,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            out_channels=out_channels,
            pre_norm=resnet_pre_norm,
            eps=resnet_eps,
            up=False,
            down=False,
            dtype=dtype,
            reader_patterns_cache=reader_patterns_cache,
        )

        hidden_states = resnet
        output_states += (hidden_states,)

    if add_downsample:
        downsamplers = [
            downsample_2d(
                in_channels=in_channels,
                hidden_states=hidden_states,
                use_conv=True,
                out_channels=out_channels,
                padding=downsample_padding,
                name="op",
                parameters=parameters.downsamplers[0],
                device=device,
                dtype=dtype,
                reader_patterns_cache=reader_patterns_cache,
                compute_kernel_config=compute_kernel_config,
            )
        ]

        for downsampler in downsamplers:
            hidden_states = downsampler
            output_states += (hidden_states,)

    return hidden_states, output_states
