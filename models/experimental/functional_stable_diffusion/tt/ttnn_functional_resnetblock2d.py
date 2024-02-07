# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
import torch
from typing import Optional, Dict


def torch_to_ttnn(input, device, layout=ttnn.TILE_LAYOUT):
    input = ttnn.from_torch(input, ttnn.bfloat16)
    input = ttnn.to_layout(input, layout)
    input = ttnn.to_device(input, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return input


def ttnn_to_torch(input):
    input = ttnn.to_layout(input, ttnn.ROW_MAJOR_LAYOUT)
    input = ttnn.from_device(input)
    input = ttnn.to_torch(input)
    return input


def permute_conv_weights(weight, bias):
    weight = ttnn.to_layout(weight, layout=ttnn.ROW_MAJOR_LAYOUT)
    weight = ttnn.to_torch(weight)
    weight = torch.permute(weight, (2, 3, 0, 1))
    bias = ttnn.to_layout(bias, layout=ttnn.ROW_MAJOR_LAYOUT)
    bias = ttnn.to_torch(bias)
    return weight, bias


config_override = {
    (320, 320, 64, 64): {"act_block_h": 64},
    (640, 640, 32, 32): {"act_block_h": 64},
    (640, 1920, 32, 32): {"act_block_h": 32},
    (1280, 1920, 16, 16): {"act_block_h": 32},
    (1280, 1280, 32, 32): {"act_block_h": 32},
    (320, 960, 64, 64): {"act_block_h": 32},
    (640, 960, 32, 32): {"act_block_h": 32},
    (320, 640, 64, 64): {"act_block_h": 32},
}

split_chunks = {
    (320, 960, 64, 64): 2,
    (640, 1920, 32, 32): 3,
    (1280, 1920, 16, 16): 3,
    (1280, 2560, 8, 8): 2,
    (1280, 2560, 16, 16): 2,
}


def resnetBlock2D(
    input_tensor,
    *,
    temb,
    in_channels,
    parameters,
    device,
    reader_patterns_cache,
    temb_channels=1280,
    groups: int = 32,
    time_embedding_norm: str = "default",
    output_scale_factor: float = 1.0,
    out_channels: Optional[int] = None,
    non_linearity="silu",
    pre_norm=True,
    eps=1e-5,
    up=False,
    down=False,
    use_in_shortcut: Optional[bool] = None,
    convs_on_device: bool = True,
):
    if non_linearity == "mish":
        assert False, "Mish is not implemented!"
    else:
        nonlinearity = ttnn.silu

    out_channels = in_channels if out_channels is None else out_channels
    hidden_states = input_tensor

    hidden_states = ttnn.group_norm(
        hidden_states, num_groups=groups, weight=parameters.norm1.weight, bias=parameters.norm1.bias, epsilon=eps
    )
    hidden_states = nonlinearity(hidden_states)

    if up:
        assert False, "Up block within residual block is not implemented!"
    elif down:
        assert False, "Down block within residual block is not implemented"

    parameters.conv1.weight, parameters.conv1.bias = permute_conv_weights(
        parameters.conv1.weight, parameters.conv1.bias
    )
    if convs_on_device:
        batch_size = hidden_states.shape[0]
        input_height = hidden_states.shape[2]
        input_width = hidden_states.shape[3]
        parameters.conv1.bias = torch.reshape(parameters.conv1.bias, (1, 1, 1, out_channels))
        conv1_split_chunks = 1
        if (out_channels, in_channels, input_height, input_width) in split_chunks:
            conv1_split_chunks = split_chunks[(out_channels, in_channels, input_height, input_width)]
        split_input_channels = in_channels // conv1_split_chunks
        if conv1_split_chunks == 1:
            split_weight_tensors = [parameters.conv1.weight]
        else:
            split_weight_tensors = torch.split(parameters.conv1.weight, split_input_channels, 1)
        conv1s = []
        for i in range(conv1_split_chunks):
            tt_weight_tensor = ttnn.from_torch(split_weight_tensors[i], ttnn.float32)
            if i == 0:
                tt_bias_tensor = ttnn.from_torch(parameters.conv1.bias, ttnn.float32)
            else:
                # TODO: fix no bias in conv error
                torch_bias_zeros_tensor = torch.zeros(parameters.conv1.bias.shape, dtype=torch.bfloat16).float()
                tt_bias_tensor = ttnn.from_torch(torch_bias_zeros_tensor, ttnn.float32)
            conv1_config_override = {}
            if (out_channels, in_channels, input_height, input_width) in config_override:
                conv1_config_override = config_override[(out_channels, in_channels, input_height, input_width)]
            conv1s.append(
                ttnn.Conv2D(
                    split_input_channels,
                    out_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    dtype=ttnn.bfloat8_b,
                    device=device,
                    use_1d_systolic_array=False,
                    batch_size=batch_size,
                    input_height=input_height,
                    input_width=input_width,
                    reader_patterns_cache=reader_patterns_cache,
                    weight=tt_weight_tensor,
                    bias=tt_bias_tensor,
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    weights_dtype=ttnn.bfloat8_b,
                    conv_blocking_and_parallelization_config_override=conv1_config_override,
                    use_shallow_conv_variant=False,
                    enable_auto_formatting=True,
                )
            )

        hidden_states = ttnn_to_torch(hidden_states)
        hidden_states = hidden_states.permute((0, 2, 3, 1))
        # Reshape 4d to 2d
        hidden_states_2d_height = batch_size * input_height * input_width
        hidden_states = torch.reshape(
            hidden_states,
            (1, 1, hidden_states_2d_height, in_channels),
        )

        hidden_states = ttnn.from_torch(hidden_states, ttnn.bfloat16)
        hidden_states = ttnn.to_device(hidden_states, device)

        hidden_states = ttnn.pad_to_tile(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        if conv1_split_chunks == 1:
            hidden_states = [hidden_states]
        else:
            split_hidden_states = []
            output_tensor_start_width_dim = 0
            output_tensor_end_width_dim = split_input_channels - 1
            for i in range(conv1_split_chunks):
                split_hidden_states.append(
                    ttnn.unpad(
                        hidden_states,
                        (0, 0, 0, output_tensor_start_width_dim),
                        (0, 0, hidden_states.shape.padded()[2] - 1, output_tensor_end_width_dim),
                    )
                )
                output_tensor_start_width_dim += split_input_channels
                output_tensor_end_width_dim += split_input_channels
            hidden_states = split_hidden_states

        if conv1_split_chunks == 1:
            hidden_states = conv1s[0](hidden_states[0])
        else:
            for i in range(conv1_split_chunks):
                hidden_states[i] = conv1s[i](hidden_states[i])
                if i != 0:
                    hidden_states[i] = ttnn.add(hidden_states[i], hidden_states[i - 1])
                    ttnn.deallocate(hidden_states[i - 1])
            hidden_states = hidden_states[-1]

        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_torch(hidden_states)
        hidden_states = torch.reshape(hidden_states, (batch_size, input_height, input_width, out_channels))
        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
        split_hidden_states = []
    else:
        parameters.conv1.weight = torch_to_tt_tensor_rm(parameters.conv1.weight, device, put_on_device=False)
        parameters.conv1.bias = torch_to_tt_tensor_rm(parameters.conv1.bias, device, put_on_device=False)
        # Using fallback Conv2D as we face issue with ttnn.Conv2D
        conv1 = fallback_ops.Conv2d(
            parameters.conv1.weight,
            parameters.conv1.bias,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        hidden_states = ttnn_to_torch(hidden_states)
        hidden_states = torch_to_tt_tensor_rm(hidden_states, device)
        hidden_states = conv1(hidden_states)
        hidden_states = tt_to_torch_tensor(hidden_states)
    hidden_states = torch_to_ttnn(hidden_states, device=device)

    if temb is not None:
        temb = nonlinearity(temb)
        if temb_channels is not None:
            if time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {time_embedding_norm} ")
            # temb=ttnn.linear(temb,parameters.time_emb_proj.weight,bias=parameters.time_emb_proj.bias)
            temb = ttnn.matmul(temb, parameters.time_emb_proj.weight)
            temb = ttnn.add(temb, parameters.time_emb_proj.bias)
        temb = ttnn.permute(temb, (2, 3, 0, 1))

    if temb is not None and time_embedding_norm == "default":
        hidden_states = hidden_states + temb

    hidden_states = ttnn.group_norm(
        hidden_states, num_groups=groups, weight=parameters.norm2.weight, bias=parameters.norm2.bias, epsilon=eps
    )
    hidden_states = nonlinearity(hidden_states)
    parameters.conv2.weight, parameters.conv2.bias = permute_conv_weights(
        parameters.conv2.weight, parameters.conv2.bias
    )
    if convs_on_device:
        batch_size = hidden_states.shape[0]
        input_height = hidden_states.shape[2]
        input_width = hidden_states.shape[3]
        parameters.conv2.bias = torch.reshape(parameters.conv2.bias, (1, 1, 1, out_channels))
        # print("conv2 weight shape=", parameters.conv2.weight.shape)
        # print("conv2 bias shape=", parameters.conv2.bias.shape)
        tt_weight_tensor = ttnn.from_torch(parameters.conv2.weight, ttnn.float32)
        tt_bias_tensor = ttnn.from_torch(parameters.conv2.bias, ttnn.float32)
        conv2_config_override = {}
        if (out_channels, in_channels, input_height, input_width) in config_override:
            conv2_config_override = config_override[(out_channels, in_channels, input_height, input_width)]
        conv2 = ttnn.Conv2D(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            dtype=ttnn.bfloat8_b,
            device=device,
            use_1d_systolic_array=False,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            reader_patterns_cache=reader_patterns_cache,
            weight=tt_weight_tensor,
            bias=tt_bias_tensor,
            math_fidelity=ttnn.MathFidelity.LoFi,
            weights_dtype=ttnn.bfloat8_b,
            conv_blocking_and_parallelization_config_override=conv2_config_override,
            use_shallow_conv_variant=False,
            enable_auto_formatting=True,
        )

        hidden_states = ttnn_to_torch(hidden_states)
        hidden_states = hidden_states.permute((0, 2, 3, 1))
        # Reshape 4d to 2d
        hidden_states = torch.reshape(
            hidden_states,
            (1, 1, batch_size * input_height * input_width, out_channels),
        )

        hidden_states = ttnn.from_torch(hidden_states, ttnn.bfloat16)
        hidden_states = ttnn.to_device(hidden_states, device)

        hidden_states = ttnn.pad_to_tile(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        hidden_states = conv2(hidden_states)
        hidden_states = ttnn.to_layout(hidden_states, ttnn.ROW_MAJOR_LAYOUT)
        hidden_states = ttnn.from_device(hidden_states)
        hidden_states = ttnn.to_torch(hidden_states)
        hidden_states = torch.reshape(hidden_states, (batch_size, input_height, input_width, out_channels))
        hidden_states = torch.permute(hidden_states, (0, 3, 1, 2))
    else:
        parameters.conv2.weight = torch_to_tt_tensor_rm(parameters.conv2.weight, device, put_on_device=False)
        parameters.conv2.bias = torch_to_tt_tensor_rm(parameters.conv2.bias, device, put_on_device=False)
        # Using fallback Conv2D as we face issue with ttnn.Conv2D
        conv2 = fallback_ops.Conv2d(
            parameters.conv2.weight,
            parameters.conv2.bias,
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        hidden_states = ttnn_to_torch(hidden_states)
        hidden_states = torch_to_tt_tensor_rm(hidden_states, device)
        hidden_states = conv2(hidden_states)
        hidden_states = tt_to_torch_tensor(hidden_states)
    hidden_states = torch_to_ttnn(hidden_states, device=device)

    use_in_shortcut = in_channels != out_channels if use_in_shortcut is None else use_in_shortcut
    if use_in_shortcut:
        parameters.conv_shortcut.weight, parameters.conv_shortcut.bias = permute_conv_weights(
            parameters.conv_shortcut.weight, parameters.conv_shortcut.bias
        )
        if convs_on_device:
            batch_size = input_tensor.shape[0]
            input_height = input_tensor.shape[2]
            input_width = input_tensor.shape[3]
            parameters.conv_shortcut.bias = torch.reshape(parameters.conv_shortcut.bias, (1, 1, 1, out_channels))
            tt_weight_tensor = ttnn.from_torch(parameters.conv_shortcut.weight, ttnn.float32)
            tt_bias_tensor = ttnn.from_torch(parameters.conv_shortcut.bias, ttnn.float32)
            conv_shortcut_config_override = {}
            # if (out_channels, in_channels, input_height, input_width) in config_override:
            #     conv2_config_override = config_override[(out_channels, in_channels, input_height, input_width)]
            conv_shortcut = ttnn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                dtype=ttnn.bfloat8_b,
                device=device,
                use_1d_systolic_array=False,
                batch_size=batch_size,
                input_height=input_height,
                input_width=input_width,
                reader_patterns_cache=reader_patterns_cache,
                weight=tt_weight_tensor,
                bias=tt_bias_tensor,
                math_fidelity=ttnn.MathFidelity.LoFi,
                weights_dtype=ttnn.bfloat8_b,
                conv_blocking_and_parallelization_config_override=conv_shortcut_config_override,
                use_shallow_conv_variant=False,
                enable_auto_formatting=True,
            )

            input_tensor = ttnn_to_torch(input_tensor)
            input_tensor = input_tensor.permute((0, 2, 3, 1))
            # Reshape 4d to 2d
            input_tensor = torch.reshape(
                input_tensor,
                (1, 1, batch_size * input_height * input_width, in_channels),
            )

            input_tensor = ttnn.from_torch(input_tensor, ttnn.bfloat16)
            input_tensor = ttnn.to_device(input_tensor, device)

            input_tensor = ttnn.pad_to_tile(input_tensor)
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT)
            input_tensor = conv_shortcut(input_tensor)
            input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
            input_tensor = ttnn.from_device(input_tensor)
            input_tensor = ttnn.to_torch(input_tensor)
            input_tensor = torch.reshape(input_tensor, (batch_size, input_height, input_width, out_channels))
            input_tensor = torch.permute(input_tensor, (0, 3, 1, 2))
        else:
            parameters.conv_shortcut.weight = torch_to_tt_tensor_rm(
                parameters.conv_shortcut.weight, device, put_on_device=False
            )
            parameters.conv_shortcut.bias = torch_to_tt_tensor_rm(
                parameters.conv_shortcut.bias, device, put_on_device=False
            )
            # Using fallback Conv2D as we face issue with ttnn.Conv2D
            conv_shortcut = fallback_ops.Conv2d(
                parameters.conv_shortcut.weight,
                parameters.conv_shortcut.bias,
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            input_tensor = ttnn_to_torch(input_tensor)
            input_tensor = torch_to_tt_tensor_rm(input_tensor, device)
            input_tensor = conv_shortcut(input_tensor)
            input_tensor = tt_to_torch_tensor(input_tensor)
        input_tensor = torch_to_ttnn(input_tensor, device=device)

    output_sc_recip = 1 / output_scale_factor
    output_tensor = ttnn.add(input_tensor, hidden_states)
    output_tensor = ttnn.mul(output_tensor, output_sc_recip)

    return output_tensor
