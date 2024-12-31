# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.stable_diffusion.tt.ttnn_resnetblock2d import ResnetBlock2D


def down_block_2d(device, parameters, config, input, temb, num_layers=2):
    hidden_states = input
    output_states = ()
    for i in range(num_layers):
        hidden_states = ResnetBlock2D(
            config=config,
            input_tensor=hidden_states,
            temb=temb,
            parameters=parameters.resnets[i],
            device=device,
        )
        output_states = output_states + (hidden_states,)
    hidden_states = run_conv_with_split_1(
        device,
        hidden_states,
        batch_size=1,
        parameters=parameters.downsamplers[0],
        kernel_size=3,
        stride=2,
        pad=1,
        split_factor=2,
        weights_dtype=ttnn.bfloat8_b,
        ttnn_weight=parameters.downsamplers[0].conv.weight,
        ttnn_bias=parameters.downsamplers[0].conv.bias,
    )

    output_states = output_states + (hidden_states,)
    return hidden_states, output_states


def run_conv_with_split_1(
    device,
    input_tensor,
    batch_size,
    parameters,
    kernel_size,
    stride,
    pad,
    split_factor,
    weights_dtype=ttnn.bfloat8_b,
    ttnn_weight=None,
    ttnn_bias=None,
):
    input_channels = input_tensor.shape[1]
    assert input_channels % split_factor == 0
    split_input_channels = input_channels // split_factor
    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    ttnn_weight = ttnn.to_torch(ttnn_weight)
    ttnn_weight = ttnn.from_torch(ttnn_weight, dtype=ttnn.float32)
    ttnn_bias = ttnn.to_torch(ttnn_bias)
    ttnn_bias = ttnn.from_torch(ttnn_bias, dtype=ttnn.float32)

    split_input_tensors = ttnn.split(input_tensor, split_factor, 1)
    split_weight_tensors = ttnn.split(ttnn_weight, split_factor, 1)
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=weights_dtype,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_weight_tensor = split_weight_tensors
    out_channels = tt_weight_tensor[1].shape[0]
    for i in range(split_factor):
        tt_input_tensor = ttnn.permute(split_input_tensors[i], (0, 2, 3, 1))
        tt_input_tensor = ttnn.from_device(tt_input_tensor)
        [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
            input_tensor=tt_input_tensor,
            weight_tensor=tt_weight_tensor[i],
            in_channels=split_input_channels,
            out_channels=tt_weight_tensor[i].shape[0],
            device=device,
            bias_tensor=ttnn_bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(pad, pad),
            batch_size=batch_size,
            input_height=tt_input_tensor.shape[1],
            input_width=tt_input_tensor.shape[2],
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache=reader_patterns_cache,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        conv_output_tensor = ttnn.reshape(tt_output_tensor_on_device, (batch_size, out_height, out_width, out_channels))
        conv_output_tensor = ttnn.sharded_to_interleaved(conv_output_tensor)
        conv_output_tensor = ttnn.permute(conv_output_tensor, (0, 3, 1, 2))
        if i == 0:
            output_tensor = conv_output_tensor
        else:
            output_tensor = ttnn.add(output_tensor, conv_output_tensor)

    return output_tensor
