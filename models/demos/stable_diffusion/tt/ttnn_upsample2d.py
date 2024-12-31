# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def run_conv_with_split(
    device, input_tensor, batch_size, parameters, conv_params, kernel_size, split_factor, weights_dtype=ttnn.bfloat8_b
):
    input_channels = input_tensor.shape[1]
    assert input_channels % split_factor == 0
    split_input_channels = input_channels // split_factor
    ttnn_weight = parameters.conv.weight
    ttnn_bias = parameters.conv.bias
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_weight = ttnn.to_torch(ttnn_weight)
    ttnn_bias = ttnn.to_torch(ttnn_bias)
    ttnn_bias = ttnn_bias.reshape((1, 1, 1, -1))
    ttnn_weight = ttnn.from_torch(ttnn_weight, dtype=ttnn.float32)
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
            stride=(conv_params[0], conv_params[1]),
            padding=(conv_params[2], conv_params[3]),
            batch_size=batch_size,
            input_height=tt_input_tensor.shape[1],
            input_width=tt_input_tensor.shape[2],
            conv_config=conv_config,
            compute_config=compute_config,
            conv_op_cache=reader_patterns_cache,
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


def upsample(input_tensor, parameters, device):
    input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
    input_tensor = ttnn.upsample(input_tensor, scale_factor=2)
    input_tensor = ttnn.permute(input_tensor, (0, 3, 1, 2))
    output_tensor = run_conv_with_split(
        device,
        input_tensor,
        batch_size=1,
        parameters=parameters,
        conv_params=[1, 1, 1, 1],
        kernel_size=3,
        split_factor=16,
        weights_dtype=ttnn.bfloat8_b,
    )

    return output_tensor
