# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn


def ttnn_alexnet(device, x, parameters):
    batch_size = x.shape[0]

    x = ttnn.permute(x, (0, 2, 3, 1))

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        math_approx_mode_enabled=True,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )

    conv1_weight = ttnn.from_device(parameters.features[0].weight)
    conv1_bias = ttnn.from_device(parameters.features[0].bias)

    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv1_weight,
        in_channels=3,
        out_channels=64,
        device=device,
        bias_tensor=conv1_bias,
        kernel_size=(11, 11),
        stride=(4, 4),
        padding=(2, 2),
        batch_size=batch_size,
        input_height=x.shape[1],
        input_width=x.shape[2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    x = ttnn.reshape(x, (batch_size, 7, 7, 64))

    conv2_weight = ttnn.from_device(parameters.features[3].weight)
    conv2_bias = ttnn.from_device(parameters.features[3].bias)

    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv2_weight,
        in_channels=64,
        out_channels=192,
        device=device,
        bias_tensor=conv2_bias,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(2, 2),
        batch_size=batch_size,
        input_height=x.shape[1],
        input_width=x.shape[2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    conv3_weight = ttnn.from_device(parameters.features[6].weight)
    conv3_bias = ttnn.from_device(parameters.features[6].bias)

    x = ttnn.reshape(x, (batch_size, 3, 3, 192))

    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv3_weight,
        in_channels=192,
        out_channels=384,
        device=device,
        bias_tensor=conv3_bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=x.shape[1],
        input_width=x.shape[2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    conv4_weight = ttnn.from_device(parameters.features[8].weight)
    conv4_bias = ttnn.from_device(parameters.features[8].bias)

    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv4_weight,
        in_channels=384,
        out_channels=256,
        device=device,
        bias_tensor=conv4_bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    conv5_weight = ttnn.from_device(parameters.features[10].weight)
    conv5_bias = ttnn.from_device(parameters.features[10].bias)

    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv5_weight,
        in_channels=256,
        out_channels=256,
        device=device,
        bias_tensor=conv5_bias,
        kernel_size=(3, 3),
        stride=(1, 1),
        padding=(1, 1),
        batch_size=batch_size,
        input_height=out_height,
        input_width=out_width,
        conv_config=conv_config,
        conv_op_cache={},
        debug=False,
        groups=1,
    )

    x = ttnn.max_pool2d(
        input_tensor=x,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=x.shape[-1],
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    x = ttnn.reshape(x, (batch_size, 1, 1, 256))

    # ttnn currently only support AAP2 with output_size=(1,1), so torch op has been used.

    avg_pool = nn.AdaptiveAvgPool2d(output_size=(6, 6))

    x = ttnn.to_memory_config(x, memory_config=ttnn.L1_MEMORY_CONFIG)
    tt_output_tensor = ttnn.permute(x, (0, 3, 1, 2))
    tt_output_tensor = ttnn.from_device(tt_output_tensor)

    torch_output_tensor = ttnn.to_torch(tt_output_tensor)

    x = avg_pool(torch_output_tensor)

    x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)

    x = ttnn.reshape(x, (x.shape[0], -1))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    x = ttnn.linear(x, parameters.classifier[1].weight, bias=parameters.classifier[1].bias, activation="relu")
    x = ttnn.linear(x, parameters.classifier[4].weight, bias=parameters.classifier[4].bias, activation="relu")
    x = ttnn.linear(x, parameters.classifier[6].weight, bias=parameters.classifier[6].bias)

    return x


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype)
    return parameter


def custom_preprocessor(model, device):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        weight = model.weight
        bias = model.bias
        while weight.dim() < 4:
            weight = weight.unsqueeze(0)
        while bias.dim() < 4:
            bias = bias.unsqueeze(0)
        parameters["weight"] = preprocess_conv_parameter(weight, dtype=ttnn.bfloat16)
        parameters["bias"] = preprocess_conv_parameter(bias, dtype=ttnn.bfloat16)

    return parameters
