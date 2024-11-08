# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import numpy as np
import torch


def conv(
    activation, input_tensor, weight, bias, in_channels, out_channels, device, kernel_size, padding, stride, in_h, in_w
):
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        activation=activation,
        weights_dtype=ttnn.bfloat16,
        math_fidelity=ttnn.MathFidelity.LoFi,
        input_channels_alignment=(16 if False or (in_channels == 16 and input_tensor.shape[-2] == 115) else 32),
        shard_layout=None,
        fp32_dest_acc_enabled=False,
        packer_l1_accum_enabled=False,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
    )
    [output_tt_tensor, out_height, out_width, weights, bias] = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight,
        bias_tensor=bias,
        in_channels=in_channels,
        out_channels=out_channels,
        device=device,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        batch_size=input_tensor.shape[0],
        input_height=in_h,
        input_width=in_w,
        conv_config=conv_config,
        groups=1,
    )
    return output_tt_tensor, out_height, out_width


def tt_Fire(
    inplanes: int,
    squeeze_planes: int,
    expand1x1_planes: int,
    expand3x3_planes: int,
    input_tensor: ttnn.Tensor,
    parameters,
    device,
    dtype=ttnn.bfloat16,
    activation="relu",
):
    sweight = parameters.squeeze.weight
    sbias = parameters.squeeze.bias
    output_tt_tensor_1, out_height_1, out_width_1 = conv(
        activation="relu",
        input_tensor=input_tensor,
        weight=sweight,
        bias=sbias,
        in_channels=inplanes,
        out_channels=squeeze_planes,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        in_h=input_tensor.shape[1],
        in_w=input_tensor.shape[2],
    )

    expand1x1_weights = parameters.expand1x1.weight
    expand1x1_bias = parameters.expand1x1.bias
    output_tt_tensor_2, out_height_2, out_width_2 = conv(
        activation="relu",
        input_tensor=output_tt_tensor_1,
        weight=expand1x1_weights,
        bias=expand1x1_bias,
        in_channels=squeeze_planes,
        out_channels=expand1x1_planes,
        device=device,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        in_h=out_height_1,
        in_w=out_width_1,
    )

    expand3x3_weights = parameters.expand3x3.weight
    expand3x3_bias = parameters.expand3x3.bias
    output_tt_tensor_3, out_height_3, out_width_3 = conv(
        activation="relu",
        input_tensor=output_tt_tensor_1,
        weight=expand3x3_weights,
        bias=expand3x3_bias,
        in_channels=squeeze_planes,
        out_channels=expand3x3_planes,
        device=device,
        kernel_size=(3, 3),
        padding=(1, 1),
        stride=(1, 1),
        in_h=out_height_1,
        in_w=out_width_1,
    )

    output_tt_tensor_2 = ttnn.to_memory_config(output_tt_tensor_2, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tt_tensor_3 = ttnn.to_memory_config(output_tt_tensor_3, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tt_tensor_2 = ttnn.to_layout(output_tt_tensor_2, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tt_tensor_3 = ttnn.to_layout(output_tt_tensor_3, layout=ttnn.ROW_MAJOR_LAYOUT)
    final_output = ttnn.concat([output_tt_tensor_2, output_tt_tensor_3], dim=3)

    final_output_reshaped = ttnn.reshape(
        final_output, (output_tt_tensor_3.shape[0], out_height_3, out_width_3, expand3x3_planes + expand1x1_planes)
    )
    final_output_reshaped = ttnn.to_layout(final_output_reshaped, layout=ttnn.TILE_LAYOUT)
    return final_output_reshaped


def tt_squeezenet(device, parameters, tt_input, dtype=ttnn.bfloat16, activation="relu"):
    conv_1_weights = parameters.features[0].weight
    conv_1_bias = parameters.features[0].bias

    output_tt, out_height_1, out_width_1 = conv(
        activation="relu",
        input_tensor=tt_input,
        weight=conv_1_weights,
        bias=conv_1_bias,
        in_channels=conv_1_weights.shape[1],
        out_channels=conv_1_weights.shape[0],
        device=device,
        kernel_size=(conv_1_weights.shape[2], conv_1_weights.shape[3]),
        padding=(0, 0),
        stride=(2, 2),
        in_h=tt_input.shape[1],
        in_w=tt_input.shape[2],
    )
    output_tt = ttnn.to_layout(output_tt, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tt = ttnn.reshape(output_tt, (1, 109, 109, 96))
    torch_tensor_1 = ttnn.to_torch(output_tt)
    torch_tensor_1 = torch.permute(torch_tensor_1, (0, 3, 1, 2))
    torch_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch op
    torch_out = torch_pool(torch_tensor_1)
    tt_input_2 = ttnn.from_torch(torch_out, dtype=ttnn.bfloat16, device=device)
    tt_input_2 = ttnn.permute(tt_input_2, (0, 2, 3, 1))
    tt_input_2 = ttnn.from_device(tt_input_2)
    tt_int_tensor_3 = tt_Fire(
        96, 16, 64, 64, input_tensor=tt_input_2, device=device, parameters=parameters["features"][3]
    )

    tt_int_tensor_4 = tt_Fire(
        128, 16, 64, 64, input_tensor=tt_int_tensor_3, device=device, parameters=parameters["features"][4]
    )

    tt_int_tensor_5 = tt_Fire(
        128, 32, 128, 128, input_tensor=tt_int_tensor_4, device=device, parameters=parameters["features"][5]
    )
    tt_int_tensor_5 = ttnn.permute(tt_int_tensor_5, (0, 3, 1, 2))
    torch_in = ttnn.to_torch(tt_int_tensor_5)
    torch_out_2 = torch_pool(torch_in)
    torch_out_permute_2 = torch.permute(torch_out_2, (0, 2, 3, 1))
    tt_input_3 = ttnn.from_torch(torch_out_permute_2, dtype=ttnn.bfloat16)
    tt_int_tensor_6 = tt_Fire(
        256, 32, 128, 128, input_tensor=tt_input_3, device=device, parameters=parameters["features"][7]
    )

    tt_int_tensor_7 = tt_Fire(
        256, 48, 192, 192, input_tensor=tt_int_tensor_6, device=device, parameters=parameters["features"][8]
    )

    tt_int_tensor_8 = tt_Fire(
        384, 48, 192, 192, input_tensor=tt_int_tensor_7, device=device, parameters=parameters["features"][9]
    )

    tt_int_tensor_9 = tt_Fire(
        384, 64, 256, 256, input_tensor=tt_int_tensor_8, device=device, parameters=parameters["features"][10]
    )

    # pool 3
    tt_int_tensor_9 = ttnn.to_layout(tt_int_tensor_9, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_int_tensor_9 = ttnn.reshape(tt_int_tensor_9, (1, 1, (27 * 27), 512))
    out_pool = ttnn.max_pool2d(
        input_tensor=tt_int_tensor_9,
        batch_size=1,
        input_h=27,
        input_w=27,
        channels=512,
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
        memory_config=None,
        applied_shard_scheme=None,
    )

    out_pool = ttnn.from_device(out_pool)  # bought outside device due to seg fault
    out_pool_reshaped = ttnn.reshape(out_pool, (1, 13, 13, 512))
    out_pool_reshaped = ttnn.to_device(out_pool_reshaped, device=device)
    tt_int_tensor_10 = tt_Fire(
        512, 64, 256, 256, input_tensor=out_pool_reshaped, device=device, parameters=parameters["features"][12]
    )
    classifier_w = parameters.classifier[1].weight
    classifier_b = parameters.classifier[1].bias

    output_tt_tensor_11, out_height_11, out_width_11 = conv(
        activation="relu",
        input_tensor=tt_int_tensor_10,
        weight=classifier_w,
        bias=classifier_b,
        in_channels=classifier_w.shape[1],
        out_channels=classifier_w.shape[0],
        device=device,
        kernel_size=(classifier_w.shape[2], classifier_w.shape[3]),
        padding=(0, 0),
        stride=(1, 1),
        in_h=tt_int_tensor_10.shape[1],
        in_w=tt_int_tensor_10.shape[2],
    )

    output_tt_tensor_11 = ttnn.to_memory_config(output_tt_tensor_11, memory_config=ttnn.L1_MEMORY_CONFIG)
    output_tt_tensor_11 = ttnn.reshape(output_tt_tensor_11, (1, 13, 13, 1000))
    output_tensor = ttnn.global_avg_pool2d(output_tt_tensor_11)

    out_torch = ttnn.to_torch(output_tensor).squeeze(0).squeeze(0)
    out_tt = ttnn.from_torch(out_torch, dtype=ttnn.bfloat16)
    return out_tt
