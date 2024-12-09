# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
import ttnn
import numpy as np
import torch


def conv(
    activation,
    input_tensor,
    weight,
    bias,
    in_channels,
    out_channels,
    device,
    kernel_size,
    padding,
    stride,
    in_h,
    in_w,
    shard_layout=None,
):
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        activation=activation,
        weights_dtype=ttnn.bfloat16,
        input_channels_alignment=(16 if (in_channels == 16 and input_tensor.shape[-2] == 115) else 32),
        shard_layout=shard_layout,
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.TILE_LAYOUT,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    output_tt_tensor, [out_height, out_width] = ttnn.conv2d(
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
        compute_config=compute_config,
        groups=1,
        return_weights_and_bias=False,
        return_output_dim=True,
        conv_op_cache={},
        debug=False,
        memory_config=None,
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
    batch_size=1,
    dtype=ttnn.bfloat16,
    activation="relu",
):
    sweight = parameters.squeeze.weight
    sbias = parameters.squeeze.bias
    expand1x1_weights = parameters.expand1x1.weight
    expand1x1_bias = parameters.expand1x1.bias
    expand3x3_weights = parameters.expand3x3.weight
    expand3x3_bias = parameters.expand3x3.bias
    shard_b = None if (inplanes == 128 and batch_size == 8) else ttnn.TensorMemoryLayout.BLOCK_SHARDED
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
        shard_layout=shard_b,
    )
    output_tt_tensor_1 = ttnn.sharded_to_interleaved(output_tt_tensor_1, ttnn.L1_MEMORY_CONFIG)
    output_tt_tensor_1 = ttnn.reshape(
        output_tt_tensor_1, (batch_size, out_height_1, out_width_1, output_tt_tensor_1.shape[-1])
    )
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
        in_h=output_tt_tensor_1.shape[1],
        in_w=output_tt_tensor_1.shape[2],
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )
    output_tt_tensor_2 = ttnn.sharded_to_interleaved(output_tt_tensor_2, ttnn.L1_MEMORY_CONFIG)
    output_tt_tensor_2 = ttnn.to_layout(output_tt_tensor_2, layout=ttnn.ROW_MAJOR_LAYOUT)
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
        in_h=output_tt_tensor_1.shape[1],
        in_w=output_tt_tensor_1.shape[2],
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    )
    output_tt_tensor_3 = ttnn.sharded_to_interleaved(output_tt_tensor_3, ttnn.L1_MEMORY_CONFIG)
    output_tt_tensor_3 = ttnn.to_layout(output_tt_tensor_3, layout=ttnn.ROW_MAJOR_LAYOUT)
    final_output = ttnn.concat([output_tt_tensor_2, output_tt_tensor_3], dim=3)
    final_output = ttnn.reshape(
        final_output, (batch_size, out_height_3, out_width_3, expand3x3_planes + expand1x1_planes)
    )
    return final_output


def tt_squeezenet(
    device,
    parameters,
    tt_input,
    dtype=ttnn.bfloat16,
    activation="relu",
    num_classes=1000,
):
    max_pool_in_tt = False  # ceilmode issue
    conv_1_weights = parameters.features[0].weight
    conv_1_bias = parameters.features[0].bias
    batch_size = tt_input.shape[0]
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
        shard_layout=None,
    )
    max_pool_in_tt_1 = False  # PCC Drops once tt_maxpool is enabled and this drop persists after it is disabled
    if max_pool_in_tt_1:
        out_pool = ttnn.max_pool2d(
            input_tensor=output_tt,
            batch_size=batch_size,
            input_h=out_height_1,
            input_w=out_width_1,
            channels=output_tt.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=None,
            applied_shard_scheme=None,
        )
        out_pool_reshaped = ttnn.reshape(out_pool, (batch_size, 54, 54, out_pool.shape[3]))
        tt_input_2 = ttnn.from_device(out_pool_reshaped)
    else:
        torch_tensor_1 = ttnn.to_torch(output_tt)
        torch_tensor_1 = torch.reshape(torch_tensor_1, (batch_size, out_height_1, out_width_1, output_tt.shape[3]))
        torch_tensor_1 = torch.permute(torch_tensor_1, (0, 3, 1, 2))
        torch_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch op
        torch_out = torch_pool(torch_tensor_1).permute(0, 2, 3, 1)
        tt_input_2 = ttnn.from_torch(torch_out, dtype=ttnn.bfloat16)
    ttnn.deallocate(output_tt)
    tt_int_tensor_3 = tt_Fire(
        96,
        16,
        64,
        64,
        input_tensor=tt_input_2,
        device=device,
        parameters=parameters["features"][3],
        batch_size=batch_size,
    )
    tt_int_tensor_4 = tt_Fire(
        128,
        16,
        64,
        64,
        input_tensor=tt_int_tensor_3,
        device=device,
        parameters=parameters["features"][4],
        batch_size=batch_size,
    )
    tt_int_tensor_5 = tt_Fire(
        128,
        32,
        128,
        128,
        input_tensor=tt_int_tensor_4,
        device=device,
        parameters=parameters["features"][5],
        batch_size=batch_size,
    )
    if max_pool_in_tt:  # enable when ceil_mode issue is fixed(#15039)
        tt_int_tensor_5_reshaped = ttnn.reshape(
            tt_int_tensor_5,
            (
                1,
                1,
                (tt_int_tensor_5.shape[0] * tt_int_tensor_5.shape[1] * tt_int_tensor_5.shape[2]),
                tt_int_tensor_5.shape[3],
            ),
        )
        out_pool_2 = ttnn.max_pool2d(
            input_tensor=tt_int_tensor_5_reshaped,
            batch_size=batch_size,
            input_h=tt_int_tensor_5.shape[1],
            input_w=tt_int_tensor_5.shape[2],
            channels=tt_int_tensor_5_reshaped.shape[3],
            kernel_size=[3, 3],
            stride=[2, 2],
            padding=[0, 0],
            dilation=[1, 1],
            memory_config=None,
            applied_shard_scheme=None,
        )
        out_pool_2_reshaped = ttnn.reshape(out_pool_2, (batch_size, 27, 27, out_pool_2.shape[3]))
        tt_input_3 = ttnn.from_device(out_pool_2_reshaped)
    else:
        torch_in = ttnn.to_torch(tt_int_tensor_5).permute(0, 3, 1, 2)
        torch_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)  # torch op
        torch_out_2 = torch_pool(torch_in).permute(0, 2, 3, 1)
        tt_input_3 = ttnn.from_torch(torch_out_2, dtype=ttnn.bfloat16)
    tt_int_tensor_6 = tt_Fire(
        256,
        32,
        128,
        128,
        input_tensor=tt_input_3,
        device=device,
        parameters=parameters["features"][7],
        batch_size=batch_size,
    )
    tt_int_tensor_7 = tt_Fire(
        256,
        48,
        192,
        192,
        input_tensor=tt_int_tensor_6,
        device=device,
        parameters=parameters["features"][8],
        batch_size=batch_size,
    )
    tt_int_tensor_8 = tt_Fire(
        384,
        48,
        192,
        192,
        input_tensor=tt_int_tensor_7,
        device=device,
        parameters=parameters["features"][9],
        batch_size=batch_size,
    )
    tt_int_tensor_9 = tt_Fire(
        384,
        64,
        256,
        256,
        input_tensor=tt_int_tensor_8,
        device=device,
        parameters=parameters["features"][10],
        batch_size=batch_size,
    )
    out_pool_3 = ttnn.max_pool2d(
        input_tensor=tt_int_tensor_9,
        batch_size=batch_size,
        input_h=tt_int_tensor_9.shape[1],
        input_w=tt_int_tensor_9.shape[2],
        channels=tt_int_tensor_9.shape[3],
        kernel_size=[3, 3],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
        memory_config=None,
        applied_shard_scheme=None,
    )
    out_pool_reshaped = ttnn.reshape(out_pool_3, (batch_size, 13, 13, out_pool_3.shape[-1]))
    out_pool_reshaped = ttnn.from_device(out_pool_reshaped)
    tt_int_tensor_10 = tt_Fire(
        512,
        64,
        256,
        256,
        input_tensor=out_pool_reshaped,
        device=device,
        parameters=parameters["features"][12],
        batch_size=batch_size,
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
        shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    )
    output_tt_tensor_11 = ttnn.sharded_to_interleaved(output_tt_tensor_11, ttnn.L1_MEMORY_CONFIG)
    output_tt_tensor_11 = ttnn.to_layout(output_tt_tensor_11, layout=ttnn.ROW_MAJOR_LAYOUT)
    output_tt_tensor_11 = ttnn.reshape(output_tt_tensor_11, (batch_size, out_height_11, out_width_11, num_classes))
    output_tt_tensor_11 = ttnn.to_layout(output_tt_tensor_11, layout=ttnn.TILE_LAYOUT)
    output_tensor = ttnn.global_avg_pool2d(output_tt_tensor_11)
    output_tensor = ttnn.squeeze(output_tensor, dim=1)
    output_tensor = ttnn.squeeze(output_tensor, dim=1)
    return output_tensor
