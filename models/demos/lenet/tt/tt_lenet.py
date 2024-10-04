# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch.nn as nn
import torch


def conv(device, input_tensor, batch_size, parameters):
    weight = parameters.weight
    bias = parameters.bias

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
    # x = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
    [x, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight,
        in_channels=input_tensor.shape[3],
        out_channels=weight.shape[0],
        device=device,
        bias_tensor=bias,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=input_tensor.shape[1],
        input_width=input_tensor.shape[2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=True,
        groups=1,
    )
    return x, out_height, out_width


def Lenet(input_tensor, model, batch_size, num_classes, device, parameters, reset_seeds):
    conv_1, out_height, out_width = conv(device, input_tensor, batch_size, parameters.layer1)

    conv_1 = ttnn.from_device(conv_1)
    conv_1 = ttnn.to_layout(conv_1, layout=ttnn.TILE_LAYOUT)
    conv_1 = ttnn.to_device(conv_1, device=device)
    conv_1 = ttnn.reshape(conv_1, (batch_size, out_height, out_width, conv_1.shape[-1]))
    conv_1 = ttnn.permute(conv_1, (0, 3, 1, 2))
    conv_1 = ttnn.to_torch(conv_1)

    max = nn.MaxPool2d(kernel_size=2, stride=2)
    maxpool_1 = max(conv_1)
    maxpool_1 = ttnn.from_torch(
        maxpool_1, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    maxpool_1 = ttnn.permute(maxpool_1, (0, 2, 3, 1))

    conv_2, out_height, out_width = conv(device, maxpool_1, batch_size, parameters.layer2)

    conv_2 = ttnn.from_device(conv_2)
    conv_2 = ttnn.to_layout(conv_2, layout=ttnn.TILE_LAYOUT)
    conv_2 = ttnn.to_device(conv_2, device=device)
    conv_2 = ttnn.reshape(conv_2, (batch_size, out_height, out_width, conv_2.shape[-1]))
    conv_2 = ttnn.permute(conv_2, (0, 3, 1, 2))
    conv_2 = ttnn.to_torch(conv_2)

    max = nn.MaxPool2d(kernel_size=2, stride=2)
    maxpool_2 = max(conv_2)

    maxpool_2 = ttnn.from_torch(maxpool_2, dtype=ttnn.bfloat16)

    maxpool_2 = ttnn.reshape(maxpool_2, (maxpool_2.shape[0], -1))
    maxpool_2 = ttnn.to_device(maxpool_2, device=device)
    maxpool_2 = ttnn.to_layout(maxpool_2, layout=ttnn.TILE_LAYOUT)

    linear_1 = ttnn.linear(
        maxpool_2,
        parameters.fc.weight,
        bias=parameters.fc.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    linear_2 = ttnn.linear(
        linear_1,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    linear_3 = ttnn.linear(
        linear_2,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=device.core_grid.y, x=device.core_grid.x),
    )

    return linear_3
