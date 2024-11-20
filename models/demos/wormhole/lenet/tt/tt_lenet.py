# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import torch.nn.functional as F

from torch import nn


def conv(mesh_device, input_tensor, batch_size, parameters):
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
        device=mesh_device,
        bias_tensor=bias,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=input_tensor.shape[0],
        input_height=input_tensor.shape[1],
        input_width=input_tensor.shape[2],
        conv_config=conv_config,
        conv_op_cache={},
        debug=True,
        groups=1,
    )
    return x, out_height, out_width


def Lenet(
    input_tensor, model, batch_size, num_classes, mesh_device, parameters, reset_seeds, mesh_mapper, mesh_composer
):
    conv1, out_height, out_width = conv(mesh_device, input_tensor, batch_size, parameters.layer1)
    conv1 = ttnn.to_torch(conv1, mesh_composer=mesh_composer)
    conv1 = torch.reshape(conv1, (batch_size, out_height, out_width, conv1.shape[-1]))
    conv1 = torch.permute(conv1, (0, 3, 1, 2))
    max = nn.MaxPool2d(kernel_size=2, stride=2)
    maxpool1 = max(conv1)
    maxpool1 = torch.permute(maxpool1, (0, 2, 3, 1))
    maxpool1 = ttnn.from_torch(maxpool1, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=mesh_mapper)
    maxpool1 = ttnn.from_device(maxpool1)
    conv2, out_height, out_width = conv(mesh_device, maxpool1, batch_size, parameters.layer2)
    conv_2 = ttnn.to_layout(conv2, layout=ttnn.ROW_MAJOR_LAYOUT)
    maxpool_2 = ttnn.max_pool2d(
        input_tensor=conv_2,
        batch_size=input_tensor.shape[0],
        input_h=out_height,
        input_w=out_width,
        channels=conv_2.shape[3],
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )
    maxpool_2 = ttnn.to_torch(maxpool_2, mesh_composer=mesh_composer)
    maxpool_2 = torch.reshape(maxpool_2, (batch_size, 5, 5, maxpool_2.shape[3]))
    maxpool_2 = torch.permute(maxpool_2, (0, 3, 1, 2))

    maxpool_2 = torch.reshape(maxpool_2, (maxpool_2.shape[0], -1))
    maxpool_2 = ttnn.from_torch(
        maxpool_2, dtype=ttnn.bfloat16, device=mesh_device, mesh_mapper=mesh_mapper, layout=ttnn.TILE_LAYOUT
    )

    L1 = ttnn.linear(
        maxpool_2,
        parameters.fc.weight,
        bias=parameters.fc.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )
    L2 = ttnn.linear(
        L1,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )
    L3 = ttnn.linear(
        L2,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return L3
