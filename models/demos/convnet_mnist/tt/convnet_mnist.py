# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import torch.nn.functional as F
from torch import nn


def convnet_mnist(
    input_tensor,
    parameters,
    device,
):
    batch_size = input_tensor.shape[0]
    torch_maxpool = True

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    x = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)

    tt_weight = parameters.conv1.weight
    tt_bias = parameters.conv1.bias
    conv_kwargs = {
        "in_channels": 1,
        "out_channels": 32,
        "batch_size": batch_size,
        "input_height": input_tensor.shape[1],
        "input_width": input_tensor.shape[2],
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    if not ttnn.is_tensor_storage_on_device(tt_weight):
        tt_weight = ttnn.prepare_conv_weights(
            weight_tensor=tt_weight,
            weights_format="OIHW",
            input_memory_config=ttnn.L1_MEMORY_CONFIG,
            input_layout=x.get_layout(),
            **conv_kwargs,
        )
        tt_weight = ttnn.to_device(tt_weight, device)

    x = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=tt_weight,
        **conv_kwargs,
        compute_config=compute_config,
        conv_op_cache={},
        debug=True,
    )
    x = ttnn.relu(x)

    if torch_maxpool:  # Can be removed once issue #12642 is resolved
        x = ttnn.to_torch(x)
        x = torch.reshape(x, (batch_size, 30, 30, 32))
        x = torch.permute(x, (0, 3, 1, 2))
        x = F.max_pool2d(x, 2)
        x = torch.permute(x, (0, 2, 3, 1))
        x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    else:
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=batch_size,
            input_h=30,
            input_w=30,
            channels=32,
            kernel_size=[2, 2],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
        )

    tt_weight = parameters.conv2.weight
    tt_bias = parameters.conv2.bias
    conv_kwargs = {
        "in_channels": 32,
        "out_channels": 64,
        "batch_size": batch_size,
        "input_height": 15,
        "input_width": 15,
        "kernel_size": (3, 3),
        "stride": (1, 1),
        "padding": (0, 0),
        "dilation": (1, 1),
        "groups": 1,
        "device": device,
        "conv_config": conv_config,
    }

    if not ttnn.is_tensor_storage_on_device(tt_weight):
        tt_weight = ttnn.prepare_conv_weights(
            weight_tensor=tt_weight,
            weights_format="OIHW",
            input_memory_config=ttnn.L1_MEMORY_CONFIG,
            input_layout=x.get_layout(),
            **conv_kwargs,
        )
        tt_weight = ttnn.to_device(tt_weight, device)

    x, [out_height, out_width] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=tt_weight,
        **conv_kwargs,
        conv_op_cache={},
        debug=False,
        return_output_dim=True,
        return_weights_and_bias=False,
    )

    x = ttnn.relu(x)

    if torch_maxpool:  # Can be removed once issue #12642 is resolved
        x = ttnn.to_torch(x)
        x = torch.reshape(x, (batch_size, 13, 13, 64))
        x = torch.permute(x, (0, 3, 1, 2))
        x = F.max_pool2d(x, 2)
        x = ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    else:
        x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=batch_size,
            input_h=out_height,
            input_w=out_width,
            channels=x.shape[-1],
            kernel_size=[2, 2],
            stride=[1, 1],
            padding=[0, 0],
            dilation=[1, 1],
        )
    x = ttnn.from_device(x)
    x = ttnn.reshape(x, (x.shape[0], -1))

    x = ttnn.to_device(x, device)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    x = ttnn.linear(x, parameters.fc1.weight, bias=parameters.fc1.bias, activation="relu")

    x = ttnn.linear(x, parameters.fc2.weight, bias=parameters.fc2.bias)

    output = torch.softmax(ttnn.to_torch(x), dim=-1)
    output = ttnn.from_torch(output, device=device, dtype=ttnn.bfloat16)
    return output


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
