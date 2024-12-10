# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def conv(mesh_device, input_tensor, batch_size, parameters):
    weight = parameters.weight
    bias = parameters.bias
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="relu",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=32,
        transpose_shards=False,
        reshard_if_not_optimal=True,
        deallocate_activation=True,
        reallocate_halo_output=True,
    )
    compute_config = ttnn.init_device_compute_kernel_config(
        mesh_device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    x, [out_height, out_width] = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weight,
        in_channels=input_tensor.shape[3],
        out_channels=weight.shape[0],
        device=mesh_device,
        bias_tensor=bias,
        kernel_size=(5, 5),
        stride=(1, 1),
        padding=(0, 0),
        batch_size=batch_size,
        input_height=input_tensor.shape[1],
        input_width=input_tensor.shape[2],
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        groups=1,
        return_output_dim=True,
        return_weights_and_bias=False,
    )
    return x, [out_height, out_width]


def lenet(input_tensor, mesh_device, parameters, mesh_mapper, mesh_composer):
    batch_size = input_tensor.shape[0]
    conv_1, [out_height, out_width] = conv(mesh_device, input_tensor, batch_size, parameters.layer1)
    conv_1 = ttnn.sharded_to_interleaved(conv_1, ttnn.L1_MEMORY_CONFIG)
    conv_1 = ttnn.to_layout(conv_1, layout=ttnn.ROW_MAJOR_LAYOUT)
    conv_1 = ttnn.pad(conv_1, [(0, 10)], value=0.0)

    maxpool_1 = ttnn.max_pool2d(
        input_tensor=conv_1,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=conv_1.shape[3],
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )

    maxpool_1 = ttnn.sharded_to_interleaved(maxpool_1, ttnn.L1_MEMORY_CONFIG)
    maxpool_1 = ttnn.reshape(maxpool_1, (batch_size, 14, 14, maxpool_1.shape[3]))
    conv_2, [out_height, out_width] = conv(mesh_device, maxpool_1, batch_size, parameters.layer2)
    conv_2 = ttnn.to_layout(conv_2, layout=ttnn.ROW_MAJOR_LAYOUT)

    maxpool_2 = ttnn.max_pool2d(
        input_tensor=conv_2,
        batch_size=batch_size,
        input_h=out_height,
        input_w=out_width,
        channels=conv_2.shape[3],
        kernel_size=[2, 2],
        stride=[2, 2],
        padding=[0, 0],
        dilation=[1, 1],
    )
    maxpool_2 = ttnn.sharded_to_interleaved(maxpool_2, ttnn.L1_MEMORY_CONFIG)
    maxpool_2 = ttnn.reshape(maxpool_2, (batch_size, 5, 5, maxpool_2.shape[3]))
    maxpool_2 = ttnn.permute(maxpool_2, (0, 3, 1, 2))
    maxpool_2 = ttnn.reshape(maxpool_2, (maxpool_2.shape[0], -1))
    maxpool_2 = ttnn.from_device(maxpool_2)
    maxpool_2 = ttnn.to_device(maxpool_2, device=mesh_device, memory_config=ttnn.L1_MEMORY_CONFIG)
    maxpool_2 = ttnn.to_layout(maxpool_2, layout=ttnn.TILE_LAYOUT)
    linear_1 = ttnn.linear(
        maxpool_2,
        parameters.fc.weight,
        bias=parameters.fc.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )
    linear_2 = ttnn.linear(
        linear_1,
        parameters.fc1.weight,
        bias=parameters.fc1.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        activation="relu",
    )
    linear_3 = ttnn.linear(
        linear_2,
        parameters.fc2.weight,
        bias=parameters.fc2.bias,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    return linear_3
