# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from typing import List, Union, Dict

import ttnn

import math

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
}
conv_ttnn_params = [
    [3, 64, 224, 224],
    [64, 64, 224, 224],
    [64, 128, 112, 112],
    [128, 128, 112, 112],
    [128, 256, 56, 56],
    [256, 256, 56, 56],
    [256, 256, 56, 56],
    [256, 512, 28, 28],
    [512, 512, 28, 28],
    [512, 512, 28, 28],
    [512, 512, 14, 14],
    [512, 512, 14, 14],
    [512, 512, 14, 14],
]
conv_feature_ids = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
classifier_ids = [0, 3, 6]
h_override = [None, None, None, None, None, 7 * 32, 7 * 32, None, None, None, None, None, None]


def ttnn_vgg16(
    device,
    tt_x,
    parameters,
    batch_size,
    model_config,
):
    iter_conv_id = 0
    for itr, v in enumerate(cfgs["D"]):
        if v == "M":
            l = list(tt_x.shape)
            in_n, in_c, in_h, in_w = list(tt_x.shape)

            tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
            ttact_d = ttnn.to_device(tt_x, device)
            tt_x = ttnn.max_pool2d(
                input_tensor=ttact_d,
                batch_size=batch_size,
                input_h=int(math.sqrt(in_h / batch_size)),
                input_w=int(math.sqrt(in_h / batch_size)),
                channels=l[3],
                kernel_size=[2, 2],
                stride=[2, 2],
                padding=[0, 0],
                dilation=[1, 1],
            )
            ttnn.deallocate(ttact_d)
            tt_x = ttnn.from_device(tt_x)

        else:
            h_sharding = True

            if conv_ttnn_params[iter_conv_id][0] > 128:
                h_sharding = False
            conv_config = ttnn.Conv2dConfig(
                dtype=model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                activation="relu",
                deallocate_activation=False,
                input_channels_alignment=32,
                reallocate_halo_output=False,
                transpose_shards=True,
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if h_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                reshard_if_not_optimal=True,
                enable_weights_double_buffer=True,
            )
            if h_override[iter_conv_id] is not None:
                conv_config.act_block_h_override = h_override[iter_conv_id]
            compute_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

            tt_weight = parameters.features[conv_feature_ids[iter_conv_id]].weight
            tt_weight = ttnn.to_layout(ttnn.from_device(tt_weight), layout=ttnn.ROW_MAJOR_LAYOUT)
            tt_bias = parameters.features[conv_feature_ids[iter_conv_id]].bias
            tt_bias = ttnn.to_layout(ttnn.from_device(tt_bias), layout=ttnn.ROW_MAJOR_LAYOUT)
            # Call ttnn.conv
            conv_op_cache = {}
            [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
                input_tensor=tt_x,
                weight_tensor=tt_weight,
                in_channels=conv_ttnn_params[iter_conv_id][0],
                out_channels=conv_ttnn_params[iter_conv_id][1],
                device=device,
                bias_tensor=tt_bias,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=batch_size,
                input_height=conv_ttnn_params[iter_conv_id][2],
                input_width=conv_ttnn_params[iter_conv_id][3],
                conv_config=conv_config,
                compute_config=compute_config,
                conv_op_cache=conv_op_cache,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            tt_x = ttnn.from_device(tt_output_tensor_on_device)
            ttnn.deallocate(tt_output_tensor_on_device)
            iter_conv_id += 1

    tt_x = ttnn.to_device(tt_x, device)
    tt_x = ttnn.to_layout(tt_x, ttnn.TILE_LAYOUT)
    tt_x = ttnn.permute(tt_x, (0, 3, 1, 2))
    tt_x = ttnn.reshape(tt_x, (batch_size, 1, 1, -1))

    # Linear 1
    tt_x = ttnn.linear(
        tt_x,
        parameters["classifier"][classifier_ids[0]]["weight"],
        bias=parameters["classifier"][classifier_ids[0]]["bias"],
        activation="relu",
    )

    # Linear 2
    tt_x = ttnn.linear(
        tt_x,
        parameters["classifier"][classifier_ids[1]]["weight"],
        bias=parameters["classifier"][classifier_ids[1]]["bias"],
        activation="relu",
    )

    # Linear 3
    tt_x = ttnn.linear(
        tt_x,
        parameters["classifier"][classifier_ids[2]]["weight"],
        bias=parameters["classifier"][classifier_ids[2]]["bias"],
    )
    return tt_x


conv_feature_ids_2 = [0, 3, 6, 8, 11, 13, 16, 18]
conv_ttnn_params_2 = [
    [3, 64, 224, 224],
    [64, 128, 112, 112],
    [128, 256, 56, 56],
    [256, 256, 56, 56],
    [256, 512, 28, 28],
    [512, 512, 28, 28],
    [512, 512, 14, 14],
    [512, 512, 14, 14],
]
height_override_11 = [None, None, None, 7 * 32, None, None, None, None]


def ttnn_vgg11(
    device,
    tt_x,
    parameters,
    batch_size,
    model_config,
):
    iter_conv_id = 0
    for itr, v in enumerate(cfgs["A"]):
        if v == "M":
            l = list(tt_x.shape)

            in_n, in_c, in_h, in_w = list(tt_x.shape)

            tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
            ttact_d = ttnn.to_device(tt_x, device)
            tt_x = ttnn.max_pool2d(
                input_tensor=ttact_d,
                batch_size=batch_size,
                input_h=int(math.sqrt(in_h / batch_size)),
                input_w=int(math.sqrt(in_h / batch_size)),
                channels=l[3],
                kernel_size=[2, 2],
                stride=[2, 2],
                padding=[0, 0],
                dilation=[1, 1],
            )
            tt_x = ttnn.from_device(tt_x)
            ttnn.deallocate(ttact_d)

        else:
            h_sharding = True
            if conv_ttnn_params_2[iter_conv_id][0] > 128:
                h_sharding = False
            conv_config = ttnn.Conv2dConfig(
                dtype=model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                activation="relu",
                deallocate_activation=False,
                input_channels_alignment=32,
                reallocate_halo_output=False,
                transpose_shards=True,
                shard_layout=(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED if h_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
                ),
                enable_weights_double_buffer=True,
            )
            if height_override_11[iter_conv_id] is not None:
                conv_config.act_block_h_override = height_override_11[iter_conv_id]

            compute_config = ttnn.init_device_compute_kernel_config(
                device.arch(),
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode=True,
                fp32_dest_acc_en=True,
            )
            tt_weight = parameters.features[conv_feature_ids_2[iter_conv_id]].weight
            tt_weight = ttnn.to_layout(ttnn.from_device(tt_weight), layout=ttnn.ROW_MAJOR_LAYOUT)
            tt_bias = parameters.features[conv_feature_ids_2[iter_conv_id]].bias
            tt_bias = ttnn.to_layout(ttnn.from_device(tt_bias), layout=ttnn.ROW_MAJOR_LAYOUT)

            # Call ttnn.conv
            conv_op_cache = {}
            [tt_output_tensor_on_device, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
                input_tensor=tt_x,
                weight_tensor=tt_weight,
                in_channels=conv_ttnn_params_2[iter_conv_id][0],
                out_channels=conv_ttnn_params_2[iter_conv_id][1],
                device=device,
                bias_tensor=tt_bias,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                batch_size=batch_size,
                input_height=conv_ttnn_params_2[iter_conv_id][2],
                input_width=conv_ttnn_params_2[iter_conv_id][3],
                conv_config=conv_config,
                compute_config=compute_config,
                conv_op_cache=conv_op_cache,
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            tt_x = ttnn.from_device(tt_output_tensor_on_device)
            ttnn.deallocate(tt_output_tensor_on_device)
            iter_conv_id += 1

    tt_x = ttnn.to_device(tt_x, device)
    tt_x = ttnn.to_layout(tt_x, ttnn.TILE_LAYOUT)
    tt_x = ttnn.permute(tt_x, (0, 3, 1, 2))
    tt_x = ttnn.reshape(tt_x, (batch_size, 1, 1, -1))

    # Linear 1
    tt_x = ttnn.linear(
        tt_x,
        parameters["classifier"][classifier_ids[0]]["weight"],
        bias=parameters["classifier"][classifier_ids[0]]["bias"],
        activation="relu",
    )

    # Linear 2
    tt_x = ttnn.linear(
        tt_x,
        parameters["classifier"][classifier_ids[1]]["weight"],
        bias=parameters["classifier"][classifier_ids[1]]["bias"],
        activation="relu",
    )

    # Linear 3
    tt_x = ttnn.linear(
        tt_x,
        parameters["classifier"][classifier_ids[2]]["weight"],
        bias=parameters["classifier"][classifier_ids[2]]["bias"],
    )

    return tt_x


def preprocess_conv_parameter(parameter, *, dtype):
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return parameter


def custom_preprocessor(model, name):
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
