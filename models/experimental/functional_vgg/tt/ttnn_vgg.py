# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch

from torchvision import models
from typing import List, Union, Dict, cast

import tt_lib
import ttnn

from tt_lib.fallback_ops import fallback_ops
from models.helper_funcs import Linear as TtLinear
from models.utility_functions import (
    is_conv_supported_on_device,
    run_conv_on_device_wrapper,
)
from models.experimental.vgg.vgg_utils import format_tensor

from models.utility_functions import (
    is_grayskull,
    is_wormhole_b0,
    pad_and_fold_conv_activation_for_unity_stride,
)
import math


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters[f"weight"] = model.weight
        parameters[f"bias"] = model.bias
    return parameters


cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64,
        64,
        "M",  #
        128,
        128,
        "M",  #
        256,
        256,
        256,  #
        "M",
        512,
        512,  #
        512,  # -> 0.9901333620190489
        "M",  # -> 0.8176576791584075
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

"""
inp: (8x3x224) x (224) =
conv1_config = ttnn.Conv2dConfig(
                dtype=model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode_enabled=True,
                fp32_dest_acc_enabled=True,
                activation="relu",
                deallocate_activation=False,
                input_channels_alignment=32,
                reallocate_halo_output=False,
                act_block_h_override=0,
                transpose_shards=True,
                height_sharding=True,
            )
"""

# h_override = [64, 64, 64, 32, -] #working
#
h_override = [128, 128, 128, 64, 32, 32, 32, 32, 32, 32, 32, 32, 32]  # Works for batch_size = 4


# tt_dump = 1
def ttnn_vgg16(
    device,
    tt_x,
    parameters,
    batch_size,
    model_config,
):
    tt_dump = 1
    iter_conv_id = 0
    for v in cfgs["D"]:
        if v == "M":
            l = list(tt_x.shape)
            max_pool_reader_patterns_cache = {}
            max_pool_parallel_config_override = {}
            in_n, in_c, in_h, in_w = list(tt_x.shape)

            tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
            maxpool = ttnn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=(0, 0),
                dilation=(1, 1),
                dtype=ttnn.bfloat16,
                device=device,
                batch_size=batch_size,
                input_height=int(math.sqrt(in_h / batch_size)),
                input_width=int(math.sqrt(in_h / batch_size)),
                deallocate_activation=True,
                parallel_config_override=max_pool_parallel_config_override,
                reader_patterns_cache=max_pool_reader_patterns_cache,
                channels=l[3],
            )

            ttact_d = maxpool.copy_input_to_device(tt_x)
            tt_x = maxpool(ttact_d)
            ttnn.deallocate(ttact_d)
            tt_x = maxpool.copy_output_from_device(tt_x)
            max_pool_reader_patterns_cache.clear()
            max_pool_parallel_config_override.clear()
            tt_x_dump = ttnn.to_torch(tt_x)
            torch.save(
                tt_x_dump,
                f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/tt_{tt_dump}",
            )
            tt_dump += 1
        else:
            h_sharding = True
            print(f"Running conv {iter_conv_id}...{tt_x.shape}")
            if conv_ttnn_params[iter_conv_id][0] > 128:  # or conv_ttnn_params[iter_conv_id][1] > 256:
                h_sharding = False
            conv_config = ttnn.Conv2dConfig(
                dtype=model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode_enabled=True,
                fp32_dest_acc_enabled=False,
                packer_l1_accum_enabled=False,
                activation="relu",
                deallocate_activation=False,
                input_channels_alignment=32,
                reallocate_halo_output=False,
                act_block_h_override=h_override[iter_conv_id],
                transpose_shards=True,
                height_sharding=h_sharding,
                reshard_if_not_optimal=True,
            )

            # Prepare ttnn conv
            # Prepare weights and bias
            weight = parameters["features"][conv_feature_ids[iter_conv_id]]["weight"]
            print("weight_shape:", weight.shape)

            tt_weight = ttnn.from_torch(weight, ttnn.bfloat16)
            bias = parameters["features"][conv_feature_ids[iter_conv_id]]["bias"]
            bias = ((bias.unsqueeze(0)).unsqueeze(0)).unsqueeze(0)
            tt_bias = ttnn.from_torch(bias, ttnn.bfloat16)

            # Call ttnn.conv
            conv_op_cache = {}
            [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
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
                conv_op_cache=conv_op_cache,
            )
            tt_x = ttnn.from_device(tt_output_tensor_on_device)
            ttnn.deallocate(tt_output_tensor_on_device)
            iter_conv_id += 1
            tt_x_dump = ttnn.to_torch(tt_x)
            # tt_dump += 1
            torch.save(
                tt_x_dump,
                f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/tt_{tt_dump}",
            )
            tt_dump += 1
            torch.save(
                tt_x_dump,
                f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/tt_{tt_dump}",
            )
            tt_dump += 1
    # Adaptive Pooling layer is redundant
    # torch shape: batch_size, 512, 7, 7

    # tt_x: 1, 1, batch_size x 7 x 7, 512

    # Flatten the tt_x
    # exit()
    # tt_x = ttnn.from_device(tt_x)
    tt_x = ttnn.to_layout(tt_x, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.reshape(tt_x, (batch_size, 7, 7, -1))  # batch_size, 7, 7, 512
    tt_x = ttnn.to_layout(tt_x, layout=ttnn.TILE_LAYOUT)
    tt_x = ttnn.to_device(tt_x, device)
    tt_x = ttnn.permute(tt_x, (0, 3, 1, 2))  # batch_size, 512, 7, 7
    # print("Permuted tensor:", type(tt_x), len(tt_x), tt_x[0].shape)
    # exit()
    tt_x = ttnn.from_device(tt_x)
    # , blocking=True)
    tt_x = ttnn.to_layout(tt_x, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.reshape(tt_x, (batch_size, 1, 1, -1))
    tt_x = ttnn.to_layout(tt_x, layout=ttnn.TILE_LAYOUT)
    tt_x = ttnn.to_device(tt_x, device)

    tt_x_dump = ttnn.to_torch(tt_x)
    torch.save(
        tt_x_dump, f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/tt_flatten"
    )
    tt_dump += 1
    tt_dump = 0
    # Linear 1
    print("\n tt_x shape0:", tt_x.shape)
    tt_x = tt_x @ parameters["classifier"][classifier_ids[0]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[0]]["bias"]
    tt_x_dump = ttnn.to_torch(tt_x)
    torch.save(
        tt_x_dump,
        f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/ttt_{tt_dump}",
    )
    tt_dump += 1
    tt_x = ttnn.relu(tt_x)
    tt_x_dump = ttnn.to_torch(tt_x)
    torch.save(
        tt_x_dump,
        f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/ttt_{tt_dump}",
    )
    tt_dump += 1
    # Linear 2
    tt_x = tt_x @ parameters["classifier"][classifier_ids[1]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[1]]["bias"]
    tt_x_dump = ttnn.to_torch(tt_x)
    torch.save(
        tt_x_dump,
        f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/ttt_{tt_dump}",
    )
    tt_dump += 1
    tt_x = ttnn.relu(tt_x)
    tt_x_dump = ttnn.to_torch(tt_x)
    torch.save(
        tt_x_dump,
        f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/ttt_{tt_dump}",
    )
    tt_dump += 1
    # Linear 3
    tt_x = tt_x @ parameters["classifier"][classifier_ids[2]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[2]]["bias"]
    tt_x_dump = ttnn.to_torch(tt_x)
    torch.save(
        tt_x_dump,
        f"/home/ubuntu/jayasurya/fresh/tt-metal/models/experimental/functional_vgg/dumps_batch1/ttt_{tt_dump}",
    )
    tt_dump += 1
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


def ttnn_vgg11(
    device,
    tt_x,
    parameters,
    batch_size,
    model_config,
    conv_config,
):
    iter_conv_id = 0
    for v in cfgs["A"]:
        if v == "M":
            # Call MaxPool
            l = list(tt_x.shape)
            in_n, in_c, in_h, in_w = list(tt_x.shape)
            max_pool_reader_patterns_cache = {}
            max_pool_parallel_config_override = {}
            in_n, in_c, in_h, in_w = list(tt_x.shape)

            tt_x = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
            maxpool = ttnn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=(0, 0),
                dilation=(1, 1),
                dtype=ttnn.bfloat16,
                device=device,
                batch_size=batch_size,
                input_height=int(math.sqrt(in_h)),
                input_width=int(math.sqrt(in_h)),
                deallocate_activation=True,
                parallel_config_override=max_pool_parallel_config_override,
                reader_patterns_cache=max_pool_reader_patterns_cache,
                channels=l[3],
            )

            ttact_d = maxpool.copy_input_to_device(tt_x)
            tt_x = maxpool(ttact_d)
            ttnn.deallocate(ttact_d)
            tt_x = maxpool.copy_output_from_device(tt_x)
            max_pool_reader_patterns_cache.clear()
            max_pool_parallel_config_override.clear()
        else:
            h_sharding = True
            if conv_ttnn_params_2[iter_conv_id][0] > 256 or conv_ttnn_params_2[iter_conv_id][1] > 256:
                h_sharding = False
            conv_config = ttnn.Conv2dConfig(
                dtype=model_config["ACTIVATIONS_DTYPE"],
                weights_dtype=model_config["WEIGHTS_DTYPE"],
                math_fidelity=model_config["MATH_FIDELITY"],
                math_approx_mode_enabled=True,
                fp32_dest_acc_enabled=True,
                activation="relu",
                deallocate_activation=False,
                input_channels_alignment=32,
                reallocate_halo_output=False,
                act_block_h_override=0,
                transpose_shards=True,
                height_sharding=h_sharding,
            )

            # Prepare ttnn conv
            # Prepare weights and bias
            weight = parameters["features"][conv_feature_ids_2[iter_conv_id]]["weight"]
            tt_weight = ttnn.from_torch(weight, ttnn.bfloat16)
            bias = parameters["features"][conv_feature_ids_2[iter_conv_id]]["bias"]
            bias = ((bias.unsqueeze(0)).unsqueeze(0)).unsqueeze(0)
            tt_bias = ttnn.from_torch(bias, ttnn.bfloat16)

            # Call ttnn.conv
            conv_op_cache = {}
            [tt_output_tensor_on_device, out_height, out_width, weights_device, bias_device] = ttnn.conv2d(
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
                conv_op_cache=conv_op_cache,
            )
            tt_x = ttnn.from_device(tt_output_tensor_on_device)
            ttnn.deallocate(tt_output_tensor_on_device)
            iter_conv_id += 1

    # Adaptive Pooling layer is redundant

    # Flatten the tt_x
    tt_x = ttnn.to_layout(tt_x, layout=ttnn.TILE_LAYOUT)
    tt_x = ttnn.to_device(tt_x, device)
    tt_x = ttnn.permute(tt_x, (0, 3, 1, 2))
    tt_x = ttnn.from_device(tt_x)
    tt_x = ttnn.to_layout(tt_x, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_x = ttnn.reshape(tt_x, (batch_size, 1, 1, -1))
    tt_x = ttnn.to_layout(tt_x, layout=ttnn.TILE_LAYOUT)
    tt_x = ttnn.to_device(tt_x, device)

    # Linear 1
    tt_x = tt_x @ parameters["classifier"][classifier_ids[0]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[0]]["bias"]
    tt_x = ttnn.relu(tt_x)
    # Linear 2
    tt_x = tt_x @ parameters["classifier"][classifier_ids[1]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[1]]["bias"]
    tt_x = ttnn.relu(tt_x)
    # Linear 3
    tt_x = tt_x @ parameters["classifier"][classifier_ids[2]]["weight"]
    tt_x = tt_x + parameters["classifier"][classifier_ids[2]]["bias"]
    return tt_x
