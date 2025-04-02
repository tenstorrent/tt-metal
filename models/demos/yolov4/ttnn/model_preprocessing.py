# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from ttnn.model_preprocessing import (
    fold_batch_norm2d_into_conv2d,
    infer_ttnn_module_args,
    preprocess_model_parameters,
)

import ttnn
from models.demos.yolov4.reference import yolov4
from models.demos.yolov4.reference.resblock import ResBlock


def custom_preprocessor(model, name):
    parameters = {}

    # Helper function to process Conv2d + BatchNorm2d pairs
    def process_conv_bn_pair(conv_layer, bn_layer, base_name):
        parameters[base_name] = {}
        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
        parameters[base_name]["weight"] = ttnn.from_torch(conv_weight)
        parameters[base_name]["bias"] = ttnn.from_torch(torch.reshape(conv_bias, (1, 1, 1, -1)))

    def process_conv_param(conv_layer, base_name):
        parameters[base_name] = {}
        conv_weight, conv_bias = conv_layer.weight, conv_layer.bias
        conv_bias = torch.reshape(conv_bias, (1, 1, 1, -1))

        if conv_weight.shape[0] == 255:
            conv_weight = torch.nn.functional.pad(conv_weight, (0, 0, 0, 0, 0, 0, 0, 1))
        if conv_bias.shape[-1] == 255:
            conv_bias = torch.nn.functional.pad(conv_bias, (0, 1, 0, 0, 0, 0, 0, 0))

        parameters[base_name]["weight"] = ttnn.from_torch(conv_weight)
        parameters[base_name]["bias"] = ttnn.from_torch(conv_bias)

    # Recursive function to process all layers
    def process_layers(layers, prefix=""):
        i = 0
        while i < len(layers):
            layer_name, layer = layers[i]
            full_name = f"{layer_name}" if prefix else layer_name
            if isinstance(layer, torch.nn.Conv2d):
                # Check if the next layer is BatchNorm2d
                if i + 1 < len(layers) and isinstance(layers[i + 1][1], torch.nn.BatchNorm2d):
                    process_conv_bn_pair(layer, layers[i + 1][1], full_name)
                    i += 1  # Skip the BatchNorm layer in the next iteration
                else:
                    # Handle Conv2d without BatchNorm2d (e.g., store as-is or skip)
                    process_conv_param(layer, full_name)
            elif isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)):
                # Recursively process nested layers
                process_layers(list(layer.named_children()), full_name)
            elif isinstance(layer, ResBlock):
                # Special handling for ResBlock
                process_resblock(layer, full_name)
            i += 1

    # Special handling for ResBlock
    def process_resblock(resblock, prefix):
        module_list = resblock.module_list  # Access the ModuleList inside ResBlock
        if prefix not in parameters:
            parameters[prefix] = {}  # Create a nested dictionary for the ResBlock
        for outer_idx, inner_module_list in enumerate(module_list):
            if str(outer_idx) not in parameters[prefix]:
                parameters[prefix][str(outer_idx)] = {}  # Create a nested dictionary for each outer index
            # Process Conv2d at index 0 and 3 of each inner ModuleList
            for inner_idx in [0, 3]:
                conv_layer = inner_module_list[inner_idx]
                bn_layer = inner_module_list[inner_idx + 1]  # BatchNorm2d follows Conv2d
                base_name = f"{inner_idx}"
                parameters[prefix][str(outer_idx)][base_name] = {}
                conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
                parameters[prefix][str(outer_idx)][base_name]["weight"] = ttnn.from_torch(conv_weight)
                parameters[prefix][str(outer_idx)][base_name]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1))
                )

    # Process the model
    if isinstance(
        model,
        (
            yolov4.DownSample1,
            yolov4.DownSample2,
            yolov4.DownSample3,
            yolov4.DownSample4,
            yolov4.DownSample5,
            yolov4.Neck,
            yolov4.Head,
        ),
    ):
        layers = list(model.named_children())
        process_layers(layers, name)
    elif isinstance(model, ResBlock):
        process_resblock(model, name)

    return parameters


def create_yolov4_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    # DS1
    parameters.downsample1["resolution"] = resolution
    if resolution[0] == 320:
        parameters.conv_args.downsample1.c1["act_block_h"] = 128
        parameters.conv_args.downsample1.c1["enable_split_reader"] = True
        parameters.conv_args.downsample1.c1["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c1["deallocate_activation"] = True
        parameters.conv_args.downsample1.c1["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c1["transpose_shards"] = False

        parameters.conv_args.downsample1.c2["act_block_h"] = None
        parameters.conv_args.downsample1.c2["enable_split_reader"] = True
        parameters.conv_args.downsample1.c2["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c2["deallocate_activation"] = True
        parameters.conv_args.downsample1.c2["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c2["transpose_shards"] = False

        parameters.conv_args.downsample1.c3["act_block_h"] = None
        parameters.conv_args.downsample1.c3["enable_split_reader"] = True
        parameters.conv_args.downsample1.c3["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c3["deallocate_activation"] = False
        parameters.conv_args.downsample1.c3["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c3["transpose_shards"] = False

        parameters.conv_args.downsample1.c4["act_block_h"] = None
        parameters.conv_args.downsample1.c4["enable_split_reader"] = True
        parameters.conv_args.downsample1.c4["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c4["deallocate_activation"] = True
        parameters.conv_args.downsample1.c4["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c4["transpose_shards"] = False

        parameters.conv_args.downsample1.c5["act_block_h"] = None
        parameters.conv_args.downsample1.c5["enable_split_reader"] = True
        parameters.conv_args.downsample1.c5["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c5["deallocate_activation"] = False
        parameters.conv_args.downsample1.c5["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c5["transpose_shards"] = False

        parameters.conv_args.downsample1.c6["act_block_h"] = None
        parameters.conv_args.downsample1.c6["enable_split_reader"] = True
        parameters.conv_args.downsample1.c6["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c6["deallocate_activation"] = True
        parameters.conv_args.downsample1.c6["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c6["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c6["transpose_shards"] = False

        parameters.conv_args.downsample1.c7["act_block_h"] = None
        parameters.conv_args.downsample1.c7["enable_split_reader"] = True
        parameters.conv_args.downsample1.c7["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c7["deallocate_activation"] = True
        parameters.conv_args.downsample1.c7["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c7["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c7["transpose_shards"] = False

        parameters.conv_args.downsample1.c8["act_block_h"] = None
        parameters.conv_args.downsample1.c8["enable_split_reader"] = True
        parameters.conv_args.downsample1.c8["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c8["deallocate_activation"] = True
        parameters.conv_args.downsample1.c8["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c8["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c8["transpose_shards"] = False

    else:
        parameters.conv_args.downsample1.c1["act_block_h"] = 128
        parameters.conv_args.downsample1.c1["enable_split_reader"] = True
        parameters.conv_args.downsample1.c1["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c1["deallocate_activation"] = True
        parameters.conv_args.downsample1.c1["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c1["transpose_shards"] = False

        parameters.conv_args.downsample1.c2["act_block_h"] = None
        parameters.conv_args.downsample1.c2["enable_split_reader"] = False
        parameters.conv_args.downsample1.c2["enable_act_double_buffer"] = False
        parameters.conv_args.downsample1.c2["deallocate_activation"] = True
        parameters.conv_args.downsample1.c2["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c2["shard_layout"] = None
        parameters.conv_args.downsample1.c2["transpose_shards"] = False

        parameters.conv_args.downsample1.c3["act_block_h"] = None
        parameters.conv_args.downsample1.c3["enable_split_reader"] = True
        parameters.conv_args.downsample1.c3["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c3["deallocate_activation"] = False
        parameters.conv_args.downsample1.c3["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c3["transpose_shards"] = False

        parameters.conv_args.downsample1.c4["act_block_h"] = None
        parameters.conv_args.downsample1.c4["enable_split_reader"] = True
        parameters.conv_args.downsample1.c4["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c4["deallocate_activation"] = True
        parameters.conv_args.downsample1.c4["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c4["transpose_shards"] = False

        parameters.conv_args.downsample1.c5["act_block_h"] = None
        parameters.conv_args.downsample1.c5["enable_split_reader"] = True
        parameters.conv_args.downsample1.c5["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c5["deallocate_activation"] = False
        parameters.conv_args.downsample1.c5["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c5["transpose_shards"] = False

        parameters.conv_args.downsample1.c6["act_block_h"] = 256
        parameters.conv_args.downsample1.c6["enable_split_reader"] = False
        parameters.conv_args.downsample1.c6["enable_act_double_buffer"] = False
        parameters.conv_args.downsample1.c6["deallocate_activation"] = True
        parameters.conv_args.downsample1.c6["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c6["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c6["transpose_shards"] = False

        parameters.conv_args.downsample1.c7["act_block_h"] = None
        parameters.conv_args.downsample1.c7["enable_split_reader"] = True
        parameters.conv_args.downsample1.c7["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c7["deallocate_activation"] = True
        parameters.conv_args.downsample1.c7["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c7["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c7["transpose_shards"] = False

        parameters.conv_args.downsample1.c8["act_block_h"] = None
        parameters.conv_args.downsample1.c8["enable_split_reader"] = True
        parameters.conv_args.downsample1.c8["enable_act_double_buffer"] = True
        parameters.conv_args.downsample1.c8["deallocate_activation"] = True
        parameters.conv_args.downsample1.c8["reshard_if_not_optimal"] = False
        parameters.conv_args.downsample1.c8["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.downsample1.c8["transpose_shards"] = False

    # DS2
    parameters.downsample2["resolution"] = resolution
    parameters.conv_args.downsample2.c1["act_block_h"] = None
    parameters.conv_args.downsample2.c1["enable_split_reader"] = True
    parameters.conv_args.downsample2.c1["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.c1["deallocate_activation"] = True
    parameters.conv_args.downsample2.c1["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.c1["transpose_shards"] = False

    parameters.conv_args.downsample2.c2["act_block_h"] = None
    parameters.conv_args.downsample2.c2["enable_split_reader"] = True
    parameters.conv_args.downsample2.c2["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.c2["deallocate_activation"] = False
    parameters.conv_args.downsample2.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.c2["transpose_shards"] = False

    parameters.conv_args.downsample2.c3["act_block_h"] = None
    parameters.conv_args.downsample2.c3["enable_split_reader"] = True
    parameters.conv_args.downsample2.c3["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.c3["deallocate_activation"] = True
    parameters.conv_args.downsample2.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.c3["transpose_shards"] = False

    parameters.conv_args.downsample2.c4["act_block_h"] = None
    parameters.conv_args.downsample2.c4["enable_split_reader"] = True
    parameters.conv_args.downsample2.c4["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.c4["deallocate_activation"] = False
    parameters.conv_args.downsample2.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.c4["transpose_shards"] = False

    parameters.conv_args.downsample2.c5["act_block_h"] = None
    parameters.conv_args.downsample2.c5["enable_split_reader"] = True
    parameters.conv_args.downsample2.c5["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.c5["deallocate_activation"] = True
    parameters.conv_args.downsample2.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.c5["transpose_shards"] = False

    parameters.conv_args.downsample2.res["0"]["act_block_h"] = None
    parameters.conv_args.downsample2.res["0"]["enable_split_reader"] = True
    parameters.conv_args.downsample2.res["0"]["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.res["0"]["deallocate_activation"] = False
    parameters.conv_args.downsample2.res["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.res["0"]["transpose_shards"] = False

    parameters.conv_args.downsample2.res["3"]["act_block_h"] = None
    parameters.conv_args.downsample2.res["3"]["enable_split_reader"] = True
    parameters.conv_args.downsample2.res["3"]["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.res["3"]["deallocate_activation"] = True
    parameters.conv_args.downsample2.res["3"]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.res["3"]["transpose_shards"] = False

    parameters.conv_args.downsample2.res[0]["act_block_h"] = None
    parameters.conv_args.downsample2.res[0]["enable_split_reader"] = True
    parameters.conv_args.downsample2.res[0]["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.res[0]["deallocate_activation"] = False
    parameters.conv_args.downsample2.res[0]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.res[0]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.res[0]["transpose_shards"] = False

    parameters.conv_args.downsample2.res[3]["act_block_h"] = None
    parameters.conv_args.downsample2.res[3]["enable_split_reader"] = True
    parameters.conv_args.downsample2.res[3]["enable_act_double_buffer"] = True
    parameters.conv_args.downsample2.res[3]["deallocate_activation"] = True
    parameters.conv_args.downsample2.res[3]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample2.res[3]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample2.res[3]["transpose_shards"] = False

    # DS3
    parameters.downsample3["resolution"] = resolution
    parameters.conv_args.downsample3.c1["act_block_h"] = None
    parameters.conv_args.downsample3.c1["enable_split_reader"] = False
    parameters.conv_args.downsample3.c1["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.c1["deallocate_activation"] = True
    parameters.conv_args.downsample3.c1["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.c1["transpose_shards"] = False

    parameters.conv_args.downsample3.c2["act_block_h"] = None
    parameters.conv_args.downsample3.c2["enable_split_reader"] = False
    parameters.conv_args.downsample3.c2["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.c2["deallocate_activation"] = False
    parameters.conv_args.downsample3.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.c2["transpose_shards"] = False

    parameters.conv_args.downsample3.c3["act_block_h"] = None
    parameters.conv_args.downsample3.c3["enable_split_reader"] = False
    parameters.conv_args.downsample3.c3["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.c3["deallocate_activation"] = True
    parameters.conv_args.downsample3.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.c3["transpose_shards"] = False

    parameters.conv_args.downsample3.c4["act_block_h"] = None
    parameters.conv_args.downsample3.c4["enable_split_reader"] = False
    parameters.conv_args.downsample3.c4["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.c4["deallocate_activation"] = False
    parameters.conv_args.downsample3.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.c4["transpose_shards"] = False

    parameters.conv_args.downsample3.c5["act_block_h"] = None
    parameters.conv_args.downsample3.c5["enable_split_reader"] = False
    parameters.conv_args.downsample3.c5["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.c5["deallocate_activation"] = True
    parameters.conv_args.downsample3.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.c5["transpose_shards"] = False

    parameters.conv_args.downsample3.res["0"]["act_block_h"] = None
    parameters.conv_args.downsample3.res["0"]["enable_split_reader"] = False
    parameters.conv_args.downsample3.res["0"]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.res["0"]["deallocate_activation"] = False
    parameters.conv_args.downsample3.res["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.res["0"]["transpose_shards"] = False

    parameters.conv_args.downsample3.res["3"]["act_block_h"] = None
    parameters.conv_args.downsample3.res["3"]["enable_split_reader"] = False
    parameters.conv_args.downsample3.res["3"]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.res["3"]["deallocate_activation"] = True
    parameters.conv_args.downsample3.res["3"]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.res["3"]["transpose_shards"] = False

    parameters.conv_args.downsample3.res[0]["act_block_h"] = None
    parameters.conv_args.downsample3.res[0]["enable_split_reader"] = False
    parameters.conv_args.downsample3.res[0]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.res[0]["deallocate_activation"] = False
    parameters.conv_args.downsample3.res[0]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.res[0]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.res[0]["transpose_shards"] = False

    parameters.conv_args.downsample3.res[3]["act_block_h"] = None
    parameters.conv_args.downsample3.res[3]["enable_split_reader"] = False
    parameters.conv_args.downsample3.res[3]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample3.res[3]["deallocate_activation"] = True
    parameters.conv_args.downsample3.res[3]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample3.res[3]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample3.res[3]["transpose_shards"] = False

    # DS4
    parameters.downsample4["resolution"] = resolution
    parameters.conv_args.downsample4.c1["act_block_h"] = None
    parameters.conv_args.downsample4.c1["enable_split_reader"] = False
    parameters.conv_args.downsample4.c1["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.c1["deallocate_activation"] = False
    parameters.conv_args.downsample4.c1["reshard_if_not_optimal"] = True
    parameters.conv_args.downsample4.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample4.c1["transpose_shards"] = False

    parameters.conv_args.downsample4.c2["act_block_h"] = None
    parameters.conv_args.downsample4.c2["enable_split_reader"] = False
    parameters.conv_args.downsample4.c2["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.c2["deallocate_activation"] = False
    parameters.conv_args.downsample4.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample4.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample4.c2["transpose_shards"] = False

    parameters.conv_args.downsample4.c3["act_block_h"] = None
    parameters.conv_args.downsample4.c3["enable_split_reader"] = False
    parameters.conv_args.downsample4.c3["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.c3["deallocate_activation"] = False
    parameters.conv_args.downsample4.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample4.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample4.c3["transpose_shards"] = False

    parameters.conv_args.downsample4.c4["act_block_h"] = None
    parameters.conv_args.downsample4.c4["enable_split_reader"] = False
    parameters.conv_args.downsample4.c4["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.c4["deallocate_activation"] = False
    parameters.conv_args.downsample4.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample4.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample4.c4["transpose_shards"] = False

    parameters.conv_args.downsample4.c5["act_block_h"] = None
    parameters.conv_args.downsample4.c5["enable_split_reader"] = False
    parameters.conv_args.downsample4.c5["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.c5["deallocate_activation"] = True
    parameters.conv_args.downsample4.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample4.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample4.c5["transpose_shards"] = False

    parameters.conv_args.downsample4.res["0"]["act_block_h"] = None
    parameters.conv_args.downsample4.res["0"]["enable_split_reader"] = False
    parameters.conv_args.downsample4.res["0"]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.res["0"]["deallocate_activation"] = False
    parameters.conv_args.downsample4.res["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample4.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample4.res["0"]["transpose_shards"] = False

    parameters.conv_args.downsample4.res["3"]["act_block_h"] = None
    parameters.conv_args.downsample4.res["3"]["enable_split_reader"] = False
    parameters.conv_args.downsample4.res["3"]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.res["3"]["deallocate_activation"] = True
    parameters.conv_args.downsample4.res["3"]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample4.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample4.res["3"]["transpose_shards"] = False

    parameters.conv_args.downsample4.res[0]["act_block_h"] = None
    parameters.conv_args.downsample4.res[0]["enable_split_reader"] = False
    parameters.conv_args.downsample4.res[0]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.res[0]["deallocate_activation"] = False
    parameters.conv_args.downsample4.res[0]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample4.res[0]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample4.res[0]["transpose_shards"] = False

    parameters.conv_args.downsample4.res[3]["act_block_h"] = None
    parameters.conv_args.downsample4.res[3]["enable_split_reader"] = False
    parameters.conv_args.downsample4.res[3]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample4.res[3]["deallocate_activation"] = True
    parameters.conv_args.downsample4.res[3]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample4.res[3]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample4.res[3]["transpose_shards"] = False

    # DS5
    parameters.downsample5["resolution"] = resolution
    parameters.conv_args.downsample5.c1["act_block_h"] = None
    parameters.conv_args.downsample5.c1["enable_split_reader"] = False
    parameters.conv_args.downsample5.c1["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.c1["deallocate_activation"] = False
    parameters.conv_args.downsample5.c1["reshard_if_not_optimal"] = True
    parameters.conv_args.downsample5.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample5.c1["transpose_shards"] = False

    parameters.conv_args.downsample5.c2["act_block_h"] = None
    parameters.conv_args.downsample5.c2["enable_split_reader"] = False
    parameters.conv_args.downsample5.c2["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.c2["deallocate_activation"] = False
    parameters.conv_args.downsample5.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample5.c2["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.downsample5.c2["transpose_shards"] = False

    parameters.conv_args.downsample5.c3["act_block_h"] = None
    parameters.conv_args.downsample5.c3["enable_split_reader"] = False
    parameters.conv_args.downsample5.c3["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.c3["deallocate_activation"] = True
    parameters.conv_args.downsample5.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample5.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.downsample5.c3["transpose_shards"] = False

    parameters.conv_args.downsample5.c4["act_block_h"] = None
    parameters.conv_args.downsample5.c4["enable_split_reader"] = False
    parameters.conv_args.downsample5.c4["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.c4["deallocate_activation"] = False
    parameters.conv_args.downsample5.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample5.c4["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.downsample5.c4["transpose_shards"] = False

    parameters.conv_args.downsample5.c5["act_block_h"] = None
    parameters.conv_args.downsample5.c5["enable_split_reader"] = False
    parameters.conv_args.downsample5.c5["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.c5["deallocate_activation"] = True
    parameters.conv_args.downsample5.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample5.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.downsample5.c5["transpose_shards"] = False

    parameters.conv_args.downsample5.res["0"]["act_block_h"] = None
    parameters.conv_args.downsample5.res["0"]["enable_split_reader"] = False
    parameters.conv_args.downsample5.res["0"]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.res["0"]["deallocate_activation"] = False
    parameters.conv_args.downsample5.res["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample5.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.downsample5.res["0"]["transpose_shards"] = False

    parameters.conv_args.downsample5.res["3"]["act_block_h"] = None
    parameters.conv_args.downsample5.res["3"]["enable_split_reader"] = False
    parameters.conv_args.downsample5.res["3"]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.res["3"]["deallocate_activation"] = True
    parameters.conv_args.downsample5.res["3"]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample5.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.downsample5.res["3"]["transpose_shards"] = False

    parameters.conv_args.downsample5.res[0]["act_block_h"] = None
    parameters.conv_args.downsample5.res[0]["enable_split_reader"] = False
    parameters.conv_args.downsample5.res[0]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.res[0]["deallocate_activation"] = False
    parameters.conv_args.downsample5.res[0]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample5.res[0]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.downsample5.res[0]["transpose_shards"] = False

    parameters.conv_args.downsample5.res[3]["act_block_h"] = None
    parameters.conv_args.downsample5.res[3]["enable_split_reader"] = False
    parameters.conv_args.downsample5.res[3]["enable_act_double_buffer"] = False
    parameters.conv_args.downsample5.res[3]["deallocate_activation"] = True
    parameters.conv_args.downsample5.res[3]["reshard_if_not_optimal"] = False
    parameters.conv_args.downsample5.res[3]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.downsample5.res[3]["transpose_shards"] = False

    # neck
    parameters.neck["resolution"] = resolution
    parameters.conv_args.neck.c1["act_block_h"] = None
    parameters.conv_args.neck.c1["enable_split_reader"] = False
    parameters.conv_args.neck.c1["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c1["deallocate_activation"] = True
    parameters.conv_args.neck.c1["reshard_if_not_optimal"] = True
    parameters.conv_args.neck.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c1["transpose_shards"] = False

    parameters.conv_args.neck.c2["act_block_h"] = None
    parameters.conv_args.neck.c2["enable_split_reader"] = False
    parameters.conv_args.neck.c2["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c2["deallocate_activation"] = True
    parameters.conv_args.neck.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c2["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.neck.c2["transpose_shards"] = False

    parameters.conv_args.neck.c3["act_block_h"] = None
    parameters.conv_args.neck.c3["enable_split_reader"] = False
    parameters.conv_args.neck.c3["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c3["deallocate_activation"] = True
    parameters.conv_args.neck.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c3["transpose_shards"] = False

    parameters.conv_args.neck.c4["act_block_h"] = None
    parameters.conv_args.neck.c4["enable_split_reader"] = False
    parameters.conv_args.neck.c4["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c4["deallocate_activation"] = True
    parameters.conv_args.neck.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c4["transpose_shards"] = False

    parameters.conv_args.neck.c5["act_block_h"] = None
    parameters.conv_args.neck.c5["enable_split_reader"] = False
    parameters.conv_args.neck.c5["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c5["deallocate_activation"] = True
    parameters.conv_args.neck.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c5["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.neck.c5["transpose_shards"] = False

    parameters.conv_args.neck.c6["act_block_h"] = None
    parameters.conv_args.neck.c6["enable_split_reader"] = False
    parameters.conv_args.neck.c6["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c6["deallocate_activation"] = True
    parameters.conv_args.neck.c6["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c6["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c6["transpose_shards"] = False

    parameters.conv_args.neck.c7["act_block_h"] = None
    parameters.conv_args.neck.c7["enable_split_reader"] = False
    parameters.conv_args.neck.c7["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c7["deallocate_activation"] = False
    parameters.conv_args.neck.c7["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c7["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.neck.c7["transpose_shards"] = False

    parameters.conv_args.neck.c7_2["act_block_h"] = None
    parameters.conv_args.neck.c7_2["enable_split_reader"] = True
    parameters.conv_args.neck.c7_2["enable_act_double_buffer"] = True
    parameters.conv_args.neck.c7_2["deallocate_activation"] = True
    parameters.conv_args.neck.c7_2["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c7_2["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c7_2["transpose_shards"] = False

    parameters.conv_args.neck.c7_3["act_block_h"] = None
    parameters.conv_args.neck.c7_3["enable_split_reader"] = True
    parameters.conv_args.neck.c7_3["enable_act_double_buffer"] = True
    parameters.conv_args.neck.c7_3["deallocate_activation"] = True
    parameters.conv_args.neck.c7_3["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c7_3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c7_3["transpose_shards"] = False

    parameters.conv_args.neck.c7_4["act_block_h"] = None
    parameters.conv_args.neck.c7_4["enable_split_reader"] = True
    parameters.conv_args.neck.c7_4["enable_act_double_buffer"] = True
    parameters.conv_args.neck.c7_4["deallocate_activation"] = True
    parameters.conv_args.neck.c7_4["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c7_4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c7_4["transpose_shards"] = False

    parameters.conv_args.neck.c7_5["act_block_h"] = None
    parameters.conv_args.neck.c7_5["enable_split_reader"] = True
    parameters.conv_args.neck.c7_5["enable_act_double_buffer"] = True
    parameters.conv_args.neck.c7_5["deallocate_activation"] = True
    parameters.conv_args.neck.c7_5["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c7_5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c7_5["transpose_shards"] = False

    parameters.conv_args.neck.c8["act_block_h"] = None
    parameters.conv_args.neck.c8["enable_split_reader"] = False
    parameters.conv_args.neck.c8["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c8["deallocate_activation"] = True
    parameters.conv_args.neck.c8["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c8["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c8["transpose_shards"] = False

    parameters.conv_args.neck.c8_2["act_block_h"] = None
    parameters.conv_args.neck.c8_2["enable_split_reader"] = False
    parameters.conv_args.neck.c8_2["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c8_2["deallocate_activation"] = True
    parameters.conv_args.neck.c8_2["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c8_2["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c8_2["transpose_shards"] = False

    parameters.conv_args.neck.c9["act_block_h"] = None
    parameters.conv_args.neck.c9["enable_split_reader"] = True
    parameters.conv_args.neck.c9["enable_act_double_buffer"] = True
    parameters.conv_args.neck.c9["deallocate_activation"] = False
    parameters.conv_args.neck.c9["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c9["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.neck.c9["transpose_shards"] = False

    parameters.conv_args.neck.c9_2["act_block_h"] = None
    parameters.conv_args.neck.c9_2["enable_split_reader"] = False
    parameters.conv_args.neck.c9_2["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c9_2["deallocate_activation"] = True
    parameters.conv_args.neck.c9_2["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c9_2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.neck.c9_2["transpose_shards"] = False

    parameters.conv_args.neck.c9_3["act_block_h"] = None
    parameters.conv_args.neck.c9_3["enable_split_reader"] = False
    parameters.conv_args.neck.c9_3["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c9_3["deallocate_activation"] = True
    parameters.conv_args.neck.c9_3["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c9_3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.neck.c9_3["transpose_shards"] = False

    parameters.conv_args.neck.c9_4["act_block_h"] = None
    parameters.conv_args.neck.c9_4["enable_split_reader"] = False
    parameters.conv_args.neck.c9_4["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c9_4["deallocate_activation"] = True
    parameters.conv_args.neck.c9_4["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c9_4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.neck.c9_4["transpose_shards"] = False

    parameters.conv_args.neck.c9_5["act_block_h"] = None
    parameters.conv_args.neck.c9_5["enable_split_reader"] = False
    parameters.conv_args.neck.c9_5["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c9_5["deallocate_activation"] = True
    parameters.conv_args.neck.c9_5["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c9_5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.neck.c9_5["transpose_shards"] = False

    parameters.conv_args.neck.c10["act_block_h"] = None
    parameters.conv_args.neck.c10["enable_split_reader"] = False
    parameters.conv_args.neck.c10["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c10["deallocate_activation"] = True
    parameters.conv_args.neck.c10["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c10["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.neck.c10["transpose_shards"] = False

    parameters.conv_args.neck.c10_2["act_block_h"] = None
    parameters.conv_args.neck.c10_2["enable_split_reader"] = False
    parameters.conv_args.neck.c10_2["enable_act_double_buffer"] = False
    parameters.conv_args.neck.c10_2["deallocate_activation"] = True
    parameters.conv_args.neck.c10_2["reshard_if_not_optimal"] = False
    parameters.conv_args.neck.c10_2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.neck.c10_2["transpose_shards"] = False

    # head
    parameters.head["resolution"] = resolution
    parameters.conv_args.head.c1["act_block_h"] = None
    parameters.conv_args.head.c1["enable_split_reader"] = False
    parameters.conv_args.head.c1["enable_act_double_buffer"] = False
    parameters.conv_args.head.c1["deallocate_activation"] = False
    parameters.conv_args.head.c1["reshard_if_not_optimal"] = True
    parameters.conv_args.head.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.head.c1["transpose_shards"] = False

    parameters.conv_args.head.c2["act_block_h"] = None
    parameters.conv_args.head.c2["enable_split_reader"] = False
    parameters.conv_args.head.c2["enable_act_double_buffer"] = False
    parameters.conv_args.head.c2["deallocate_activation"] = True
    parameters.conv_args.head.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.head.c2["transpose_shards"] = False
    parameters.conv_args.head.c2["out_channels"] = 256

    parameters.conv_args.head.c3["act_block_h"] = None
    parameters.conv_args.head.c3["enable_split_reader"] = False
    parameters.conv_args.head.c3["enable_act_double_buffer"] = False
    parameters.conv_args.head.c3["deallocate_activation"] = False
    parameters.conv_args.head.c3["reshard_if_not_optimal"] = True
    parameters.conv_args.head.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c3["transpose_shards"] = False

    parameters.conv_args.head.c4["act_block_h"] = None
    parameters.conv_args.head.c4["enable_split_reader"] = False
    parameters.conv_args.head.c4["enable_act_double_buffer"] = False
    parameters.conv_args.head.c4["deallocate_activation"] = True
    parameters.conv_args.head.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c4["transpose_shards"] = False

    parameters.conv_args.head.c5["act_block_h"] = None
    parameters.conv_args.head.c5["enable_split_reader"] = False
    parameters.conv_args.head.c5["enable_act_double_buffer"] = False
    parameters.conv_args.head.c5["deallocate_activation"] = True
    parameters.conv_args.head.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c5["transpose_shards"] = False

    parameters.conv_args.head.c6["act_block_h"] = None
    parameters.conv_args.head.c6["enable_split_reader"] = False
    parameters.conv_args.head.c6["enable_act_double_buffer"] = False
    parameters.conv_args.head.c6["deallocate_activation"] = True
    parameters.conv_args.head.c6["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c6["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c6["transpose_shards"] = False

    parameters.conv_args.head.c7["act_block_h"] = None
    parameters.conv_args.head.c7["enable_split_reader"] = False
    parameters.conv_args.head.c7["enable_act_double_buffer"] = False
    parameters.conv_args.head.c7["deallocate_activation"] = True
    parameters.conv_args.head.c7["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c7["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c7["transpose_shards"] = False

    parameters.conv_args.head.c8["act_block_h"] = None
    parameters.conv_args.head.c8["enable_split_reader"] = False
    parameters.conv_args.head.c8["enable_act_double_buffer"] = False
    parameters.conv_args.head.c8["deallocate_activation"] = True
    parameters.conv_args.head.c8["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c8["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c8["transpose_shards"] = False

    parameters.conv_args.head.c9["act_block_h"] = None
    parameters.conv_args.head.c9["enable_split_reader"] = False
    parameters.conv_args.head.c9["enable_act_double_buffer"] = False
    parameters.conv_args.head.c9["deallocate_activation"] = False
    parameters.conv_args.head.c9["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c9["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c9["transpose_shards"] = False

    parameters.conv_args.head.c10["act_block_h"] = None
    parameters.conv_args.head.c10["enable_split_reader"] = False
    parameters.conv_args.head.c10["enable_act_double_buffer"] = False
    parameters.conv_args.head.c10["deallocate_activation"] = True
    parameters.conv_args.head.c10["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c10["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c10["transpose_shards"] = False
    parameters.conv_args.head.c10["out_channels"] = 256

    parameters.conv_args.head.c11["act_block_h"] = None
    parameters.conv_args.head.c11["enable_split_reader"] = False
    parameters.conv_args.head.c11["enable_act_double_buffer"] = False
    parameters.conv_args.head.c11["deallocate_activation"] = True
    parameters.conv_args.head.c11["reshard_if_not_optimal"] = True
    parameters.conv_args.head.c11["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.head.c11["transpose_shards"] = False

    parameters.conv_args.head.c12["act_block_h"] = None
    parameters.conv_args.head.c12["enable_split_reader"] = False
    parameters.conv_args.head.c12["enable_act_double_buffer"] = False
    parameters.conv_args.head.c12["deallocate_activation"] = True
    parameters.conv_args.head.c12["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c12["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c12["transpose_shards"] = False

    parameters.conv_args.head.c13["act_block_h"] = None
    parameters.conv_args.head.c13["enable_split_reader"] = False
    parameters.conv_args.head.c13["enable_act_double_buffer"] = False
    parameters.conv_args.head.c13["deallocate_activation"] = True
    parameters.conv_args.head.c13["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c13["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.head.c13["transpose_shards"] = False

    parameters.conv_args.head.c14["act_block_h"] = None
    parameters.conv_args.head.c14["enable_split_reader"] = False
    parameters.conv_args.head.c14["enable_act_double_buffer"] = False
    parameters.conv_args.head.c14["deallocate_activation"] = True
    parameters.conv_args.head.c14["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c14["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c14["transpose_shards"] = False

    parameters.conv_args.head.c15["act_block_h"] = None
    parameters.conv_args.head.c15["enable_split_reader"] = False
    parameters.conv_args.head.c15["enable_act_double_buffer"] = False
    parameters.conv_args.head.c15["deallocate_activation"] = True
    parameters.conv_args.head.c15["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c15["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.head.c15["transpose_shards"] = False

    parameters.conv_args.head.c16["act_block_h"] = None
    parameters.conv_args.head.c16["enable_split_reader"] = False
    parameters.conv_args.head.c16["enable_act_double_buffer"] = False
    parameters.conv_args.head.c16["deallocate_activation"] = True
    parameters.conv_args.head.c16["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c16["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c16["transpose_shards"] = False

    parameters.conv_args.head.c17["act_block_h"] = None
    parameters.conv_args.head.c17["enable_split_reader"] = False
    parameters.conv_args.head.c17["enable_act_double_buffer"] = False
    parameters.conv_args.head.c17["deallocate_activation"] = True
    parameters.conv_args.head.c17["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c17["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.head.c17["transpose_shards"] = False

    parameters.conv_args.head.c18["act_block_h"] = None
    parameters.conv_args.head.c18["enable_split_reader"] = False
    parameters.conv_args.head.c18["enable_act_double_buffer"] = False
    parameters.conv_args.head.c18["deallocate_activation"] = True
    parameters.conv_args.head.c18["reshard_if_not_optimal"] = False
    parameters.conv_args.head.c18["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.head.c18["transpose_shards"] = False
    parameters.conv_args.head.c18["out_channels"] = 256

    return parameters


def create_ds1_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    # DS1
    if resolution[0] == 320:
        parameters.conv_args.c1["act_block_h"] = 128
        parameters.conv_args.c1["enable_split_reader"] = True
        parameters.conv_args.c1["enable_act_double_buffer"] = True
        parameters.conv_args.c1["deallocate_activation"] = True
        parameters.conv_args.c1["reshard_if_not_optimal"] = False
        parameters.conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c1["transpose_shards"] = False

        parameters.conv_args.c2["act_block_h"] = None
        parameters.conv_args.c2["enable_split_reader"] = True
        parameters.conv_args.c2["enable_act_double_buffer"] = True
        parameters.conv_args.c2["deallocate_activation"] = True
        parameters.conv_args.c2["reshard_if_not_optimal"] = False
        parameters.conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c2["transpose_shards"] = False

        parameters.conv_args.c3["act_block_h"] = None
        parameters.conv_args.c3["enable_split_reader"] = True
        parameters.conv_args.c3["enable_act_double_buffer"] = True
        parameters.conv_args.c3["deallocate_activation"] = False
        parameters.conv_args.c3["reshard_if_not_optimal"] = False
        parameters.conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c3["transpose_shards"] = False

        parameters.conv_args.c4["act_block_h"] = None
        parameters.conv_args.c4["enable_split_reader"] = True
        parameters.conv_args.c4["enable_act_double_buffer"] = True
        parameters.conv_args.c4["deallocate_activation"] = True
        parameters.conv_args.c4["reshard_if_not_optimal"] = False
        parameters.conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c4["transpose_shards"] = False

        parameters.conv_args.c5["act_block_h"] = None
        parameters.conv_args.c5["enable_split_reader"] = True
        parameters.conv_args.c5["enable_act_double_buffer"] = True
        parameters.conv_args.c5["deallocate_activation"] = False
        parameters.conv_args.c5["reshard_if_not_optimal"] = False
        parameters.conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c5["transpose_shards"] = False

        parameters.conv_args.c6["act_block_h"] = None
        parameters.conv_args.c6["enable_split_reader"] = True
        parameters.conv_args.c6["enable_act_double_buffer"] = True
        parameters.conv_args.c6["deallocate_activation"] = True
        parameters.conv_args.c6["reshard_if_not_optimal"] = False
        parameters.conv_args.c6["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c6["transpose_shards"] = False

        parameters.conv_args.c7["act_block_h"] = None
        parameters.conv_args.c7["enable_split_reader"] = True
        parameters.conv_args.c7["enable_act_double_buffer"] = True
        parameters.conv_args.c7["deallocate_activation"] = True
        parameters.conv_args.c7["reshard_if_not_optimal"] = False
        parameters.conv_args.c7["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c7["transpose_shards"] = False

        parameters.conv_args.c8["act_block_h"] = None
        parameters.conv_args.c8["enable_split_reader"] = True
        parameters.conv_args.c8["enable_act_double_buffer"] = True
        parameters.conv_args.c8["deallocate_activation"] = True
        parameters.conv_args.c8["reshard_if_not_optimal"] = False
        parameters.conv_args.c8["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c8["transpose_shards"] = False
    else:
        parameters.conv_args.c1["act_block_h"] = 240
        parameters.conv_args.c1["enable_split_reader"] = False
        parameters.conv_args.c1["enable_act_double_buffer"] = False
        parameters.conv_args.c1["deallocate_activation"] = True
        parameters.conv_args.c1["reshard_if_not_optimal"] = False
        parameters.conv_args.c1["shard_layout"] = None
        parameters.conv_args.c1["transpose_shards"] = True

        parameters.conv_args.c2["act_block_h"] = None
        parameters.conv_args.c2["enable_split_reader"] = False
        parameters.conv_args.c2["enable_act_double_buffer"] = False
        parameters.conv_args.c2["deallocate_activation"] = True
        parameters.conv_args.c2["reshard_if_not_optimal"] = False
        parameters.conv_args.c2["shard_layout"] = None
        parameters.conv_args.c2["transpose_shards"] = False

        parameters.conv_args.c3["act_block_h"] = None
        parameters.conv_args.c3["enable_split_reader"] = True
        parameters.conv_args.c3["enable_act_double_buffer"] = True
        parameters.conv_args.c3["deallocate_activation"] = False
        parameters.conv_args.c3["reshard_if_not_optimal"] = False
        parameters.conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c3["transpose_shards"] = False

        parameters.conv_args.c4["act_block_h"] = None
        parameters.conv_args.c4["enable_split_reader"] = True
        parameters.conv_args.c4["enable_act_double_buffer"] = True
        parameters.conv_args.c4["deallocate_activation"] = True
        parameters.conv_args.c4["reshard_if_not_optimal"] = False
        parameters.conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c4["transpose_shards"] = False

        parameters.conv_args.c5["act_block_h"] = None
        parameters.conv_args.c5["enable_split_reader"] = True
        parameters.conv_args.c5["enable_act_double_buffer"] = True
        parameters.conv_args.c5["deallocate_activation"] = False
        parameters.conv_args.c5["reshard_if_not_optimal"] = False
        parameters.conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c5["transpose_shards"] = False

        parameters.conv_args.c6["act_block_h"] = 240
        parameters.conv_args.c6["enable_split_reader"] = False
        parameters.conv_args.c6["enable_act_double_buffer"] = False
        parameters.conv_args.c6["deallocate_activation"] = True
        parameters.conv_args.c6["reshard_if_not_optimal"] = False
        parameters.conv_args.c6["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c6["transpose_shards"] = False

        parameters.conv_args.c7["act_block_h"] = None
        parameters.conv_args.c7["enable_split_reader"] = True
        parameters.conv_args.c7["enable_act_double_buffer"] = True
        parameters.conv_args.c7["deallocate_activation"] = True
        parameters.conv_args.c7["reshard_if_not_optimal"] = False
        parameters.conv_args.c7["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c7["transpose_shards"] = False

        parameters.conv_args.c8["act_block_h"] = None
        parameters.conv_args.c8["enable_split_reader"] = True
        parameters.conv_args.c8["enable_act_double_buffer"] = True
        parameters.conv_args.c8["deallocate_activation"] = True
        parameters.conv_args.c8["reshard_if_not_optimal"] = False
        parameters.conv_args.c8["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        parameters.conv_args.c8["transpose_shards"] = False

    return parameters


def create_ds2_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    # DS2
    parameters.conv_args.c1["act_block_h"] = None
    parameters.conv_args.c1["enable_split_reader"] = True
    parameters.conv_args.c1["enable_act_double_buffer"] = True
    parameters.conv_args.c1["deallocate_activation"] = True
    parameters.conv_args.c1["reshard_if_not_optimal"] = False
    parameters.conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c1["transpose_shards"] = False

    parameters.conv_args.c2["act_block_h"] = None
    parameters.conv_args.c2["enable_split_reader"] = True
    parameters.conv_args.c2["enable_act_double_buffer"] = True
    parameters.conv_args.c2["deallocate_activation"] = False
    parameters.conv_args.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c2["transpose_shards"] = False

    parameters.conv_args.c3["act_block_h"] = None
    parameters.conv_args.c3["enable_split_reader"] = True
    parameters.conv_args.c3["enable_act_double_buffer"] = True
    parameters.conv_args.c3["deallocate_activation"] = True
    parameters.conv_args.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c3["transpose_shards"] = False

    parameters.conv_args.c4["act_block_h"] = None
    parameters.conv_args.c4["enable_split_reader"] = True
    parameters.conv_args.c4["enable_act_double_buffer"] = True
    parameters.conv_args.c4["deallocate_activation"] = False
    parameters.conv_args.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c4["transpose_shards"] = False

    parameters.conv_args.c5["act_block_h"] = None
    parameters.conv_args.c5["enable_split_reader"] = True
    parameters.conv_args.c5["enable_act_double_buffer"] = True
    parameters.conv_args.c5["deallocate_activation"] = True
    parameters.conv_args.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c5["transpose_shards"] = False

    parameters.conv_args.res["0"]["act_block_h"] = None
    parameters.conv_args.res["0"]["enable_split_reader"] = True
    parameters.conv_args.res["0"]["enable_act_double_buffer"] = True
    parameters.conv_args.res["0"]["deallocate_activation"] = False
    parameters.conv_args.res["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.res["0"]["transpose_shards"] = False

    parameters.conv_args.res["3"]["act_block_h"] = None
    parameters.conv_args.res["3"]["enable_split_reader"] = True
    parameters.conv_args.res["3"]["enable_act_double_buffer"] = True
    parameters.conv_args.res["3"]["deallocate_activation"] = True
    parameters.conv_args.res["3"]["reshard_if_not_optimal"] = False
    parameters.conv_args.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.res["3"]["transpose_shards"] = False

    parameters.conv_args.res[0]["act_block_h"] = None
    parameters.conv_args.res[0]["enable_split_reader"] = True
    parameters.conv_args.res[0]["enable_act_double_buffer"] = True
    parameters.conv_args.res[0]["deallocate_activation"] = False
    parameters.conv_args.res[0]["reshard_if_not_optimal"] = False
    parameters.conv_args.res[0]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.res[0]["transpose_shards"] = False

    parameters.conv_args.res[3]["act_block_h"] = None
    parameters.conv_args.res[3]["enable_split_reader"] = True
    parameters.conv_args.res[3]["enable_act_double_buffer"] = True
    parameters.conv_args.res[3]["deallocate_activation"] = True
    parameters.conv_args.res[3]["reshard_if_not_optimal"] = False
    parameters.conv_args.res[3]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.res[3]["transpose_shards"] = False

    return parameters


def create_ds3_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    # DS3
    parameters.conv_args.c1["act_block_h"] = None
    parameters.conv_args.c1["enable_split_reader"] = False
    parameters.conv_args.c1["enable_act_double_buffer"] = False
    parameters.conv_args.c1["deallocate_activation"] = True
    parameters.conv_args.c1["reshard_if_not_optimal"] = False
    parameters.conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c1["transpose_shards"] = False

    parameters.conv_args.c2["act_block_h"] = None
    parameters.conv_args.c2["enable_split_reader"] = False
    parameters.conv_args.c2["enable_act_double_buffer"] = False
    parameters.conv_args.c2["deallocate_activation"] = False
    parameters.conv_args.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c2["transpose_shards"] = False

    parameters.conv_args.c3["act_block_h"] = None
    parameters.conv_args.c3["enable_split_reader"] = False
    parameters.conv_args.c3["enable_act_double_buffer"] = False
    parameters.conv_args.c3["deallocate_activation"] = True
    parameters.conv_args.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c3["transpose_shards"] = False

    parameters.conv_args.c4["act_block_h"] = None
    parameters.conv_args.c4["enable_split_reader"] = False
    parameters.conv_args.c4["enable_act_double_buffer"] = False
    parameters.conv_args.c4["deallocate_activation"] = False
    parameters.conv_args.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c4["transpose_shards"] = False

    parameters.conv_args.c5["act_block_h"] = None
    parameters.conv_args.c5["enable_split_reader"] = False
    parameters.conv_args.c5["enable_act_double_buffer"] = False
    parameters.conv_args.c5["deallocate_activation"] = True
    parameters.conv_args.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c5["transpose_shards"] = False

    parameters.conv_args.res["0"]["act_block_h"] = None
    parameters.conv_args.res["0"]["enable_split_reader"] = False
    parameters.conv_args.res["0"]["enable_act_double_buffer"] = False
    parameters.conv_args.res["0"]["deallocate_activation"] = False
    parameters.conv_args.res["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.res["0"]["transpose_shards"] = False

    parameters.conv_args.res["3"]["act_block_h"] = None
    parameters.conv_args.res["3"]["enable_split_reader"] = False
    parameters.conv_args.res["3"]["enable_act_double_buffer"] = False
    parameters.conv_args.res["3"]["deallocate_activation"] = True
    parameters.conv_args.res["3"]["reshard_if_not_optimal"] = False
    parameters.conv_args.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.res["3"]["transpose_shards"] = False

    parameters.conv_args.res[0]["act_block_h"] = None
    parameters.conv_args.res[0]["enable_split_reader"] = False
    parameters.conv_args.res[0]["enable_act_double_buffer"] = False
    parameters.conv_args.res[0]["deallocate_activation"] = False
    parameters.conv_args.res[0]["reshard_if_not_optimal"] = False
    parameters.conv_args.res[0]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.res[0]["transpose_shards"] = False

    parameters.conv_args.res[3]["act_block_h"] = None
    parameters.conv_args.res[3]["enable_split_reader"] = False
    parameters.conv_args.res[3]["enable_act_double_buffer"] = False
    parameters.conv_args.res[3]["deallocate_activation"] = True
    parameters.conv_args.res[3]["reshard_if_not_optimal"] = False
    parameters.conv_args.res[3]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.res[3]["transpose_shards"] = False

    return parameters


def create_ds4_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    # DS4
    parameters.conv_args.c1["act_block_h"] = None
    parameters.conv_args.c1["enable_split_reader"] = False
    parameters.conv_args.c1["enable_act_double_buffer"] = False
    parameters.conv_args.c1["deallocate_activation"] = False
    parameters.conv_args.c1["reshard_if_not_optimal"] = True
    parameters.conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c1["transpose_shards"] = False

    parameters.conv_args.c2["act_block_h"] = None
    parameters.conv_args.c2["enable_split_reader"] = False
    parameters.conv_args.c2["enable_act_double_buffer"] = False
    parameters.conv_args.c2["deallocate_activation"] = False
    parameters.conv_args.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c2["transpose_shards"] = False

    parameters.conv_args.c3["act_block_h"] = None
    parameters.conv_args.c3["enable_split_reader"] = False
    parameters.conv_args.c3["enable_act_double_buffer"] = False
    parameters.conv_args.c3["deallocate_activation"] = False
    parameters.conv_args.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c3["transpose_shards"] = False

    parameters.conv_args.c4["act_block_h"] = None
    parameters.conv_args.c4["enable_split_reader"] = False
    parameters.conv_args.c4["enable_act_double_buffer"] = False
    parameters.conv_args.c4["deallocate_activation"] = False
    parameters.conv_args.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c4["transpose_shards"] = False

    parameters.conv_args.c5["act_block_h"] = None
    parameters.conv_args.c5["enable_split_reader"] = False
    parameters.conv_args.c5["enable_act_double_buffer"] = False
    parameters.conv_args.c5["deallocate_activation"] = True
    parameters.conv_args.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c5["transpose_shards"] = False

    parameters.conv_args.res["0"]["act_block_h"] = None
    parameters.conv_args.res["0"]["enable_split_reader"] = False
    parameters.conv_args.res["0"]["enable_act_double_buffer"] = False
    parameters.conv_args.res["0"]["deallocate_activation"] = False
    parameters.conv_args.res["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.res["0"]["transpose_shards"] = False

    parameters.conv_args.res["3"]["act_block_h"] = None
    parameters.conv_args.res["3"]["enable_split_reader"] = False
    parameters.conv_args.res["3"]["enable_act_double_buffer"] = False
    parameters.conv_args.res["3"]["deallocate_activation"] = True
    parameters.conv_args.res["3"]["reshard_if_not_optimal"] = False
    parameters.conv_args.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.res["3"]["transpose_shards"] = False

    parameters.conv_args.res[0]["act_block_h"] = None
    parameters.conv_args.res[0]["enable_split_reader"] = False
    parameters.conv_args.res[0]["enable_act_double_buffer"] = False
    parameters.conv_args.res[0]["deallocate_activation"] = False
    parameters.conv_args.res[0]["reshard_if_not_optimal"] = False
    parameters.conv_args.res[0]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.res[0]["transpose_shards"] = False

    parameters.conv_args.res[3]["act_block_h"] = None
    parameters.conv_args.res[3]["enable_split_reader"] = False
    parameters.conv_args.res[3]["enable_act_double_buffer"] = False
    parameters.conv_args.res[3]["deallocate_activation"] = True
    parameters.conv_args.res[3]["reshard_if_not_optimal"] = False
    parameters.conv_args.res[3]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.res[3]["transpose_shards"] = False

    return parameters


def create_ds5_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    # DS5
    parameters.conv_args.c1["act_block_h"] = None
    parameters.conv_args.c1["enable_split_reader"] = False
    parameters.conv_args.c1["enable_act_double_buffer"] = False
    parameters.conv_args.c1["deallocate_activation"] = False
    parameters.conv_args.c1["reshard_if_not_optimal"] = True
    parameters.conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c1["transpose_shards"] = False

    parameters.conv_args.c2["act_block_h"] = None
    parameters.conv_args.c2["enable_split_reader"] = False
    parameters.conv_args.c2["enable_act_double_buffer"] = False
    parameters.conv_args.c2["deallocate_activation"] = False
    parameters.conv_args.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.c2["transpose_shards"] = False

    parameters.conv_args.c3["act_block_h"] = None
    parameters.conv_args.c3["enable_split_reader"] = False
    parameters.conv_args.c3["enable_act_double_buffer"] = False
    parameters.conv_args.c3["deallocate_activation"] = True
    parameters.conv_args.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c3["transpose_shards"] = False

    parameters.conv_args.c4["act_block_h"] = None
    parameters.conv_args.c4["enable_split_reader"] = False
    parameters.conv_args.c4["enable_act_double_buffer"] = False
    parameters.conv_args.c4["deallocate_activation"] = False
    parameters.conv_args.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.c4["transpose_shards"] = False

    parameters.conv_args.c5["act_block_h"] = None
    parameters.conv_args.c5["enable_split_reader"] = False
    parameters.conv_args.c5["enable_act_double_buffer"] = False
    parameters.conv_args.c5["deallocate_activation"] = True
    parameters.conv_args.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c5["transpose_shards"] = False

    parameters.conv_args.res["0"]["act_block_h"] = None
    parameters.conv_args.res["0"]["enable_split_reader"] = False
    parameters.conv_args.res["0"]["enable_act_double_buffer"] = False
    parameters.conv_args.res["0"]["deallocate_activation"] = False
    parameters.conv_args.res["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.res["0"]["transpose_shards"] = False

    parameters.conv_args.res["3"]["act_block_h"] = None
    parameters.conv_args.res["3"]["enable_split_reader"] = False
    parameters.conv_args.res["3"]["enable_act_double_buffer"] = False
    parameters.conv_args.res["3"]["deallocate_activation"] = True
    parameters.conv_args.res["3"]["reshard_if_not_optimal"] = False
    parameters.conv_args.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.res["3"]["transpose_shards"] = False

    parameters.conv_args.res[0]["act_block_h"] = None
    parameters.conv_args.res[0]["enable_split_reader"] = False
    parameters.conv_args.res[0]["enable_act_double_buffer"] = False
    parameters.conv_args.res[0]["deallocate_activation"] = False
    parameters.conv_args.res[0]["reshard_if_not_optimal"] = False
    parameters.conv_args.res[0]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.res[0]["transpose_shards"] = False

    parameters.conv_args.res[3]["act_block_h"] = None
    parameters.conv_args.res[3]["enable_split_reader"] = False
    parameters.conv_args.res[3]["enable_act_double_buffer"] = False
    parameters.conv_args.res[3]["deallocate_activation"] = True
    parameters.conv_args.res[3]["reshard_if_not_optimal"] = False
    parameters.conv_args.res[3]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.res[3]["transpose_shards"] = False

    return parameters


def create_neck_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensor[0], input_tensor[1], input_tensor[2]), device=None
    )
    # neck
    parameters.conv_args.c1["act_block_h"] = None
    parameters.conv_args.c1["enable_split_reader"] = False
    parameters.conv_args.c1["enable_act_double_buffer"] = False
    parameters.conv_args.c1["deallocate_activation"] = True
    parameters.conv_args.c1["reshard_if_not_optimal"] = True
    parameters.conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c1["transpose_shards"] = False

    parameters.conv_args.c2["act_block_h"] = None
    parameters.conv_args.c2["enable_split_reader"] = False
    parameters.conv_args.c2["enable_act_double_buffer"] = False
    parameters.conv_args.c2["deallocate_activation"] = True
    parameters.conv_args.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.c2["transpose_shards"] = False

    parameters.conv_args.c3["act_block_h"] = None
    parameters.conv_args.c3["enable_split_reader"] = False
    parameters.conv_args.c3["enable_act_double_buffer"] = False
    parameters.conv_args.c3["deallocate_activation"] = True
    parameters.conv_args.c3["reshard_if_not_optimal"] = False
    parameters.conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c3["transpose_shards"] = False

    parameters.conv_args.c4["act_block_h"] = None
    parameters.conv_args.c4["enable_split_reader"] = False
    parameters.conv_args.c4["enable_act_double_buffer"] = False
    parameters.conv_args.c4["deallocate_activation"] = True
    parameters.conv_args.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c4["transpose_shards"] = False

    parameters.conv_args.c5["act_block_h"] = None
    parameters.conv_args.c5["enable_split_reader"] = False
    parameters.conv_args.c5["enable_act_double_buffer"] = False
    parameters.conv_args.c5["deallocate_activation"] = True
    parameters.conv_args.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.c5["transpose_shards"] = False

    parameters.conv_args.c6["act_block_h"] = None
    parameters.conv_args.c6["enable_split_reader"] = False
    parameters.conv_args.c6["enable_act_double_buffer"] = False
    parameters.conv_args.c6["deallocate_activation"] = True
    parameters.conv_args.c6["reshard_if_not_optimal"] = False
    parameters.conv_args.c6["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c6["transpose_shards"] = False

    parameters.conv_args.c7["act_block_h"] = None
    parameters.conv_args.c7["enable_split_reader"] = False
    parameters.conv_args.c7["enable_act_double_buffer"] = False
    parameters.conv_args.c7["deallocate_activation"] = False
    parameters.conv_args.c7["reshard_if_not_optimal"] = False
    parameters.conv_args.c7["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.c7["transpose_shards"] = False

    parameters.conv_args.c7_2["act_block_h"] = None
    parameters.conv_args.c7_2["enable_split_reader"] = True
    parameters.conv_args.c7_2["enable_act_double_buffer"] = True
    parameters.conv_args.c7_2["deallocate_activation"] = True
    parameters.conv_args.c7_2["reshard_if_not_optimal"] = False
    parameters.conv_args.c7_2["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c7_2["transpose_shards"] = False

    parameters.conv_args.c7_3["act_block_h"] = None
    parameters.conv_args.c7_3["enable_split_reader"] = True
    parameters.conv_args.c7_3["enable_act_double_buffer"] = True
    parameters.conv_args.c7_3["deallocate_activation"] = True
    parameters.conv_args.c7_3["reshard_if_not_optimal"] = False
    parameters.conv_args.c7_3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c7_3["transpose_shards"] = False

    parameters.conv_args.c7_4["act_block_h"] = None
    parameters.conv_args.c7_4["enable_split_reader"] = True
    parameters.conv_args.c7_4["enable_act_double_buffer"] = True
    parameters.conv_args.c7_4["deallocate_activation"] = True
    parameters.conv_args.c7_4["reshard_if_not_optimal"] = False
    parameters.conv_args.c7_4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c7_4["transpose_shards"] = False

    parameters.conv_args.c7_5["act_block_h"] = None
    parameters.conv_args.c7_5["enable_split_reader"] = True
    parameters.conv_args.c7_5["enable_act_double_buffer"] = True
    parameters.conv_args.c7_5["deallocate_activation"] = True
    parameters.conv_args.c7_5["reshard_if_not_optimal"] = False
    parameters.conv_args.c7_5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c7_5["transpose_shards"] = False

    parameters.conv_args.c8["act_block_h"] = None
    parameters.conv_args.c8["enable_split_reader"] = False
    parameters.conv_args.c8["enable_act_double_buffer"] = False
    parameters.conv_args.c8["deallocate_activation"] = True
    parameters.conv_args.c8["reshard_if_not_optimal"] = False
    parameters.conv_args.c8["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c8["transpose_shards"] = False

    parameters.conv_args.c8_2["act_block_h"] = None
    parameters.conv_args.c8_2["enable_split_reader"] = False
    parameters.conv_args.c8_2["enable_act_double_buffer"] = False
    parameters.conv_args.c8_2["deallocate_activation"] = True
    parameters.conv_args.c8_2["reshard_if_not_optimal"] = False
    parameters.conv_args.c8_2["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c8_2["transpose_shards"] = False

    parameters.conv_args.c9["act_block_h"] = None
    parameters.conv_args.c9["enable_split_reader"] = True
    parameters.conv_args.c9["enable_act_double_buffer"] = True
    parameters.conv_args.c9["deallocate_activation"] = False
    parameters.conv_args.c9["reshard_if_not_optimal"] = False
    parameters.conv_args.c9["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c9["transpose_shards"] = False

    parameters.conv_args.c9_2["act_block_h"] = None
    parameters.conv_args.c9_2["enable_split_reader"] = False
    parameters.conv_args.c9_2["enable_act_double_buffer"] = False
    parameters.conv_args.c9_2["deallocate_activation"] = True
    parameters.conv_args.c9_2["reshard_if_not_optimal"] = False
    parameters.conv_args.c9_2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c9_2["transpose_shards"] = False

    parameters.conv_args.c9_3["act_block_h"] = None
    parameters.conv_args.c9_3["enable_split_reader"] = False
    parameters.conv_args.c9_3["enable_act_double_buffer"] = False
    parameters.conv_args.c9_3["deallocate_activation"] = True
    parameters.conv_args.c9_3["reshard_if_not_optimal"] = False
    parameters.conv_args.c9_3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c9_3["transpose_shards"] = False

    parameters.conv_args.c9_4["act_block_h"] = None
    parameters.conv_args.c9_4["enable_split_reader"] = False
    parameters.conv_args.c9_4["enable_act_double_buffer"] = False
    parameters.conv_args.c9_4["deallocate_activation"] = True
    parameters.conv_args.c9_4["reshard_if_not_optimal"] = False
    parameters.conv_args.c9_4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c9_4["transpose_shards"] = False

    parameters.conv_args.c9_5["act_block_h"] = None
    parameters.conv_args.c9_5["enable_split_reader"] = False
    parameters.conv_args.c9_5["enable_act_double_buffer"] = False
    parameters.conv_args.c9_5["deallocate_activation"] = True
    parameters.conv_args.c9_5["reshard_if_not_optimal"] = False
    parameters.conv_args.c9_5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c9_5["transpose_shards"] = False

    parameters.conv_args.c10["act_block_h"] = None
    parameters.conv_args.c10["enable_split_reader"] = False
    parameters.conv_args.c10["enable_act_double_buffer"] = False
    parameters.conv_args.c10["deallocate_activation"] = True
    parameters.conv_args.c10["reshard_if_not_optimal"] = False
    parameters.conv_args.c10["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c10["transpose_shards"] = False

    parameters.conv_args.c10_2["act_block_h"] = None
    parameters.conv_args.c10_2["enable_split_reader"] = False
    parameters.conv_args.c10_2["enable_act_double_buffer"] = False
    parameters.conv_args.c10_2["deallocate_activation"] = True
    parameters.conv_args.c10_2["reshard_if_not_optimal"] = False
    parameters.conv_args.c10_2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c10_2["transpose_shards"] = False

    return parameters


def create_head_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensor[0], input_tensor[1], input_tensor[2]), device=None
    )

    # head
    parameters.conv_args.c1["act_block_h"] = None
    parameters.conv_args.c1["enable_split_reader"] = False
    parameters.conv_args.c1["enable_act_double_buffer"] = False
    parameters.conv_args.c1["deallocate_activation"] = False
    parameters.conv_args.c1["reshard_if_not_optimal"] = True
    parameters.conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c1["transpose_shards"] = False

    parameters.conv_args.c2["act_block_h"] = None
    parameters.conv_args.c2["enable_split_reader"] = False
    parameters.conv_args.c2["enable_act_double_buffer"] = False
    parameters.conv_args.c2["deallocate_activation"] = True
    parameters.conv_args.c2["reshard_if_not_optimal"] = False
    parameters.conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c2["transpose_shards"] = False
    parameters.conv_args.c2["out_channels"] = 256

    parameters.conv_args.c3["act_block_h"] = None
    parameters.conv_args.c3["enable_split_reader"] = False
    parameters.conv_args.c3["enable_act_double_buffer"] = False
    parameters.conv_args.c3["deallocate_activation"] = False
    parameters.conv_args.c3["reshard_if_not_optimal"] = True
    parameters.conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c3["transpose_shards"] = False

    parameters.conv_args.c4["act_block_h"] = None
    parameters.conv_args.c4["enable_split_reader"] = False
    parameters.conv_args.c4["enable_act_double_buffer"] = False
    parameters.conv_args.c4["deallocate_activation"] = True
    parameters.conv_args.c4["reshard_if_not_optimal"] = False
    parameters.conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c4["transpose_shards"] = False

    parameters.conv_args.c5["act_block_h"] = None
    parameters.conv_args.c5["enable_split_reader"] = False
    parameters.conv_args.c5["enable_act_double_buffer"] = False
    parameters.conv_args.c5["deallocate_activation"] = True
    parameters.conv_args.c5["reshard_if_not_optimal"] = False
    parameters.conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c5["transpose_shards"] = False

    parameters.conv_args.c6["act_block_h"] = None
    parameters.conv_args.c6["enable_split_reader"] = False
    parameters.conv_args.c6["enable_act_double_buffer"] = False
    parameters.conv_args.c6["deallocate_activation"] = True
    parameters.conv_args.c6["reshard_if_not_optimal"] = False
    parameters.conv_args.c6["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c6["transpose_shards"] = False

    parameters.conv_args.c7["act_block_h"] = None
    parameters.conv_args.c7["enable_split_reader"] = False
    parameters.conv_args.c7["enable_act_double_buffer"] = False
    parameters.conv_args.c7["deallocate_activation"] = True
    parameters.conv_args.c7["reshard_if_not_optimal"] = False
    parameters.conv_args.c7["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c7["transpose_shards"] = False

    parameters.conv_args.c8["act_block_h"] = None
    parameters.conv_args.c8["enable_split_reader"] = False
    parameters.conv_args.c8["enable_act_double_buffer"] = False
    parameters.conv_args.c8["deallocate_activation"] = True
    parameters.conv_args.c8["reshard_if_not_optimal"] = False
    parameters.conv_args.c8["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c8["transpose_shards"] = False

    parameters.conv_args.c9["act_block_h"] = None
    parameters.conv_args.c9["enable_split_reader"] = False
    parameters.conv_args.c9["enable_act_double_buffer"] = False
    parameters.conv_args.c9["deallocate_activation"] = False
    parameters.conv_args.c9["reshard_if_not_optimal"] = False
    parameters.conv_args.c9["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c9["transpose_shards"] = False

    parameters.conv_args.c10["act_block_h"] = None
    parameters.conv_args.c10["enable_split_reader"] = False
    parameters.conv_args.c10["enable_act_double_buffer"] = False
    parameters.conv_args.c10["deallocate_activation"] = True
    parameters.conv_args.c10["reshard_if_not_optimal"] = False
    parameters.conv_args.c10["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c10["transpose_shards"] = False
    parameters.conv_args.c10["out_channels"] = 256

    parameters.conv_args.c11["act_block_h"] = None
    parameters.conv_args.c11["enable_split_reader"] = False
    parameters.conv_args.c11["enable_act_double_buffer"] = False
    parameters.conv_args.c11["deallocate_activation"] = True
    parameters.conv_args.c11["reshard_if_not_optimal"] = True
    if resolution[0] == 320:
        parameters.conv_args.c11["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    else:
        parameters.conv_args.c11["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.c11["transpose_shards"] = False

    parameters.conv_args.c12["act_block_h"] = None
    parameters.conv_args.c12["enable_split_reader"] = False
    parameters.conv_args.c12["enable_act_double_buffer"] = False
    parameters.conv_args.c12["deallocate_activation"] = True
    parameters.conv_args.c12["reshard_if_not_optimal"] = False
    parameters.conv_args.c12["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c12["transpose_shards"] = False

    parameters.conv_args.c13["act_block_h"] = None
    parameters.conv_args.c13["enable_split_reader"] = False
    parameters.conv_args.c13["enable_act_double_buffer"] = False
    parameters.conv_args.c13["deallocate_activation"] = True
    parameters.conv_args.c13["reshard_if_not_optimal"] = False
    parameters.conv_args.c13["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.c13["transpose_shards"] = False

    parameters.conv_args.c14["act_block_h"] = None
    parameters.conv_args.c14["enable_split_reader"] = False
    parameters.conv_args.c14["enable_act_double_buffer"] = False
    parameters.conv_args.c14["deallocate_activation"] = True
    parameters.conv_args.c14["reshard_if_not_optimal"] = False
    parameters.conv_args.c14["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c14["transpose_shards"] = False

    parameters.conv_args.c15["act_block_h"] = None
    parameters.conv_args.c15["enable_split_reader"] = False
    parameters.conv_args.c15["enable_act_double_buffer"] = False
    parameters.conv_args.c15["deallocate_activation"] = True
    parameters.conv_args.c15["reshard_if_not_optimal"] = False
    parameters.conv_args.c15["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.c15["transpose_shards"] = False

    parameters.conv_args.c16["act_block_h"] = None
    parameters.conv_args.c16["enable_split_reader"] = False
    parameters.conv_args.c16["enable_act_double_buffer"] = False
    parameters.conv_args.c16["deallocate_activation"] = True
    parameters.conv_args.c16["reshard_if_not_optimal"] = False
    parameters.conv_args.c16["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c16["transpose_shards"] = False

    parameters.conv_args.c17["act_block_h"] = None
    parameters.conv_args.c17["enable_split_reader"] = False
    parameters.conv_args.c17["enable_act_double_buffer"] = False
    parameters.conv_args.c17["deallocate_activation"] = True
    parameters.conv_args.c17["reshard_if_not_optimal"] = False
    parameters.conv_args.c17["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    parameters.conv_args.c17["transpose_shards"] = False

    parameters.conv_args.c18["act_block_h"] = None
    parameters.conv_args.c18["enable_split_reader"] = False
    parameters.conv_args.c18["enable_act_double_buffer"] = False
    parameters.conv_args.c18["deallocate_activation"] = True
    parameters.conv_args.c18["reshard_if_not_optimal"] = False
    parameters.conv_args.c18["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    parameters.conv_args.c18["transpose_shards"] = False
    parameters.conv_args.c18["out_channels"] = 256

    return parameters
