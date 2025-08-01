# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.yolov4.common import get_mesh_mappers
from models.demos.yolov4.reference import yolov4
from models.demos.yolov4.reference.resblock import ResBlock


def custom_preprocessor(model, name, mesh_mapper=None):
    parameters = {}

    # Helper function to process Conv2d + BatchNorm2d pairs
    def process_conv_bn_pair(conv_layer, bn_layer, base_name):
        parameters[base_name] = {}
        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
        parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, mesh_mapper=mesh_mapper)
        parameters[base_name]["bias"] = ttnn.from_torch(
            torch.reshape(conv_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
        )

    def process_conv_param(conv_layer, base_name):
        parameters[base_name] = {}
        conv_weight, conv_bias = conv_layer.weight, conv_layer.bias
        conv_bias = torch.reshape(conv_bias, (1, 1, 1, -1))

        if conv_weight.shape[0] == 255:
            conv_weight = torch.nn.functional.pad(conv_weight, (0, 0, 0, 0, 0, 0, 0, 1))
        if conv_bias.shape[-1] == 255:
            conv_bias = torch.nn.functional.pad(conv_bias, (0, 1, 0, 0, 0, 0, 0, 0))

        parameters[base_name]["weight"] = ttnn.from_torch(conv_weight, mesh_mapper=mesh_mapper)
        parameters[base_name]["bias"] = ttnn.from_torch(conv_bias, mesh_mapper=mesh_mapper)

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
                parameters[prefix][str(outer_idx)][base_name]["weight"] = ttnn.from_torch(
                    conv_weight, mesh_mapper=mesh_mapper
                )
                parameters[prefix][str(outer_idx)][base_name]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
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


def _create_ds1_model_parameters(conv_args, resolution):
    if resolution == (320, 320):
        conv_args.c1["act_block_h"] = 128
    elif resolution == (640, 640):
        conv_args.c1["act_block_h"] = 256
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
    conv_args.c1["deallocate_activation"] = True
    conv_args.c1["reshard_if_not_optimal"] = False
    conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    if resolution == (320, 320):
        conv_args.c2["act_block_h"] = None
    elif resolution == (640, 640):
        conv_args.c2["act_block_h"] = 320
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
    conv_args.c2["deallocate_activation"] = True
    conv_args.c2["reshard_if_not_optimal"] = False
    conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c3["act_block_h"] = None
    conv_args.c3["deallocate_activation"] = False
    conv_args.c3["reshard_if_not_optimal"] = False
    conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c4["act_block_h"] = None
    conv_args.c4["deallocate_activation"] = True
    conv_args.c4["reshard_if_not_optimal"] = False
    conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c5["act_block_h"] = None
    conv_args.c5["deallocate_activation"] = False
    conv_args.c5["reshard_if_not_optimal"] = False
    conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    if resolution == (320, 320):
        conv_args.c6["act_block_h"] = None
    elif resolution == (640, 640):
        conv_args.c6["act_block_h"] = 256
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
    conv_args.c6["deallocate_activation"] = True
    conv_args.c6["reshard_if_not_optimal"] = False
    conv_args.c6["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c7["act_block_h"] = None
    conv_args.c7["deallocate_activation"] = True
    conv_args.c7["reshard_if_not_optimal"] = False
    conv_args.c7["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c8["act_block_h"] = None
    conv_args.c8["deallocate_activation"] = True
    conv_args.c8["reshard_if_not_optimal"] = False
    conv_args.c8["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def create_ds1_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    _create_ds1_model_parameters(parameters.conv_args, resolution)

    return parameters


def _create_ds2_model_parameters(conv_args):
    conv_args.c1["act_block_h"] = None
    conv_args.c1["deallocate_activation"] = True
    conv_args.c1["reshard_if_not_optimal"] = False
    conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c2["act_block_h"] = None
    conv_args.c2["deallocate_activation"] = False
    conv_args.c2["reshard_if_not_optimal"] = False
    conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c3["act_block_h"] = None
    conv_args.c3["deallocate_activation"] = True
    conv_args.c3["reshard_if_not_optimal"] = False
    conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c4["act_block_h"] = None
    conv_args.c4["deallocate_activation"] = False
    conv_args.c4["reshard_if_not_optimal"] = False
    conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c5["act_block_h"] = None
    conv_args.c5["deallocate_activation"] = True
    conv_args.c5["reshard_if_not_optimal"] = False
    conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.res["0"]["act_block_h"] = None
    conv_args.res["0"]["deallocate_activation"] = False
    conv_args.res["0"]["reshard_if_not_optimal"] = False
    conv_args.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.res["3"]["act_block_h"] = None
    conv_args.res["3"]["deallocate_activation"] = True
    conv_args.res["3"]["reshard_if_not_optimal"] = False
    conv_args.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def create_ds2_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    _create_ds2_model_parameters(parameters.conv_args)

    return parameters


def _create_ds3_model_parameters(conv_args):
    conv_args.c1["act_block_h"] = None
    conv_args.c1["deallocate_activation"] = True
    conv_args.c1["reshard_if_not_optimal"] = False
    conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c2["act_block_h"] = None
    conv_args.c2["deallocate_activation"] = False
    conv_args.c2["reshard_if_not_optimal"] = False
    conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c3["act_block_h"] = None
    conv_args.c3["deallocate_activation"] = True
    conv_args.c3["reshard_if_not_optimal"] = False
    conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c4["act_block_h"] = None
    conv_args.c4["deallocate_activation"] = False
    conv_args.c4["reshard_if_not_optimal"] = False
    conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c5["act_block_h"] = None
    conv_args.c5["deallocate_activation"] = True
    conv_args.c5["reshard_if_not_optimal"] = False
    conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.res["0"]["act_block_h"] = None
    conv_args.res["0"]["deallocate_activation"] = False
    conv_args.res["0"]["reshard_if_not_optimal"] = False
    conv_args.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.res["3"]["act_block_h"] = None
    conv_args.res["3"]["deallocate_activation"] = True
    conv_args.res["3"]["reshard_if_not_optimal"] = False
    conv_args.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


def create_ds3_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    _create_ds3_model_parameters(parameters.conv_args)

    return parameters


def _create_ds4_model_parameters(conv_args):
    conv_args.c1["act_block_h"] = None
    conv_args.c1["deallocate_activation"] = False
    conv_args.c1["reshard_if_not_optimal"] = True
    conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c2["act_block_h"] = None
    conv_args.c2["deallocate_activation"] = False
    conv_args.c2["reshard_if_not_optimal"] = False
    conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c3["act_block_h"] = None
    conv_args.c3["deallocate_activation"] = False
    conv_args.c3["reshard_if_not_optimal"] = False
    conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c4["act_block_h"] = None
    conv_args.c4["deallocate_activation"] = False
    conv_args.c4["reshard_if_not_optimal"] = False
    conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c5["act_block_h"] = None
    conv_args.c5["deallocate_activation"] = True
    conv_args.c5["reshard_if_not_optimal"] = False
    conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.res["0"]["act_block_h"] = None
    conv_args.res["0"]["deallocate_activation"] = False
    conv_args.res["0"]["reshard_if_not_optimal"] = False
    conv_args.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.res["3"]["act_block_h"] = None
    conv_args.res["3"]["deallocate_activation"] = True
    conv_args.res["3"]["reshard_if_not_optimal"] = False
    conv_args.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED


def create_ds4_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    _create_ds4_model_parameters(parameters.conv_args)

    return parameters


def _create_ds5_model_parameters(conv_args):
    conv_args.c1["act_block_h"] = None
    conv_args.c1["deallocate_activation"] = False
    conv_args.c1["reshard_if_not_optimal"] = True
    conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c2["act_block_h"] = None
    conv_args.c2["deallocate_activation"] = False
    conv_args.c2["reshard_if_not_optimal"] = False
    conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.c3["act_block_h"] = None
    conv_args.c3["deallocate_activation"] = True
    conv_args.c3["reshard_if_not_optimal"] = False
    conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c4["act_block_h"] = None
    conv_args.c4["deallocate_activation"] = False
    conv_args.c4["reshard_if_not_optimal"] = False
    conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.c5["act_block_h"] = None
    conv_args.c5["deallocate_activation"] = True
    conv_args.c5["reshard_if_not_optimal"] = False
    conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.res["0"]["act_block_h"] = None
    conv_args.res["0"]["deallocate_activation"] = False
    conv_args.res["0"]["reshard_if_not_optimal"] = False
    conv_args.res["0"]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.res["3"]["act_block_h"] = None
    conv_args.res["3"]["deallocate_activation"] = True
    conv_args.res["3"]["reshard_if_not_optimal"] = False
    conv_args.res["3"]["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED


def create_ds5_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters["resolution"] = resolution
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    _create_ds5_model_parameters(parameters.conv_args)

    return parameters


def _create_neck_model_parameters(conv_args):
    conv_args.c1["act_block_h"] = None
    conv_args.c1["deallocate_activation"] = True
    conv_args.c1["reshard_if_not_optimal"] = True
    conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c2["act_block_h"] = None
    conv_args.c2["deallocate_activation"] = True
    conv_args.c2["reshard_if_not_optimal"] = False
    conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.c3["act_block_h"] = None
    conv_args.c3["deallocate_activation"] = True
    conv_args.c3["reshard_if_not_optimal"] = False
    conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c4["act_block_h"] = None
    conv_args.c4["deallocate_activation"] = True
    conv_args.c4["reshard_if_not_optimal"] = False
    conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c5["act_block_h"] = None
    conv_args.c5["deallocate_activation"] = True
    conv_args.c5["reshard_if_not_optimal"] = False
    conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.c6["act_block_h"] = None
    conv_args.c6["deallocate_activation"] = True
    conv_args.c6["reshard_if_not_optimal"] = False
    conv_args.c6["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c7["act_block_h"] = None
    conv_args.c7["deallocate_activation"] = False
    conv_args.c7["reshard_if_not_optimal"] = False
    conv_args.c7["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.c7_2["act_block_h"] = None
    conv_args.c7_2["deallocate_activation"] = True
    conv_args.c7_2["reshard_if_not_optimal"] = False
    conv_args.c7_2["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c7_3["act_block_h"] = None
    conv_args.c7_3["deallocate_activation"] = True
    conv_args.c7_3["reshard_if_not_optimal"] = False
    conv_args.c7_3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c7_4["act_block_h"] = None
    conv_args.c7_4["deallocate_activation"] = True
    conv_args.c7_4["reshard_if_not_optimal"] = False
    conv_args.c7_4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c7_5["act_block_h"] = None
    conv_args.c7_5["deallocate_activation"] = True
    conv_args.c7_5["reshard_if_not_optimal"] = False
    conv_args.c7_5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c8["act_block_h"] = None
    conv_args.c8["deallocate_activation"] = True
    conv_args.c8["reshard_if_not_optimal"] = False
    conv_args.c8["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c8_2["act_block_h"] = None
    conv_args.c8_2["deallocate_activation"] = True
    conv_args.c8_2["reshard_if_not_optimal"] = False
    conv_args.c8_2["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c9["act_block_h"] = None
    conv_args.c9["deallocate_activation"] = False
    conv_args.c9["reshard_if_not_optimal"] = False
    conv_args.c9["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c9_2["act_block_h"] = None
    conv_args.c9_2["deallocate_activation"] = True
    conv_args.c9_2["reshard_if_not_optimal"] = False
    conv_args.c9_2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c9_3["act_block_h"] = None
    conv_args.c9_3["deallocate_activation"] = True
    conv_args.c9_3["reshard_if_not_optimal"] = False
    conv_args.c9_3["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c9_4["act_block_h"] = None
    conv_args.c9_4["deallocate_activation"] = True
    conv_args.c9_4["reshard_if_not_optimal"] = False
    conv_args.c9_4["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c9_5["act_block_h"] = None
    conv_args.c9_5["deallocate_activation"] = True
    conv_args.c9_5["reshard_if_not_optimal"] = False
    conv_args.c9_5["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c10["act_block_h"] = None
    conv_args.c10["deallocate_activation"] = True
    conv_args.c10["reshard_if_not_optimal"] = False
    conv_args.c10["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c10_2["act_block_h"] = None
    conv_args.c10_2["deallocate_activation"] = True
    conv_args.c10_2["reshard_if_not_optimal"] = False
    conv_args.c10_2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED


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

    _create_neck_model_parameters(parameters.conv_args)

    return parameters


def _create_head_model_parameters(conv_args, resolution):
    conv_args.c1["act_block_h"] = None
    conv_args.c1["deallocate_activation"] = False
    conv_args.c1["reshard_if_not_optimal"] = True
    conv_args.c1["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED

    conv_args.c2["act_block_h"] = None
    conv_args.c2["deallocate_activation"] = True
    conv_args.c2["reshard_if_not_optimal"] = False
    conv_args.c2["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    conv_args.c2["out_channels"] = 256

    conv_args.c3["act_block_h"] = None
    conv_args.c3["deallocate_activation"] = False
    conv_args.c3["reshard_if_not_optimal"] = True
    conv_args.c3["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c4["act_block_h"] = None
    conv_args.c4["deallocate_activation"] = True
    conv_args.c4["reshard_if_not_optimal"] = False
    conv_args.c4["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c5["act_block_h"] = None
    conv_args.c5["deallocate_activation"] = True
    conv_args.c5["reshard_if_not_optimal"] = False
    conv_args.c5["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c6["act_block_h"] = None
    conv_args.c6["deallocate_activation"] = True
    conv_args.c6["reshard_if_not_optimal"] = False
    conv_args.c6["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c7["act_block_h"] = None
    conv_args.c7["deallocate_activation"] = True
    conv_args.c7["reshard_if_not_optimal"] = False
    conv_args.c7["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c8["act_block_h"] = None
    conv_args.c8["deallocate_activation"] = True
    conv_args.c8["reshard_if_not_optimal"] = False
    conv_args.c8["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c9["act_block_h"] = None
    conv_args.c9["deallocate_activation"] = False
    conv_args.c9["reshard_if_not_optimal"] = False
    conv_args.c9["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c10["act_block_h"] = None
    conv_args.c10["deallocate_activation"] = True
    conv_args.c10["reshard_if_not_optimal"] = False
    conv_args.c10["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    conv_args.c10["out_channels"] = 256

    conv_args.c11["act_block_h"] = None
    conv_args.c11["deallocate_activation"] = True
    conv_args.c11["reshard_if_not_optimal"] = True
    if resolution == (320, 320):
        conv_args.c11["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    elif resolution == (640, 640):
        conv_args.c11["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")

    conv_args.c12["act_block_h"] = None
    conv_args.c12["deallocate_activation"] = True
    conv_args.c12["reshard_if_not_optimal"] = False
    conv_args.c12["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c13["act_block_h"] = None
    conv_args.c13["deallocate_activation"] = True
    conv_args.c13["reshard_if_not_optimal"] = False
    conv_args.c13["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.c14["act_block_h"] = None
    conv_args.c14["deallocate_activation"] = True
    conv_args.c14["reshard_if_not_optimal"] = False
    conv_args.c14["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c15["act_block_h"] = None
    conv_args.c15["deallocate_activation"] = True
    conv_args.c15["reshard_if_not_optimal"] = False
    conv_args.c15["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.c16["act_block_h"] = None
    conv_args.c16["deallocate_activation"] = True
    conv_args.c16["reshard_if_not_optimal"] = False
    conv_args.c16["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    conv_args.c17["act_block_h"] = None
    conv_args.c17["deallocate_activation"] = True
    conv_args.c17["reshard_if_not_optimal"] = False
    conv_args.c17["shard_layout"] = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    conv_args.c18["act_block_h"] = None
    conv_args.c18["deallocate_activation"] = True
    conv_args.c18["reshard_if_not_optimal"] = False
    conv_args.c18["shard_layout"] = ttnn.TensorMemoryLayout.BLOCK_SHARDED
    conv_args.c18["out_channels"] = 256


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

    _create_head_model_parameters(parameters.conv_args, resolution)

    return parameters


def create_yolov4_model_parameters(model: yolov4.Yolov4, input_tensor: torch.Tensor, resolution, device):
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    parameters["resolution"] = resolution
    parameters.downsample1["resolution"] = resolution
    parameters.downsample2["resolution"] = resolution
    parameters.downsample3["resolution"] = resolution
    parameters.downsample4["resolution"] = resolution
    parameters.downsample5["resolution"] = resolution
    parameters.neck["resolution"] = resolution
    parameters.head["resolution"] = resolution

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    _create_ds1_model_parameters(parameters.conv_args.downsample1, resolution)
    _create_ds2_model_parameters(parameters.conv_args.downsample2)
    _create_ds3_model_parameters(parameters.conv_args.downsample3)
    _create_ds4_model_parameters(parameters.conv_args.downsample4)
    _create_ds5_model_parameters(parameters.conv_args.downsample5)
    _create_neck_model_parameters(parameters.conv_args.neck)
    _create_head_model_parameters(parameters.conv_args.head, resolution)

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor
