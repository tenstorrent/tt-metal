# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.vgg_unet.reference.vgg_unet import UNetVGG19


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor


def custom_preprocessor(model, name, mesh_mapper=None):
    parameters = {}

    def process_conv_bn_pair(conv_layer, bn_layer, base_name, mesh_mapper=None):
        conv_weight, conv_bias = fold_batch_norm2d_into_conv2d(conv_layer, bn_layer)
        return ttnn.from_torch(conv_weight, mesh_mapper=mesh_mapper), ttnn.from_torch(
            torch.reshape(conv_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
        )

    def process_conv_param(conv_layer, base_name, mesh_mapper=None):
        conv_weight, conv_bias = conv_layer.weight, conv_layer.bias
        conv_bias = torch.reshape(conv_bias, (1, 1, 1, -1))
        return ttnn.from_torch(conv_weight, mesh_mapper=mesh_mapper), ttnn.from_torch(
            conv_bias, mesh_mapper=mesh_mapper
        )

    # Recursive function to process all layers
    def process_layers(layers, prefix="", mesh_mapper=None):
        i = 0
        while i < len(layers):
            layer_name, layer = layers[i]
            if "d1" in layer_name or "d2" in layer_name or "d3" in layer_name or "d4" in layer_name:
                # Process the ConvTranspose2d layer
                conv_transpose_layer = layer.up
                base_name = layer_name
                parameters[base_name] = {}
                conv_transpose_weight = conv_transpose_layer.weight
                conv_transpose_bias = conv_transpose_layer.bias
                parameters[base_name]["up"] = {}
                parameters[base_name]["up"]["weight"] = ttnn.from_torch(conv_transpose_weight, mesh_mapper=mesh_mapper)
                parameters[base_name]["up"]["bias"] = ttnn.from_torch(
                    torch.reshape(conv_transpose_bias, (1, 1, 1, -1)), mesh_mapper=mesh_mapper
                )

                # Process the ConvBlock layers (conv1, bn1, conv2, bn2)
                conv_block = layer.conv_block

                # Process conv1 + bn1
                conv1_weight, conv1_bias = process_conv_bn_pair(
                    conv_block.conv1, conv_block.bn1, f"{layer_name}_conv1", mesh_mapper=mesh_mapper
                )
                parameters[base_name]["conv1"] = {}
                parameters[base_name]["conv1"]["weight"] = conv1_weight
                parameters[base_name]["conv1"]["bias"] = conv1_bias
                # Process conv2 + bn2
                conv2_weight, conv2_bias = process_conv_bn_pair(
                    conv_block.conv2, conv_block.bn2, f"{layer_name}_conv2", mesh_mapper=mesh_mapper
                )
                parameters[base_name]["conv2"] = {}
                parameters[base_name]["conv2"]["weight"] = conv2_weight
                parameters[base_name]["conv2"]["bias"] = conv2_bias
            else:
                full_name = f"{layer_name}" if prefix else layer_name
                if isinstance(layer, torch.nn.Conv2d):
                    # Check if the next layer is BatchNorm2d
                    if i + 1 < len(layers) and isinstance(layers[i + 1][1], torch.nn.BatchNorm2d):
                        weight, bias = process_conv_bn_pair(layer, layers[i + 1][1], full_name, mesh_mapper=mesh_mapper)
                        parameters[full_name] = {}
                        parameters[full_name]["weight"] = weight
                        parameters[full_name]["bias"] = bias
                        i += 1  # Skip the BatchNorm layer in the next iteration
                    else:
                        # Handle Conv2d without BatchNorm2d (e.g., store as-is or skip)
                        weight, bias = process_conv_param(layer, full_name, mesh_mapper=mesh_mapper)
                        parameters[full_name] = {}
                        parameters[full_name]["weight"] = weight
                        parameters[full_name]["bias"] = bias
                elif isinstance(layer, torch.nn.ConvTranspose2d):
                    weight, bias = process_conv_param(layer, full_name, mesh_mapper=mesh_mapper)
                    parameters[full_name] = {}
                    parameters[full_name]["weight"] = weight
                    parameters[full_name]["bias"] = bias
                elif isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)):
                    # Recursively process nested layers
                    process_layers(list(layer.named_children()), full_name)

            i += 1

    layers = list(model.named_children())
    process_layers(layers, name)

    return parameters


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = None
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


def create_vgg_unet_model_parameters(model: UNetVGG19, input_tensor: torch.Tensor, device):
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    parameters.conv_args.s1["0"]["act_block_h"] = None
    parameters.conv_args.s1["0"]["enable_split_reader"] = False
    parameters.conv_args.s1["0"]["enable_act_double_buffer"] = False
    parameters.conv_args.s1["0"]["deallocate_activation"] = True
    parameters.conv_args.s1["0"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s1["0"]["shard_layout"] = None
    parameters.conv_args.s1["0"]["activation"] = "relu"

    parameters.conv_args.s1["2"]["act_block_h"] = None
    parameters.conv_args.s1["2"]["enable_split_reader"] = False
    parameters.conv_args.s1["2"]["enable_act_double_buffer"] = False
    parameters.conv_args.s1["2"]["deallocate_activation"] = True
    parameters.conv_args.s1["2"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s1["2"]["shard_layout"] = None
    parameters.conv_args.s1["2"]["activation"] = "relu"

    parameters.conv_args.s2["5"]["act_block_h"] = None
    parameters.conv_args.s2["5"]["enable_split_reader"] = False
    parameters.conv_args.s2["5"]["enable_act_double_buffer"] = False
    parameters.conv_args.s2["5"]["deallocate_activation"] = True
    parameters.conv_args.s2["5"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s2["5"]["shard_layout"] = None
    parameters.conv_args.s2["5"]["activation"] = "relu"

    parameters.conv_args.s2["7"]["act_block_h"] = None
    parameters.conv_args.s2["7"]["enable_split_reader"] = False
    parameters.conv_args.s2["7"]["enable_act_double_buffer"] = False
    parameters.conv_args.s2["7"]["deallocate_activation"] = True
    parameters.conv_args.s2["7"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s2["7"]["shard_layout"] = None
    parameters.conv_args.s2["7"]["activation"] = "relu"

    parameters.conv_args.s3["10"]["act_block_h"] = None
    parameters.conv_args.s3["10"]["enable_split_reader"] = False
    parameters.conv_args.s3["10"]["enable_act_double_buffer"] = False
    parameters.conv_args.s3["10"]["deallocate_activation"] = True
    parameters.conv_args.s3["10"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s3["10"]["shard_layout"] = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    parameters.conv_args.s3["10"]["activation"] = "relu"

    parameters.conv_args.s3["12"]["act_block_h"] = None
    parameters.conv_args.s3["12"]["enable_split_reader"] = False
    parameters.conv_args.s3["12"]["enable_act_double_buffer"] = False
    parameters.conv_args.s3["12"]["deallocate_activation"] = True
    parameters.conv_args.s3["12"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s3["12"]["shard_layout"] = None
    parameters.conv_args.s3["12"]["activation"] = "relu"

    parameters.conv_args.s3["14"]["act_block_h"] = None
    parameters.conv_args.s3["14"]["enable_split_reader"] = False
    parameters.conv_args.s3["14"]["enable_act_double_buffer"] = False
    parameters.conv_args.s3["14"]["deallocate_activation"] = True
    parameters.conv_args.s3["14"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s3["14"]["shard_layout"] = None
    parameters.conv_args.s3["14"]["activation"] = "relu"

    parameters.conv_args.s3["16"]["act_block_h"] = None
    parameters.conv_args.s3["16"]["enable_split_reader"] = False
    parameters.conv_args.s3["16"]["enable_act_double_buffer"] = False
    parameters.conv_args.s3["16"]["deallocate_activation"] = True
    parameters.conv_args.s3["16"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s3["16"]["shard_layout"] = None
    parameters.conv_args.s3["16"]["activation"] = "relu"

    parameters.conv_args.s4["19"]["act_block_h"] = None
    parameters.conv_args.s4["19"]["enable_split_reader"] = False
    parameters.conv_args.s4["19"]["enable_act_double_buffer"] = False
    parameters.conv_args.s4["19"]["deallocate_activation"] = True
    parameters.conv_args.s4["19"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s4["19"]["shard_layout"] = None
    parameters.conv_args.s4["19"]["activation"] = "relu"

    parameters.conv_args.s4["21"]["act_block_h"] = None
    parameters.conv_args.s4["21"]["enable_split_reader"] = False
    parameters.conv_args.s4["21"]["enable_act_double_buffer"] = False
    parameters.conv_args.s4["21"]["deallocate_activation"] = True
    parameters.conv_args.s4["21"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s4["21"]["shard_layout"] = None
    parameters.conv_args.s4["21"]["activation"] = "relu"

    parameters.conv_args.s4["23"]["act_block_h"] = None
    parameters.conv_args.s4["23"]["enable_split_reader"] = False
    parameters.conv_args.s4["23"]["enable_act_double_buffer"] = False
    parameters.conv_args.s4["23"]["deallocate_activation"] = True
    parameters.conv_args.s4["23"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s4["23"]["shard_layout"] = None
    parameters.conv_args.s4["23"]["activation"] = "relu"

    parameters.conv_args.s4["25"]["act_block_h"] = None
    parameters.conv_args.s4["25"]["enable_split_reader"] = False
    parameters.conv_args.s4["25"]["enable_act_double_buffer"] = False
    parameters.conv_args.s4["25"]["deallocate_activation"] = True
    parameters.conv_args.s4["25"]["reshard_if_not_optimal"] = False
    parameters.conv_args.s4["25"]["shard_layout"] = None
    parameters.conv_args.s4["25"]["activation"] = "relu"

    parameters.conv_args.b1["28"]["act_block_h"] = None
    parameters.conv_args.b1["28"]["enable_split_reader"] = False
    parameters.conv_args.b1["28"]["enable_act_double_buffer"] = False
    parameters.conv_args.b1["28"]["deallocate_activation"] = True
    parameters.conv_args.b1["28"]["reshard_if_not_optimal"] = False
    parameters.conv_args.b1["28"]["shard_layout"] = None
    parameters.conv_args.b1["28"]["activation"] = "relu"

    parameters.conv_args.b1["30"]["act_block_h"] = None
    parameters.conv_args.b1["30"]["enable_split_reader"] = False
    parameters.conv_args.b1["30"]["enable_act_double_buffer"] = False
    parameters.conv_args.b1["30"]["deallocate_activation"] = True
    parameters.conv_args.b1["30"]["reshard_if_not_optimal"] = False
    parameters.conv_args.b1["30"]["shard_layout"] = None
    parameters.conv_args.b1["30"]["activation"] = "relu"

    parameters.conv_args.b1["32"]["act_block_h"] = None
    parameters.conv_args.b1["32"]["enable_split_reader"] = False
    parameters.conv_args.b1["32"]["enable_act_double_buffer"] = False
    parameters.conv_args.b1["32"]["deallocate_activation"] = True
    parameters.conv_args.b1["32"]["reshard_if_not_optimal"] = False
    parameters.conv_args.b1["32"]["shard_layout"] = None
    parameters.conv_args.b1["32"]["activation"] = "relu"

    parameters.conv_args.b1["34"]["act_block_h"] = None
    parameters.conv_args.b1["34"]["enable_split_reader"] = False
    parameters.conv_args.b1["34"]["enable_act_double_buffer"] = False
    parameters.conv_args.b1["34"]["deallocate_activation"] = True
    parameters.conv_args.b1["34"]["reshard_if_not_optimal"] = False
    parameters.conv_args.b1["34"]["shard_layout"] = None
    parameters.conv_args.b1["34"]["activation"] = "relu"

    parameters.conv_args.d1.up["act_block_h"] = None
    parameters.conv_args.d1.up["enable_split_reader"] = False
    parameters.conv_args.d1.up["enable_act_double_buffer"] = False
    parameters.conv_args.d1.up["deallocate_activation"] = False
    parameters.conv_args.d1.up["reshard_if_not_optimal"] = False
    parameters.conv_args.d1.up["shard_layout"] = None
    parameters.conv_args.d1.up["dtype"] = ttnn.bfloat16

    parameters.conv_args.d1.conv_block.conv1["act_block_h"] = None
    parameters.conv_args.d1.conv_block.conv1["enable_split_reader"] = False
    parameters.conv_args.d1.conv_block.conv1["enable_act_double_buffer"] = False
    parameters.conv_args.d1.conv_block.conv1["deallocate_activation"] = True
    parameters.conv_args.d1.conv_block.conv1["reshard_if_not_optimal"] = False
    parameters.conv_args.d1.conv_block.conv1["shard_layout"] = None
    parameters.conv_args.d1.conv_block.conv1["activation"] = "relu"
    parameters.conv_args.d1.conv_block.conv1["padding"] = (1, 1)
    parameters.conv_args.d1.conv_block.conv1["do_sharded_to_interleaved"] = True

    parameters.conv_args.d1.conv_block.conv2["act_block_h"] = None
    parameters.conv_args.d1.conv_block.conv2["enable_split_reader"] = False
    parameters.conv_args.d1.conv_block.conv2["enable_act_double_buffer"] = False
    parameters.conv_args.d1.conv_block.conv2["deallocate_activation"] = True
    parameters.conv_args.d1.conv_block.conv2["reshard_if_not_optimal"] = False
    parameters.conv_args.d1.conv_block.conv2["shard_layout"] = None
    parameters.conv_args.d1.conv_block.conv2["activation"] = "relu"
    parameters.conv_args.d1.conv_block.conv2["padding"] = (1, 1)
    parameters.conv_args.d1.conv_block.conv2["do_sharded_to_interleaved"] = True

    parameters.conv_args.d2.up["act_block_h"] = None
    parameters.conv_args.d2.up["enable_split_reader"] = False
    parameters.conv_args.d2.up["enable_act_double_buffer"] = False
    parameters.conv_args.d2.up["deallocate_activation"] = True
    parameters.conv_args.d2.up["reshard_if_not_optimal"] = False
    parameters.conv_args.d2.up["shard_layout"] = None
    parameters.conv_args.d2.up["dtype"] = ttnn.bfloat16

    parameters.conv_args.d2.conv_block.conv1["act_block_h"] = None
    parameters.conv_args.d2.conv_block.conv1["enable_split_reader"] = False
    parameters.conv_args.d2.conv_block.conv1["enable_act_double_buffer"] = False
    parameters.conv_args.d2.conv_block.conv1["deallocate_activation"] = True
    parameters.conv_args.d2.conv_block.conv1["reshard_if_not_optimal"] = False
    parameters.conv_args.d2.conv_block.conv1["shard_layout"] = None
    parameters.conv_args.d2.conv_block.conv1["activation"] = "relu"
    parameters.conv_args.d2.conv_block.conv1["padding"] = (1, 1)
    parameters.conv_args.d2.conv_block.conv1["do_sharded_to_interleaved"] = True

    parameters.conv_args.d2.conv_block.conv2["act_block_h"] = None
    parameters.conv_args.d2.conv_block.conv2["enable_split_reader"] = False
    parameters.conv_args.d2.conv_block.conv2["enable_act_double_buffer"] = False
    parameters.conv_args.d2.conv_block.conv2["deallocate_activation"] = True
    parameters.conv_args.d2.conv_block.conv2["reshard_if_not_optimal"] = False
    parameters.conv_args.d2.conv_block.conv2["shard_layout"] = None
    parameters.conv_args.d2.conv_block.conv2["activation"] = "relu"
    parameters.conv_args.d2.conv_block.conv2["padding"] = (1, 1)
    parameters.conv_args.d2.conv_block.conv2["do_sharded_to_interleaved"] = False

    parameters.conv_args.d3.up["act_block_h"] = None
    parameters.conv_args.d3.up["enable_split_reader"] = False
    parameters.conv_args.d3.up["enable_act_double_buffer"] = False
    parameters.conv_args.d3.up["deallocate_activation"] = True
    parameters.conv_args.d3.up["reshard_if_not_optimal"] = False
    parameters.conv_args.d3.up["shard_layout"] = None
    parameters.conv_args.d3.up["dtype"] = ttnn.bfloat16

    parameters.conv_args.d3.conv_block.conv1["act_block_h"] = None
    parameters.conv_args.d3.conv_block.conv1["enable_split_reader"] = False
    parameters.conv_args.d3.conv_block.conv1["enable_act_double_buffer"] = False
    parameters.conv_args.d3.conv_block.conv1["deallocate_activation"] = True
    parameters.conv_args.d3.conv_block.conv1["reshard_if_not_optimal"] = False
    parameters.conv_args.d3.conv_block.conv1["shard_layout"] = None
    parameters.conv_args.d3.conv_block.conv1["activation"] = "relu"
    parameters.conv_args.d3.conv_block.conv1["padding"] = (1, 1)
    parameters.conv_args.d3.conv_block.conv1["do_sharded_to_interleaved"] = True

    parameters.conv_args.d3.conv_block.conv2["act_block_h"] = None
    parameters.conv_args.d3.conv_block.conv2["enable_split_reader"] = False
    parameters.conv_args.d3.conv_block.conv2["enable_act_double_buffer"] = False
    parameters.conv_args.d3.conv_block.conv2["deallocate_activation"] = True
    parameters.conv_args.d3.conv_block.conv2["reshard_if_not_optimal"] = False
    parameters.conv_args.d3.conv_block.conv2["shard_layout"] = None
    parameters.conv_args.d3.conv_block.conv2["activation"] = "relu"
    parameters.conv_args.d3.conv_block.conv2["padding"] = (1, 1)
    parameters.conv_args.d3.conv_block.conv2["do_sharded_to_interleaved"] = True

    parameters.conv_args.d4.up["act_block_h"] = None
    parameters.conv_args.d4.up["enable_split_reader"] = False
    parameters.conv_args.d4.up["enable_act_double_buffer"] = False
    parameters.conv_args.d4.up["deallocate_activation"] = True
    parameters.conv_args.d4.up["reshard_if_not_optimal"] = False
    parameters.conv_args.d4.up["shard_layout"] = None
    parameters.conv_args.d4.up["dtype"] = ttnn.bfloat16

    parameters.conv_args.d4.conv_block.conv1["act_block_h"] = None
    parameters.conv_args.d4.conv_block.conv1["enable_split_reader"] = False
    parameters.conv_args.d4.conv_block.conv1["enable_act_double_buffer"] = False
    parameters.conv_args.d4.conv_block.conv1["deallocate_activation"] = True
    parameters.conv_args.d4.conv_block.conv1["reshard_if_not_optimal"] = False
    parameters.conv_args.d4.conv_block.conv1["shard_layout"] = None
    parameters.conv_args.d4.conv_block.conv1["activation"] = "relu"
    parameters.conv_args.d4.conv_block.conv1["padding"] = (1, 1)
    parameters.conv_args.d4.conv_block.conv1["do_sharded_to_interleaved"] = True

    parameters.conv_args.d4.conv_block.conv2["act_block_h"] = None
    parameters.conv_args.d4.conv_block.conv2["enable_split_reader"] = False
    parameters.conv_args.d4.conv_block.conv2["enable_act_double_buffer"] = False
    parameters.conv_args.d4.conv_block.conv2["deallocate_activation"] = True
    parameters.conv_args.d4.conv_block.conv2["reshard_if_not_optimal"] = False
    parameters.conv_args.d4.conv_block.conv2["shard_layout"] = None
    parameters.conv_args.d4.conv_block.conv2["activation"] = "relu"
    parameters.conv_args.d4.conv_block.conv2["padding"] = (1, 1)
    parameters.conv_args.d4.conv_block.conv2["do_sharded_to_interleaved"] = True

    parameters.conv_args.out["act_block_h"] = None
    parameters.conv_args.out["enable_split_reader"] = False
    parameters.conv_args.out["enable_act_double_buffer"] = False
    parameters.conv_args.out["deallocate_activation"] = True
    parameters.conv_args.out["reshard_if_not_optimal"] = False
    parameters.conv_args.out["shard_layout"] = None
    parameters.conv_args.out["activation"] = ""
    parameters.conv_args.out["padding"] = (0, 0)

    return parameters
