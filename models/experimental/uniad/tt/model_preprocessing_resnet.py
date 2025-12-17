# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.uniad.reference.fpn import FPN
from models.experimental.uniad.reference.resnet import ResNet, ModulatedDeformConv2dPack
from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, ResNet):
        if isinstance(model, ResNet):
            parameters["res_model"] = {}

        # Initial conv + bn
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        # Loop over all layers (layer1 to layer4)
        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            prefix = f"layer{layer_idx}"
            parameters["res_model"][prefix] = {}
            for block_idx, block in enumerate(layer):
                parameters["res_model"][prefix][block_idx] = {}

                # conv1, conv2, conv3
                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    if isinstance(conv, ModulatedDeformConv2dPack):
                        parameters["res_model"][prefix][block_idx][conv_name] = {}
                        parameters["res_model"][prefix][block_idx][conv_name]["weight"] = conv.weight
                        parameters["res_model"][prefix][block_idx][conv_name]["bias"] = conv.bias
                        parameters["res_model"][prefix][block_idx][conv_name]["conv_offset"] = {
                            "weight": ttnn.from_torch(conv.conv_offset.weight, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(conv.conv_offset.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                        bn = getattr(block, f"bn{conv_name[-1]}")
                        channel_size = bn.num_features

                        # Extract PyTorch tensors
                        weight_torch = bn.weight if bn.affine else None
                        bias_torch = bn.bias if bn.affine else None
                        batch_mean_torch = bn.running_mean
                        batch_var_torch = bn.running_var

                        # Reshape for broadcast compatibility (1, C, 1, 1)
                        batch_mean_torch = batch_mean_torch.view(1, channel_size, 1, 1)
                        batch_var_torch = batch_var_torch.view(1, channel_size, 1, 1)
                        weight_torch = weight_torch.view(1, channel_size, 1, 1) if weight_torch is not None else None
                        bias_torch = bias_torch.view(1, channel_size, 1, 1) if bias_torch is not None else None

                        parameters["res_model"][prefix][block_idx]["bn2"] = {}
                        weight = (
                            ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if weight_torch is not None
                            else None
                        )
                        parameters["res_model"][prefix][block_idx]["bn2"]["weight"] = weight

                        bias = (
                            ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if bias_torch is not None
                            else None
                        )
                        parameters["res_model"][prefix][block_idx]["bn2"]["bias"] = bias

                        running_mean = ttnn.from_torch(batch_mean_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        parameters["res_model"][prefix][block_idx]["bn2"]["running_mean"] = running_mean

                        running_var = ttnn.from_torch(batch_var_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                        parameters["res_model"][prefix][block_idx]["bn2"]["running_var"] = running_var

                        parameters["res_model"][prefix][block_idx]["bn2"][
                            "eps"
                        ] = bn.eps  # scalar, used directly in ops

                    else:
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                # downsample (if present)
                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        conv = ds[0]
                        bn = ds[1]
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

    if isinstance(model, FPN):
        parameters["fpn"] = {}

        # Lateral Convs
        parameters["fpn"]["lateral_convs"] = {}

        parameters["fpn"]["lateral_convs"]["0"] = {}
        parameters["fpn"]["lateral_convs"]["0"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[0].conv.weight, dtype=ttnn.float32
        )
        bias = model.lateral_convs[0].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["height"] = 80
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["width"] = 45
        parameters["fpn"]["lateral_convs"]["0"]["conv"]["batch"] = 6

        parameters["fpn"]["lateral_convs"]["1"] = {}
        parameters["fpn"]["lateral_convs"]["1"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[1].conv.weight, dtype=ttnn.float32
        )
        bias = model.lateral_convs[1].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["height"] = 40
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["width"] = 23
        parameters["fpn"]["lateral_convs"]["1"]["conv"]["batch"] = 6

        parameters["fpn"]["lateral_convs"]["2"] = {}
        parameters["fpn"]["lateral_convs"]["2"]["conv"] = {}
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["weight"] = ttnn.from_torch(
            model.lateral_convs[2].conv.weight, dtype=ttnn.float32
        )
        bias = model.lateral_convs[2].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["height"] = 20
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["width"] = 12
        parameters["fpn"]["lateral_convs"]["2"]["conv"]["batch"] = 6
        # FPN Convs
        parameters["fpn"]["fpn_convs"] = {}

        parameters["fpn"]["fpn_convs"]["0"] = {}
        parameters["fpn"]["fpn_convs"]["0"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[0].conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs[0].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["height"] = 80
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["width"] = 45
        parameters["fpn"]["fpn_convs"]["0"]["conv"]["batch"] = 6

        parameters["fpn"]["fpn_convs"]["1"] = {}
        parameters["fpn"]["fpn_convs"]["1"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[1].conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs[1].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["height"] = 80
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["width"] = 45
        parameters["fpn"]["fpn_convs"]["1"]["conv"]["batch"] = 6

        parameters["fpn"]["fpn_convs"]["2"] = {}
        parameters["fpn"]["fpn_convs"]["2"]["conv"] = {}
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["weight"] = ttnn.from_torch(
            model.fpn_convs[2].conv.weight, dtype=ttnn.float32
        )
        bias = model.fpn_convs[2].conv.bias.reshape((1, 1, 1, -1))
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32)
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["height"] = 40
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["width"] = 23
        parameters["fpn"]["fpn_convs"]["2"]["conv"]["batch"] = 6

    return parameters


def create_uniad_model_parameters(model, input_tensor, device=None):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {"img_backbone": {}, "img_neck": {}}

    parameters.conv_args["img_backbone"] = infer_ttnn_module_args(
        model=model.img_backbone,
        run_model=lambda model: model(input_tensor),
        device=None,
    )

    for key in parameters.conv_args.keys():
        if key == "img_backbone":
            for conv_key in parameters.conv_args[key].keys():
                parameters.conv_args[key][conv_key].module = getattr(model.img_backbone, conv_key)

    parameters["img_neck"]["model_args"] = model.img_neck

    return parameters
