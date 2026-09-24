# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)
from models.experimental.bevformer.reference.resnet import ResNet, ModulatedDeformConv2dPack


def custom_preprocessor(model, name):
    parameters = {}

    if isinstance(model, ResNet):
        parameters["res_model"] = {}

        weight, bias = fold_batch_norm2d_into_conv2d(model.conv1, model.bn1)
        parameters["res_model"]["conv1"] = {
            "weight": ttnn.from_torch(weight, dtype=ttnn.float32),
            "bias": ttnn.from_torch(bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
        }

        for layer_idx in range(1, 5):
            layer = getattr(model, f"layer{layer_idx}")
            prefix = f"layer{layer_idx}"
            parameters["res_model"][prefix] = {}
            for block_idx, block in enumerate(layer):
                parameters["res_model"][prefix][block_idx] = {}

                for conv_name in ["conv1", "conv2", "conv3"]:
                    conv = getattr(block, conv_name)
                    if isinstance(conv, ModulatedDeformConv2dPack):
                        # The DCN conv consumes torch weights, and its BatchNorm runs as a
                        # separate op after it, so neither is folded here.
                        parameters["res_model"][prefix][block_idx][conv_name] = {}
                        parameters["res_model"][prefix][block_idx][conv_name]["weight"] = conv.weight
                        parameters["res_model"][prefix][block_idx][conv_name]["bias"] = conv.bias
                        parameters["res_model"][prefix][block_idx][conv_name]["conv_offset"] = {
                            "weight": ttnn.from_torch(conv.conv_offset.weight, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(conv.conv_offset.bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                        bn = getattr(block, f"bn{conv_name[-1]}")
                        channel_size = bn.num_features

                        weight_torch = bn.weight if bn.affine else None
                        bias_torch = bn.bias if bn.affine else None
                        batch_mean_torch = bn.running_mean.view(1, channel_size, 1, 1)
                        batch_var_torch = bn.running_var.view(1, channel_size, 1, 1)
                        weight_torch = weight_torch.view(1, channel_size, 1, 1) if weight_torch is not None else None
                        bias_torch = bias_torch.view(1, channel_size, 1, 1) if bias_torch is not None else None

                        bn_params = {}
                        bn_params["weight"] = (
                            ttnn.from_torch(weight_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if weight_torch is not None
                            else None
                        )
                        bn_params["bias"] = (
                            ttnn.from_torch(bias_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
                            if bias_torch is not None
                            else None
                        )
                        bn_params["running_mean"] = ttnn.from_torch(
                            batch_mean_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                        )
                        bn_params["running_var"] = ttnn.from_torch(
                            batch_var_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
                        )
                        bn_params["eps"] = bn.eps
                        parameters["res_model"][prefix][block_idx]["bn2"] = bn_params
                    else:
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    if isinstance(ds, torch.nn.Sequential):
                        w, b = fold_batch_norm2d_into_conv2d(ds[0], ds[1])
                        parameters["res_model"][prefix][block_idx]["downsample"] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }
    return parameters


def create_resnet_parameters(model: ResNet, input_tensor, device=None):
    """Preprocess backbone weights and record per-conv shapes from one reference forward.

    ``conv_args`` pins every conv's input height, width and batch to ``input_tensor``'s
    shape, so the TT backbone built from these parameters only accepts that shape.
    """
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    for key in parameters.conv_args.keys():
        parameters.conv_args[key].module = getattr(model, key)
    return parameters
