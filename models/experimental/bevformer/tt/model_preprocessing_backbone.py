# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import ttnn

from ttnn.model_preprocessing import (
    infer_ttnn_module_args,
    preprocess_model_parameters,
    fold_batch_norm2d_into_conv2d,
)
from models.experimental.bevformer.reference.fpn import FPN
from models.experimental.bevformer.reference.resnet import ResNet, ModulatedDeformConv2dPack
from models.experimental.bevformer.tt.tt_modulated_deform_conv import grid_offset_order


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
                        # The offset rows are reordered to the (x, y) order the device DCN takes;
                        # moving whole rows is exact.
                        order = grid_offset_order(conv.kernel_size[0] * conv.kernel_size[1])
                        offset_weight = conv.conv_offset.weight.detach()
                        offset_bias = conv.conv_offset.bias.detach()
                        offset_weight = torch.cat([offset_weight[order], offset_weight[len(order) :]])
                        offset_bias = torch.cat([offset_bias[order], offset_bias[len(order) :]])
                        parameters["res_model"][prefix][block_idx][conv_name]["conv_offset"] = {
                            "weight": ttnn.from_torch(offset_weight, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(offset_bias.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
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
                        parameters["res_model"][prefix][block_idx][f"bn{conv_name[-1]}"] = bn_params
                    else:
                        bn = getattr(block, f"bn{conv_name[-1]}")
                        w, b = fold_batch_norm2d_into_conv2d(conv, bn)
                        parameters["res_model"][prefix][block_idx][conv_name] = {
                            "weight": ttnn.from_torch(w, dtype=ttnn.float32),
                            "bias": ttnn.from_torch(b.reshape((1, 1, 1, -1)), dtype=ttnn.float32),
                        }

                if hasattr(block, "downsample") and block.downsample is not None:
                    ds = block.downsample
                    assert (
                        isinstance(ds, torch.nn.Sequential) and len(ds) == 2
                    ), f"expected a (conv, norm) downsample, got {ds}"
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
    return parameters


def create_fpn_parameters(model: FPN, input_tensors):
    """Preprocess FPN weights and record every conv's geometry and input shape.

    ``input_tensors`` are the NCHW backbone outputs the FPN will run on. A conv reads the
    level it belongs to, except the extra output convs: the first reads the last level's
    output and each later one reads the previous extra conv's output.

    ``conv_args`` holds, per conv, the attributes ``TtnnConv2D`` reads, as
    ``create_resnet_parameters`` records them for the backbone.
    """
    level_shapes = [(t.shape[0], t.shape[2], t.shape[3]) for t in input_tensors]
    shapes = list(level_shapes)
    batch, height, width = level_shapes[-1]
    for fpn_conv in model.fpn_convs[len(level_shapes) :]:
        shapes.append((batch, height, width))
        conv = fpn_conv.conv
        height = (height + 2 * conv.padding[0] - conv.kernel_size[0]) // conv.stride[0] + 1
        width = (width + 2 * conv.padding[1] - conv.kernel_size[1]) // conv.stride[1] + 1

    def conv_args(conv_module, shape):
        conv = conv_module.conv
        batch, height, width = shape
        return SimpleNamespace(
            conv=SimpleNamespace(
                in_channels=conv.in_channels,
                out_channels=conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                batch_size=batch,
                input_height=height,
                input_width=width,
            )
        )

    def conv_weights(conv_module):
        conv = conv_module.conv
        return {
            "conv": {
                "weight": ttnn.from_torch(conv.weight, dtype=ttnn.bfloat16),
                "bias": ttnn.from_torch(conv.bias.reshape((1, 1, 1, -1)), dtype=ttnn.bfloat16),
            }
        }

    def preprocessor(module, name):
        if not isinstance(module, FPN):
            return {}
        return {
            "fpn": {
                "lateral_convs": {str(i): conv_weights(lateral) for i, lateral in enumerate(module.lateral_convs)},
                "fpn_convs": {str(i): conv_weights(fpn_conv) for i, fpn_conv in enumerate(module.fpn_convs)},
            }
        }

    parameters = preprocess_model_parameters(initialize_model=lambda: model, custom_preprocessor=preprocessor)
    parameters["conv_args"] = SimpleNamespace(
        lateral_convs=[conv_args(lateral, level_shapes[i]) for i, lateral in enumerate(model.lateral_convs)],
        fpn_convs=[conv_args(fpn_conv, shapes[i]) for i, fpn_conv in enumerate(model.fpn_convs)],
        add_extra_convs=model.add_extra_convs,
        relu_before_extra_convs=model.relu_before_extra_convs,
    )
    return parameters
