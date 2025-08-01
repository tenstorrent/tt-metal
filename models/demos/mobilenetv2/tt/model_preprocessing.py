# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

import ttnn
from models.demos.mobilenetv2.reference.mobilenetv2 import (  # Import Conv2dNormActivation
    Conv2dNormActivation,
    InvertedResidual,
    Mobilenetv2,
)


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


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return bias


def create_mobilenetv2_input_tensors(
    batch=1, input_channels=3, input_height=224, input_width=224, pad_channels=None, mesh_mapper=None
):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    if pad_channels:
        ttnn_input_tensor = torch.nn.functional.pad(
            ttnn_input_tensor, (0, pad_channels - ttnn_input_tensor.shape[-1]), value=0
        )
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper
    )
    ttnn_input_tensor = ttnn.reshape(
        ttnn_input_tensor,
        (
            1,
            1,
            ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
            ttnn_input_tensor.shape[3],
        ),
    )
    return torch_input_tensor, ttnn_input_tensor


def fold_batch_norm2d_into_conv2d(conv, bn, mesh_mapper=None):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")
    weight = conv.weight.data
    running_mean = bn.running_mean
    running_var = bn.running_var.data
    eps = bn.eps
    scale = bn.weight.data
    shift = bn.bias.data
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
    bias = torch.reshape(bias, (1, 1, 1, -1))
    weight = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    bias = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    return weight, bias


def create_mobilenetv2_model_parameters(model, device):
    model_parameters = {}
    conv_bn_counter = 0
    counter = 0
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    for name, module in model.named_modules():
        if isinstance(module, InvertedResidual):
            for idx, submodule in enumerate(module.conv):
                if isinstance(submodule, nn.Conv2d):
                    bn = (
                        module.conv[idx + 1]
                        if idx + 1 < len(module.conv) and isinstance(module.conv[idx + 1], nn.BatchNorm2d)
                        else None
                    )
                    if bn:
                        weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv2d(
                            submodule, bn, mesh_mapper=weights_mesh_mapper
                        )
                        model_parameters[f"conv_{counter}_weight"] = weight_ttnn
                        model_parameters[f"conv_{counter}_bias"] = bias_ttnn
                        counter += 1

        elif isinstance(module, Conv2dNormActivation):
            if len(module) == 3 and isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d):
                conv = module[0]
                bn = module[1]
                weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv2d(conv, bn, mesh_mapper=weights_mesh_mapper)
                model_parameters[f"fused_conv_{conv_bn_counter}_weight"] = weight_ttnn
                model_parameters[f"fused_conv_{conv_bn_counter}_bias"] = bias_ttnn
                conv_bn_counter += 1

        elif isinstance(module, nn.Linear):
            model_parameters["classifier_1_weight"] = preprocess_linear_weight(
                module.weight.data, dtype=ttnn.float32, mesh_mapper=weights_mesh_mapper
            )
            model_parameters["classifier_1_bias"] = preprocess_linear_bias(
                module.bias.data, dtype=ttnn.float32, mesh_mapper=weights_mesh_mapper
            )
            model_parameters["classifier_1_weight"] = ttnn.to_device(model_parameters["classifier_1_weight"], device)
            model_parameters["classifier_1_bias"] = ttnn.to_device(model_parameters["classifier_1_bias"], device)

    return model_parameters
