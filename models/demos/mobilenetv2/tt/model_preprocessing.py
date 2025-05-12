# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.demos.mobilenetv2.reference.mobilenetv2 import (  # Import Conv2dNormActivation
    Conv2dNormActivation,
    InvertedResidual,
    Mobilenetv2,
)


def create_mobilenetv2_input_tensors(batch=1, input_channels=3, input_height=224, input_width=224):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)

    return torch_input_tensor, ttnn_input_tensor


def fold_batch_norm2d_into_conv2d(conv, bn):
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
    weight = ttnn.from_torch(weight, dtype=ttnn.float32)
    bias = ttnn.from_torch(bias, dtype=ttnn.float32)
    return weight, bias


def create_mobilenetv2_model_parameters(model, device):
    model_parameters = {}
    conv_bn_counter = 0
    counter = 0

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
                        weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv2d(submodule, bn)
                        model_parameters[f"conv_{counter}_weight"] = weight_ttnn
                        model_parameters[f"conv_{counter}_bias"] = bias_ttnn
                        counter += 1

        elif isinstance(module, Conv2dNormActivation):
            if len(module) == 3 and isinstance(module[0], nn.Conv2d) and isinstance(module[1], nn.BatchNorm2d):
                conv = module[0]
                bn = module[1]
                weight_ttnn, bias_ttnn = fold_batch_norm2d_into_conv2d(conv, bn)
                model_parameters[f"fused_conv_{conv_bn_counter}_weight"] = weight_ttnn
                model_parameters[f"fused_conv_{conv_bn_counter}_bias"] = bias_ttnn
                conv_bn_counter += 1

        elif isinstance(module, nn.Linear):
            model_parameters["classifier_1_weight"] = preprocess_linear_weight(module.weight.data, dtype=ttnn.float32)
            model_parameters["classifier_1_bias"] = preprocess_linear_bias(module.bias.data, dtype=ttnn.float32)
            model_parameters["classifier_1_weight"] = ttnn.to_device(model_parameters["classifier_1_weight"], device)
            model_parameters["classifier_1_bias"] = ttnn.to_device(model_parameters["classifier_1_bias"], device)

    return model_parameters
