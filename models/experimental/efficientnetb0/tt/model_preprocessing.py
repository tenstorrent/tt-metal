# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import infer_ttnn_module_args
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias


def create_efficientnetb0_input_tensors(device, batch=1, input_channels=3, input_height=224, input_width=224):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16, device=device)
    return torch_input_tensor, ttnn_input_tensor


def fold_batch_norm2d_into_conv2d(conv, bn):
    if not bn.track_running_stats:
        raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")
    weight = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    eps = bn.eps
    scale = bn.weight
    shift = bn.bias
    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
    bias = torch.reshape(bias, (1, 1, 1, -1))
    weight = ttnn.from_torch(weight, dtype=ttnn.float32)
    bias = ttnn.from_torch(bias, dtype=ttnn.float32)
    return weight, bias


def preprocess_conv_params(conv):
    weight = conv.weight
    bias = conv.bias
    bias = torch.reshape(bias, (1, 1, 1, -1))
    weight = ttnn.from_torch(weight, dtype=ttnn.float32)
    bias = ttnn.from_torch(bias, dtype=ttnn.float32)

    return weight, bias


def create_efficientnetb0_model_parameters(model, input_tensor, device):
    conv_params = {}
    conv_params = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    parameters = {}
    parameters["_conv_stem"] = fold_batch_norm2d_into_conv2d(model.__getattr__("_conv_stem"), model.__getattr__("_bn0"))
    blocks_params = {}
    for i in range(0, 16):
        block_parameters = {}
        block = model.__getattr__(f"_blocks{i}")

        if i != 0:
            block_parameters["_expand_conv"] = fold_batch_norm2d_into_conv2d(block._expand_conv, block._bn0)

        block_parameters["_depthwise_conv"] = fold_batch_norm2d_into_conv2d(block._depthwise_conv, block._bn1)

        block_parameters["_se_reduce"] = preprocess_conv_params(block._se_reduce)

        block_parameters["_se_expand"] = preprocess_conv_params(block._se_expand)

        block_parameters["_project_conv"] = fold_batch_norm2d_into_conv2d(block._project_conv, block._bn2)

        blocks_params[f"_blocks{i}"] = block_parameters

    parameters["blocks"] = blocks_params

    parameters["_conv_head"] = fold_batch_norm2d_into_conv2d(model.__getattr__("_conv_head"), model.__getattr__("_bn1"))

    parameters["l1"] = {}
    parameters["l1"]["weight"] = model._fc.weight
    parameters["l1"]["bias"] = model._fc.bias

    parameters["l1"]["weight"] = preprocess_linear_weight(parameters["l1"]["weight"], dtype=ttnn.bfloat16)
    parameters["l1"]["bias"] = preprocess_linear_bias(parameters["l1"]["bias"], dtype=ttnn.bfloat16)

    parameters["l1"]["weight"] = ttnn.to_device(parameters["l1"]["weight"], device)
    parameters["l1"]["bias"] = ttnn.to_device(parameters["l1"]["bias"], device)

    return conv_params, parameters
