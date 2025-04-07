# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias


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


def create_mobilenetv2_model_parameters(model, device):
    model_parameters = {}

    for i in range(1, 53):
        model_parameters[i] = fold_batch_norm2d_into_conv2d(model.__getattr__(f"c{i}"), model.__getattr__(f"b{i}"))

    model_parameters["l1"] = {}
    model_parameters["l1"]["weight"] = model.l1.weight
    model_parameters["l1"]["bias"] = model.l1.bias

    model_parameters["l1"]["weight"] = preprocess_linear_weight(model_parameters["l1"]["weight"], dtype=ttnn.bfloat16)
    model_parameters["l1"]["bias"] = preprocess_linear_bias(model_parameters["l1"]["bias"], dtype=ttnn.bfloat16)

    model_parameters["l1"]["weight"] = ttnn.to_device(model_parameters["l1"]["weight"], device)
    model_parameters["l1"]["bias"] = ttnn.to_device(model_parameters["l1"]["bias"], device)

    return model_parameters
