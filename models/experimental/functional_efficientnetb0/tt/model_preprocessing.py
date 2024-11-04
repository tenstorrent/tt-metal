# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ttnn.model_preprocessing import infer_ttnn_module_args
from models.experimental.functional_efficientnetb0.reference.efficientnetb0 import (
    Efficientnetb0,
    Conv2dDynamicSamePadding,
    MBConvBlock,
)


def create_efficientnetb0_input_tensors(device, batch=1, input_channels=3, input_height=224, input_width=224):
    torch_input_tensor = torch.randn(batch, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(ttnn_input_tensor, dtype=ttnn.bfloat16)
    return torch_input_tensor, ttnn_input_tensor


def create_efficientnetb0_model_parameters(model: Efficientnetb0, input_tensor, device):
    parameters = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    assert parameters is not None
    for key in parameters.keys():
        parameters[key].module = getattr(model, key)
    # print(parameters._blocks0)

    # print(parameters._blocks0._depthwise_conv.batch_size,",",parameters._blocks0._depthwise_conv.in_channels,",",parameters._blocks0._depthwise_conv.out_channels,",",parameters._blocks0._depthwise_conv.input_height
    #       ,",",parameters._blocks0._depthwise_conv.input_width,",",parameters._blocks0._depthwise_conv.kernel_size,",",parameters._blocks0._depthwise_conv.stride,",",parameters._blocks0._depthwise_conv.padding)

    # print(parameters._blocks0._se_reduce.batch_size,",",parameters._blocks0._se_reduce.in_channels,",",parameters._blocks0._se_reduce.out_channels,",",parameters._blocks0._se_reduce.input_height
    #       ,",",parameters._blocks0._se_reduce.input_width,",",parameters._blocks0._se_reduce.kernel_size,",",parameters._blocks0._se_reduce.stride,",",parameters._blocks0._se_reduce.padding)

    # print(parameters._blocks0._se_expand.batch_size,",",parameters._blocks0._se_expand.in_channels,",",parameters._blocks0._se_expand.out_channels,",",parameters._blocks0._se_expand.input_height
    #       ,",",parameters._blocks0._se_expand.input_width,",",parameters._blocks0._se_expand.kernel_size,",",parameters._blocks0._se_expand.stride,",",parameters._blocks0._se_expand.padding)

    # print(parameters._blocks0._project_conv.batch_size,",",parameters._blocks0._project_conv.in_channels,",",parameters._blocks0._project_conv.out_channels,",",parameters._blocks0._project_conv.input_height
    #       ,",",parameters._blocks0._project_conv.input_width,",",parameters._blocks0._project_conv.kernel_size,",",parameters._blocks0._project_conv.stride,",",parameters._blocks0._project_conv.padding)

    # print(parameters._blocks10)
    # ss
    return parameters
