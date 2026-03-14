# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
)
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    L1FullSliceStrategyConfiguration,
    AutoShardedStrategyConfiguration,
)
from models.tt_cnn.tt.builder import TtConv2d, TtMaxPool2d
import torch.nn as nn
from dataclasses import replace


CONV_CONFIG_DEFAULT = {
    "weights_dtype": ttnn.bfloat8_b,
    "output_dtype": ttnn.bfloat8_b,
    "activation_dtype": ttnn.bfloat8_b,
    "math_fidelity": ttnn.MathFidelity.LoFi,
    "sharding_strategy": AutoShardedStrategyConfiguration(),
    "slice_strategy": L1FullSliceStrategyConfiguration(),
}

POOL_CONFIG_DEFAULT = {
    "dtype": ttnn.bfloat16,
    "slice_strategy": L1FullSliceStrategyConfiguration(),
}


# Converts PyTorch model layers to TTNN configuration objects
def create_config_layers(
    torch_model,
    torch_input,
    conv_config=CONV_CONFIG_DEFAULT,
    pool_config=POOL_CONFIG_DEFAULT,
    return_output_tensor=False,
):
    layer_configs = []
    # Process each layer and create corresponding TTNN configuration
    for layer in torch_model:
        if isinstance(layer, nn.Conv2d):
            layer_configs.append(
                Conv2dConfiguration.from_torch(
                    layer,
                    input_height=torch_input.shape[-2],
                    input_width=torch_input.shape[-1],
                    batch_size=torch_input.shape[0],
                    **conv_config,
                )
            )
        elif isinstance(layer, nn.MaxPool2d):
            layer_configs.append(
                MaxPool2dConfiguration.from_torch(
                    layer,
                    input_height=torch_input.shape[-2],
                    input_width=torch_input.shape[-1],
                    channels=torch_input.shape[-3],
                    batch_size=torch_input.shape[0],
                    **pool_config,
                )
            )
        torch_input = layer(torch_input)
    if return_output_tensor:
        return layer_configs, torch_input
    return layer_configs


# Wrapper for Conv2d layer
class Conv2dOperation:
    def __init__(
        self,
        device=None,
        conv_config=None,
        activation_layer=None,
    ):
        self.conv_config = conv_config
        self.activation_layer = activation_layer

        self.conv = TtConv2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        [input_tensor, [_out_height, _out_width]] = self.conv(input_tensor, return_output_dim=True)
        if self.activation_layer is not None:
            input_tensor = self.activation_layer(input_tensor)
        return input_tensor


# Wrapper for MaxPool2d layer
class Maxpool2DOperation:
    def __init__(
        self,
        device=None,
        conv_config=None,
    ):
        self.conv_config = conv_config

        self.pool = TtMaxPool2d(self.conv_config, device)

    def __call__(self, device, input_tensor, return_output_dim=True):
        input_tensor = self.pool(input_tensor)

        return input_tensor


# Overrides Conv2dConfiguration parameters
def override_conv_config(config, override_dict):
    if not isinstance(config, Conv2dConfiguration):
        return config
    return replace(config, **override_dict)


# Reshapes tensor from sharded to interleaved layout and applies spatial reshape
def post_conv_reshape(x, out_height=1, out_width=1):
    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (x.shape[0], out_height, out_width, x.shape[3]))
    return ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
