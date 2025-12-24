# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_cnn.tt.builder import Conv2dConfiguration, TtConv2d, AutoShardedStrategyConfiguration


class TtnnSharedMLP(LightweightModule):
    def __init__(self, parameters, device):
        super().__init__()
        self.device = device
        self.parameters = parameters
        # Create conv layers using TtConv2d
        self.conv1 = TtConv2d(
            self._create_conv_config(
                parameters=parameters.layer0.conv,
                layer_params=parameters.conv_args.layer0.conv,
            ),
            device,
        )
        self.conv2 = TtConv2d(
            self._create_conv_config(
                layer_params=parameters.conv_args.layer1.conv,
                parameters=parameters.layer1.conv,
            ),
            device,
        )
        self.conv3 = TtConv2d(
            self._create_conv_config(
                layer_params=parameters.conv_args.layer2.conv,
                parameters=parameters.layer2.conv,
            ),
            device,
        )

    def _create_conv_config(self, parameters, layer_params):
        # Move weights from device to host for proper conv2d preparation
        weight = parameters.weight
        if isinstance(weight, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(weight):
            weight = ttnn.from_device(weight)

        bias = None
        if hasattr(parameters, "bias") and parameters.bias is not None:
            bias = parameters.bias
            if isinstance(bias, ttnn.Tensor) and ttnn.is_tensor_storage_on_device(bias):
                bias = ttnn.from_device(bias)

        return Conv2dConfiguration(
            input_height=2048,
            input_width=64,
            in_channels=layer_params.in_channels,
            out_channels=layer_params.out_channels,
            batch_size=1,
            kernel_size=layer_params.kernel_size,
            stride=layer_params.stride,
            padding=layer_params.padding,
            weight=weight,
            bias=bias,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            activation_dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat16,
            output_dtype=ttnn.bfloat16,
            sharding_strategy=AutoShardedStrategyConfiguration(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            deallocate_activation=True,
            enable_act_double_buffer=False,
        )
        # Conv2dConfiguration.from_model_args(layer_params, weights, bias, *)

    def forward(self, features):
        conv1 = self.conv1(features)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        return conv3
