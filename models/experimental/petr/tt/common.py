# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from typing import Optional, Dict, Tuple

from models.tt_cnn.tt.builder import (
    TtConv2d,
    Conv2dConfiguration,
    AutoShardedStrategyConfiguration,
)


def create_conv_config_from_parameters(
    input_height: int,
    input_width: int,
    in_channels: int,
    out_channels: int,
    batch_size: int,
    parameters: Dict,
    conv_params: Tuple[int, int, int, int],
    kernel_size: Optional[Tuple[int, int]] = None,
    activation: Optional[ttnn.UnaryWithParam] = None,
    deallocate_activation: bool = True,
    height_sharding: bool = True,
    width_sharding: Optional[bool] = None,
    groups: int = 1,
    dilation: int = 1,
    activation_dtype: ttnn.DataType = ttnn.bfloat16,
    weights_dtype: ttnn.DataType = ttnn.bfloat16,
    output_dtype: ttnn.DataType = ttnn.bfloat16,
    math_fidelity: ttnn.MathFidelity = ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en: bool = True,
    packer_l1_acc: bool = True,
    enable_weights_double_buffer: bool = True,
    enable_act_double_buffer: bool = False,
    reshard_if_not_optimal: bool = True,
) -> Conv2dConfiguration:
    if kernel_size is None:
        weight = parameters["weight"]
        kernel_size = (weight.shape[2], weight.shape[3])

    sharding_strategy = AutoShardedStrategyConfiguration()

    if isinstance(activation, str):
        if activation == "relu":
            activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        elif activation == "gelu":
            activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)
        else:
            activation = None

    return Conv2dConfiguration(
        input_height=input_height,
        input_width=input_width,
        in_channels=in_channels,
        out_channels=out_channels,
        batch_size=batch_size,
        kernel_size=kernel_size,
        stride=(conv_params[0], conv_params[1]),
        padding=(conv_params[2], conv_params[3]),
        dilation=(dilation, dilation),
        groups=groups,
        weight=parameters["weight"],
        bias=parameters.get("bias"),
        activation=activation,
        deallocate_activation=deallocate_activation,
        sharding_strategy=sharding_strategy,
        activation_dtype=activation_dtype,
        output_dtype=output_dtype,
        weights_dtype=weights_dtype,
        math_fidelity=math_fidelity,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=packer_l1_acc,
        enable_weights_double_buffer=enable_weights_double_buffer,
        enable_act_double_buffer=enable_act_double_buffer,
        config_tensors_in_dram=True,
        slice_strategy=None,
        output_layout=ttnn.TILE_LAYOUT,
    )


class Conv:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        width_sharding=None,
        activation="",
        dtype=ttnn.bfloat16,
        groups=1,
        dilation=1,
        use_shallow_conv_variant=False,
        device=None,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters.get("bias")
        self.conv_params = conv_params

        if isinstance(self.weights, torch.Tensor):
            self.out_channels = self.weights.shape[0]
        else:
            self.out_channels = self.weights.shape[0]
        self.activation = activation
        self.dtype = dtype
        self.groups = groups
        self.dilation = dilation
        self.deallocate = deallocate
        self.height_sharding = height_sharding
        self.width_sharding = width_sharding
        self.device = device

    def __call__(self, device, input_tensor):
        batch_size = input_tensor.shape[0]
        input_height = input_tensor.shape[1]
        input_width = input_tensor.shape[2]
        input_channels = input_tensor.shape[3]

        input_tensor = ttnn.to_memory_config(input_tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if hasattr(self.weights, "memory_config") and self.weights.memory_config().is_sharded():
            self.weights = ttnn.sharded_to_interleaved(self.weights, ttnn.DRAM_MEMORY_CONFIG)

        activation_param = None
        if self.activation == "relu":
            activation_param = ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)
        elif self.activation == "gelu":
            activation_param = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU)

        conv_config = create_conv_config_from_parameters(
            input_height=input_height,
            input_width=input_width,
            in_channels=input_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            parameters={"weight": self.weights, "bias": self.bias},
            conv_params=self.conv_params,
            activation=activation_param,
            deallocate_activation=self.deallocate,
            height_sharding=False,
            width_sharding=False,
            groups=self.groups,
            dilation=self.dilation,
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            reshard_if_not_optimal=True,
        )

        tt_conv2d = TtConv2d(conv_config, device)
        output_tensor, (out_height, out_width) = tt_conv2d(input_tensor, return_output_dim=True)

        if hasattr(output_tensor, "memory_config") and output_tensor.memory_config().is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.DRAM_MEMORY_CONFIG)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (batch_size, out_height, out_width, output_tensor.shape[3]))
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        return output_tensor


class Conv_with_split:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=False,
        height_sharding=True,
        width_sharding=None,
        activation="",
        dtype=ttnn.bfloat16,
        groups=1,
        dilation=1,
        use_shallow_conv_variant=False,
        split_factor=2,
        device=None,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters.get("bias")

        if isinstance(self.weights, torch.Tensor):
            input_channels = self.weights.shape[1]
            self.output_channels = self.weights.shape[0]
            self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        else:
            input_channels = self.weights.shape[1]
            self.output_channels = self.weights.shape[0]
            self.kernel_size = (self.weights.shape[2], self.weights.shape[3])

        assert input_channels % split_factor == 0
        self.split_input_channels = input_channels // split_factor
        self.dtype = dtype
        self.split_factor = split_factor
        self.act_block_h = act_block_h
        self.conv_params = conv_params
        self.activation = activation
        self.deallocate = deallocate
        self.height_sharding = height_sharding
        self.width_sharding = width_sharding
        self.groups = groups
        self.dilation = dilation
        self.device = device

    def __call__(self, device, input_tensor):
        batch, height, width, channel = input_tensor.shape

        if not isinstance(input_tensor, torch.Tensor):
            input_tensor_torch = ttnn.to_torch(input_tensor)
        else:
            input_tensor_torch = input_tensor

        split_input_tensors = torch.split(input_tensor_torch, self.split_input_channels, 3)
        output_tensors = []
        out_height = None
        out_width = None

        for i in range(self.split_factor):
            split_input_ttnn = ttnn.from_torch(
                split_input_tensors[i], dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG, device=device
            )

            if isinstance(self.weights, torch.Tensor):
                weights_torch = self.weights
            else:
                weights_torch = ttnn.to_torch(self.weights)

            split_weight_tensors = torch.split(weights_torch, self.split_input_channels, 1)
            split_weight = split_weight_tensors[i]
            split_weight_ttnn = ttnn.from_torch(split_weight, dtype=ttnn.bfloat16)

            conv_config = create_conv_config_from_parameters(
                input_height=height,
                input_width=width,
                in_channels=self.split_input_channels,
                out_channels=self.output_channels,
                batch_size=batch,
                parameters={"weight": split_weight_ttnn, "bias": None},
                conv_params=self.conv_params,
                activation=None,
                deallocate_activation=self.deallocate,
                height_sharding=self.height_sharding,
                width_sharding=self.width_sharding,
                groups=1,
                dilation=1,
                math_fidelity=ttnn.MathFidelity.HiFi4,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
                enable_act_double_buffer=False,
            )

            tt_conv2d = TtConv2d(conv_config, device)
            output_tensor, (out_h, out_w) = tt_conv2d(split_input_ttnn, return_output_dim=True)
            out_height = out_h
            out_width = out_w
            output_tensors.append(output_tensor)

        accumulated_output = output_tensors[0]
        for i in range(1, len(output_tensors)):
            accumulated_output = ttnn.add(accumulated_output, output_tensors[i], output_tensor=accumulated_output)
            output_tensors[i].deallocate(True)

        if self.bias is not None:
            if isinstance(self.bias, torch.Tensor):
                bias_torch = self.bias
            else:
                bias_torch = ttnn.to_torch(self.bias)
            bias_reshaped = bias_torch.view(1, 1, 1, -1)
            bias_ttnn = ttnn.from_torch(bias_reshaped, dtype=ttnn.bfloat16, device=device)
            accumulated_output = ttnn.add(accumulated_output, bias_ttnn, output_tensor=accumulated_output)

        if hasattr(accumulated_output, "memory_config") and accumulated_output.memory_config().is_sharded():
            accumulated_output = ttnn.sharded_to_interleaved(accumulated_output, ttnn.L1_MEMORY_CONFIG)

        accumulated_output = ttnn.to_layout(accumulated_output, layout=ttnn.ROW_MAJOR_LAYOUT)
        if accumulated_output.shape[1] != out_height or accumulated_output.shape[2] != out_width:
            accumulated_output = ttnn.reshape(accumulated_output, (batch, out_height, out_width, self.output_channels))

        accumulated_output = ttnn.to_layout(accumulated_output, layout=ttnn.TILE_LAYOUT)

        if self.activation == "relu":
            accumulated_output = ttnn.relu(accumulated_output)

        return accumulated_output, [out_height, out_width]
