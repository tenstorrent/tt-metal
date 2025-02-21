# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import torch.nn as nn


def fold_bn_to_conv_weights_bias(model, path, device, conv="conv", eps=1e-05, seperable_conv_norm_act=False):
    if seperable_conv_norm_act:
        conv = "conv_pw"
    # bn_weight = model[path + ".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    # bn_running_var = model[path + ".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    # print("Shape of bn weight bn runnn variance :", bn_weight.shape, " ", bn_running_var.shape)
    weight = model[path + f".{conv}.weight"]
    # print("Shape of conv weight :", weight.shape)
    # weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    # print("Shape of conv weight 2:", weight.shape)
    # bn_running_mean = model[path + ".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    # bn_bias = model[path + ".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    # print("Shape of bn run mean bn bias :", bn_running_mean.shape, " ", bn_bias.shape)

    # bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias

    # bias = bias.reshape(1, 1, 1, -1)

    bias = None
    return (
        ttnn.from_torch(weight, dtype=ttnn.bfloat16),
        # ttnn.from_torch(bias, dtype=ttnn.bfloat16),
        bias,
    )


"""
def fold_batch_norm2d_into_conv2d(model, path, eps=1e-05, conv="conv", seperable_conv_norm_act=False):
    if seperable_conv_norm_act:
        conv = "conv_pw"
    weight = model[path + f".{conv}.weight"]

    bias = None
    running_mean = model[path + ".bn.running_mean"]

    running_var = model[path + ".bn.running_var"]

    eps = eps
    scale = model[path + ".bn.weight"]

    shift = model[path + ".bn.bias"]

    weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
    if bias is not None:
        bias = (bias - running_mean) * (scale / torch.sqrt(running_var + eps)) + shift
    else:
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))

    # return weight, bias
    bias = bias.reshape(1, 1, 1, -1)
    print("Shape of weight and bias:", weight.shape, " ", bias.shape)
    return (
        ttnn.from_torch(weight, dtype=ttnn.bfloat16),
        ttnn.from_torch(bias, dtype=ttnn.bfloat16),
    )
"""


class Conv:
    def __init__(
        self,
        device,
        # model,
        path,
        conv_params,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        activation="",
        fused_op=True,
        debug=False,
        groups=1,
        bias=False,
        split_conv=False,
        seperable_conv_norm_act=False,
        effective_se=False,
        parameters=None,
        pw=False,
    ) -> None:
        self.fused_op = fused_op
        path = path
        if fused_op:
            # self.weights, self.bias = fold_bn_to_conv_weights_bias(
            #     model, path, device, seperable_conv_norm_act=seperable_conv_norm_act
            # )
            # self.weights, self.bias = fold_batch_norm2d_into_conv2d(model, path, seperable_conv_norm_act = seperable_conv_norm_act)
            if pw:
                self.weights = parameters[f"{path}.conv_pw.weight"]
                self.bias = parameters[f"{path}.conv_pw.bias"]
            else:
                self.weights = parameters[f"{path}.conv.weight"]
                self.bias = parameters[f"{path}.conv.bias"]
            self.groups = groups
        else:
            if effective_se:
                # weight = model[path + ".fc.weight"]
                # bias = model[path + ".fc.bias"]
                # bias = bias.reshape(1, 1, 1, -1)
                # self.bias = ttnn.from_torch(bias, dtype=ttnn.bfloat16)
                self.weights = parameters[f"{path}.weight"]
                self.bias = parameters[f"{path}.bias"]
                self.groups = groups
            else:
                self.weights = parameters[f"{path}.conv_dw.weight"]
                self.bias = parameters[f"{path}.conv_dw.bias"]
                self.groups = self.weights.shape[0]
                # self.bias = None

            # self.weights = ttnn.from_torch(weight, dtype=ttnn.bfloat16)

        self.split_conv = split_conv
        self.conv_params = conv_params

        self.debug = debug

        self.device = device
        self.act_block_h = act_block_h

        if fused_op:
            activation = ""
        if not self.split_conv:
            self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
            self.out_channels = self.weights.shape[0]
            self.reader_patterns_cache = {}
            self.conv_config = ttnn.Conv2dConfig(
                dtype=ttnn.bfloat16,
                weights_dtype=ttnn.bfloat16,
                activation=activation,
                shard_layout=None,
                input_channels_alignment=8,  # if self.input_params[-1] < 16 else 32,
                reshard_if_not_optimal=reshard,
                deallocate_activation=deallocate,
            )
            if self.act_block_h is not None:
                self.conv_config.act_block_h_override = act_block_h

        if self.fused_op:
            self.bn = nn.BatchNorm2d(num_features=self.weights.shape[0])

            self.bn.weight.data = ttnn.to_torch(parameters[f"{path}.bn.weight"])  # Example values
            self.bn.bias.data = ttnn.to_torch(parameters[f"{path}.bn.bias"])
            # Assign new values to running mean and variance (used during inference)
            self.bn.running_mean = ttnn.to_torch(parameters[f"{path}.bn.running_mean"])
            self.bn.running_var = ttnn.to_torch(parameters[f"{path}.bn.running_var"])

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, input_tensor):
        N, C, H, W = input_tensor.shape
        input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if not self.split_conv:
            self.conv_config.input_channels_alignment = 8 if input_tensor.shape[-1] < 16 else 32
            output_tensor, [_out_height, _out_width], [self.weights, self.bias] = ttnn.conv2d(
                input_tensor=input_tensor,
                weight_tensor=self.weights,
                in_channels=input_tensor.shape[-1],
                out_channels=self.out_channels,
                device=self.device,
                bias_tensor=self.bias if self.bias else None,
                kernel_size=self.kernel_size,
                stride=(self.conv_params[0], self.conv_params[1]),
                padding=(self.conv_params[2], self.conv_params[-1]),
                batch_size=input_tensor.shape[0],
                input_height=input_tensor.shape[1],
                input_width=input_tensor.shape[2],
                conv_config=self.conv_config,
                conv_op_cache=self.reader_patterns_cache,
                debug=self.debug,
                groups=self.groups,
                compute_config=compute_config,
                return_output_dim=True,
                return_weights_and_bias=True,
            )

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        print(
            "Shape of output :",
            output_tensor.shape,
            " ",
            output_tensor.shape[0],
            " ",
            _out_height,
            " ",
            _out_width,
            " ",
            output_tensor.shape[-1],
        )
        output_tensor = ttnn.reshape(output_tensor, (N, _out_height, _out_width, output_tensor.shape[-1]))
        output_tensor = ttnn.permute(output_tensor, (0, 3, 1, 2))
        if self.fused_op:
            output_tensor = ttnn.to_torch(output_tensor)
            output_tensor = self.bn(output_tensor)
            output_tensor = ttnn.from_torch(
                output_tensor, device=self.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )
            output_tensor = ttnn.relu(output_tensor)
        return output_tensor, _out_height, _out_width
