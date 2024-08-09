# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def fold_bn_to_conv_weights_bias(conv_weights, bn_bias, bn_weights, bn_running_var, bn_running_mean):
    bn_weight = bn_weights.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = bn_running_var.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = conv_weights
    weight = (weight / torch.sqrt(bn_running_var)) * bn_weight

    bn_running_mean = bn_running_mean.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = bn_bias.unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var)) + bn_bias

    bias = bias.reshape(1, 1, 1, -1)
    return (
        weight,
        bias,
    )


class Conv:
    def __init__(
        self,
        input_params,
        conv_params,
        conv_weights,
        bias,
        *,
        bn_weights=None,
        bn_running_var=None,
        bn_running_mean=None,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="relu",
        fused_op=True,
        mesh_mapper=None,
        use_shallow_conv_variant=False,
    ) -> None:
        if fused_op:
            self.weights, self.bias = fold_bn_to_conv_weights_bias(
                conv_weights,
                bias,
                bn_weights,
                bn_running_var,
                bn_running_mean,
            )
            self.weights = ttnn.from_torch(self.weights, mesh_mapper=mesh_mapper)
            self.bias = ttnn.from_torch(self.bias, mesh_mapper=mesh_mapper)
        else:
            weight = conv_weights
            bias = bias
            self.weights = ttnn.from_torch(weight)
            bias = bias.reshape(1, 1, 1, -1)
            self.bias = ttnn.from_torch(bias)
        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.height_sharding = height_sharding
        self.deallocate = deallocate
        self.activation = activation
        self.use_shallow_conv_variant = use_shallow_conv_variant

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            math_fidelity=ttnn.MathFidelity.LoFi,
            activation=self.activation,
            height_sharding=self.height_sharding,
            math_approx_mode_enabled=True,
            fp32_dest_acc_enabled=False,
            packer_l1_accum_enabled=False,
            input_channels_alignment=(
                16
                if self.use_shallow_conv_variant or (self.input_params[3] == 16 and self.input_params[3] == 115)
                else 32
            ),
            transpose_shards=False,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h
        print
        [output_tensor, _out_height, _out_width, self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.input_params[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            batch_size=self.input_params[0],
            input_height=self.input_params[1],
            input_width=self.input_params[2],
            conv_config=conv_config,
        )
        return output_tensor
