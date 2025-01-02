# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


def fold_bn_to_conv_weights_bias(model, path):
    bn_weight = model[path + ".conv.1.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = model[path + ".conv.1.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = model[path + ".conv.0.weight"]
    weight = (weight / torch.sqrt(bn_running_var)) * bn_weight

    bn_running_mean = model[path + ".conv.1.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = model[path + ".conv.1.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var)) + bn_bias

    bias = bias.reshape(1, 1, 1, -1)
    return (
        ttnn.from_torch(
            weight,
        ),
        ttnn.from_torch(bias),
    )


class Conv:
    def __init__(
        self,
        model,
        path,
        input_params,
        conv_params,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="",
        fused_op=True,
        width_sharding=False,
        output_layout=ttnn.TILE_LAYOUT,
        enable_split_reader=False,
        enable_act_double_buffer=False,
    ) -> None:
        if fused_op:
            self.weights, self.bias = fold_bn_to_conv_weights_bias(model, path)
        else:
            weight = model[path + ".conv.0.weight"]
            bias = model[path + ".conv.0.bias"]
            self.weights = ttnn.from_torch(weight)
            bias = bias.reshape(1, 1, 1, -1)
            self.bias = ttnn.from_torch(bias)
        self.input_params = input_params
        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.output_layout = output_layout
        self.enable_split_reader = enable_split_reader
        self.enable_act_double_buffer = enable_act_double_buffer

        if width_sharding:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
        else:
            self.shard_layout = (
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            )
        self.deallocate = deallocate
        self.activation = activation

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            dtype=ttnn.bfloat16,
            weights_dtype=ttnn.bfloat8_b,
            activation=self.activation,
            shard_layout=self.shard_layout,
            act_block_w_div=1,
            input_channels_alignment=16 if self.input_params[3] < 16 else 32,
            transpose_shards=False,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=False,
            enable_split_reader=self.enable_split_reader,
            enable_act_double_buffer=self.enable_act_double_buffer,
            output_layout=self.output_layout,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        conv_kwargs = {
            "in_channels": self.input_params[3],
            "out_channels": self.out_channels,
            "batch_size": self.input_params[0],
            "input_height": self.input_params[1],
            "input_width": self.input_params[2],
            "kernel_size": self.kernel_size,
            "stride": (self.conv_params[0], self.conv_params[1]),
            "padding": (self.conv_params[2], self.conv_params[3]),
            "dilation": (1, 1),
            "groups": 1,
            "device": device,
            "conv_config": conv_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.weights):
            self.weights = ttnn.prepare_conv_weights(
                weight_tensor=self.weights,
                weights_format="OIHW",
                input_memory_config=input_tensor.memory_config(),
                input_layout=input_tensor.get_layout(),
                **conv_kwargs,
            )

            self.bias = (
                ttnn.prepare_conv_bias(
                    bias_tensor=self.bias,
                    input_memory_config=input_tensor.memory_config(),
                    input_layout=input_tensor.get_layout(),
                    **conv_kwargs,
                )
                if self.bias is not None
                else None
            )

            self.weights = ttnn.to_device(self.weights, device)
            self.bias = ttnn.to_device(self.bias, device) if self.bias else None

        output_tensor = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            **conv_kwargs,
            compute_config=compute_config,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        return output_tensor
