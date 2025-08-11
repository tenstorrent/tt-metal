# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


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
        activation=None,
        groups=1,
        dtype=ttnn.bfloat8_b,
        output_layout=ttnn.TILE_LAYOUT,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.deallocate = deallocate
        self.activation = activation
        self.groups = groups
        self.dtype = dtype
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.output_layout = output_layout

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=self.activation,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            output_layout=self.output_layout,
        )
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        conv_kwargs = {
            "in_channels": input_tensor.shape[3],
            "out_channels": self.out_channels,
            "batch_size": input_tensor.shape[0],
            "input_height": input_tensor.shape[1],
            "input_width": input_tensor.shape[2],
            "kernel_size": self.kernel_size,
            "stride": (self.conv_params[0], self.conv_params[1]),
            "padding": (self.conv_params[2], self.conv_params[3]),
            "dilation": (1, 1),
            "groups": self.groups,
            "device": device,
            "conv_config": conv_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.weights):
            self.weights = ttnn.prepare_conv_weights(
                weight_tensor=self.weights,
                weights_format="OIHW",
                input_memory_config=input_tensor.memory_config(),
                input_layout=input_tensor.get_layout(),
                has_bias=True,
                **conv_kwargs,
                input_dtype=self.dtype,
            )
            self.bias = ttnn.prepare_conv_bias(
                bias_tensor=self.bias,
                input_memory_config=input_tensor.memory_config(),
                input_layout=input_tensor.get_layout(),
                **conv_kwargs,
                input_dtype=self.dtype,
            )
            self.weights = ttnn.to_device(self.weights, device)
            self.bias = ttnn.to_device(self.bias, device)

        [output_tensor, [_out_height, _out_width]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            **conv_kwargs,
            compute_config=compute_config,
            return_output_dim=True,
            return_weights_and_bias=False,
            dtype=self.dtype,
        )

        return output_tensor, _out_height, _out_width


def preprocess_layernorm_parameter(parameter, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return parameter


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return bias


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
    return weight, bias
