# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores


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
        dtype=ttnn.bfloat16,
        use_shallow_conv_variant=False,
        core_count=None,
        override_sharding_config=False,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]
        self.weights = ttnn.from_device(self.weights)
        self.bias = ttnn.from_device(self.bias)
        self.core_count = core_count
        self.override_sharding_config = override_sharding_config

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
        self.use_shallow_conv_variant = (use_shallow_conv_variant,)

    def __call__(self, device, input_tensor):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat16,
            activation=self.activation,
            shard_layout=self.shard_layout,
            deallocate_activation=self.deallocate,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )

        if self.core_count is not None:
            shard_grid = get_shard_grid_from_num_cores(self.core_count, device)

            conv_config.core_grid = shard_grid
            conv_config.override_sharding_config = True

        if self.override_sharding_config:
            self.override_sharding_config = True

        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        # Enabling kernel stride folding lands in to matmul where is fails because of
        # dimensionality mismatch. Disabling it to avoid the issue. Once #31313 is fixed, this can be re-enabled.
        conv_config.enable_kernel_stride_folding = False
        [output_tensor, [_out_height, _out_width], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=input_tensor.shape[3],
            out_channels=self.out_channels,
            device=device,
            kernel_size=self.kernel_size,
            stride=(self.conv_params[0], self.conv_params[1]),
            padding=(self.conv_params[2], self.conv_params[3]),
            batch_size=input_tensor.shape[0],
            input_height=input_tensor.shape[1],
            input_width=input_tensor.shape[2],
            conv_config=conv_config,
            compute_config=compute_config,
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.dtype,
        )

        output_tensor = ttnn.reshape(
            output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[3])
        )
        del _out_height, _out_width

        return output_tensor


def preprocess_linear_weight(weight, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    weight = weight.T.contiguous()
    weight = ttnn.from_torch(weight, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return weight


def preprocess_linear_bias(bias, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    bias = bias.reshape((1, -1))
    bias = ttnn.from_torch(bias, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return bias


def preprocess_layernorm_parameter(parameter, *, dtype, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    parameter = parameter.reshape((1, -1))
    parameter = ttnn.from_torch(parameter, dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    return parameter
