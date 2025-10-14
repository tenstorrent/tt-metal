# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

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
        activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
        dtype=ttnn.bfloat16,
        auto_shard=False,
        enable_act_double_buffer=False,
        enable_weights_double_buffer=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        reallocate_halo_output=False,
    ) -> None:
        self.weights = parameters["weight"]
        if parameters["bias"] is not None:
            self.bias = parameters["bias"]
        else:
            self.bias = None

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
        self.in_channels = self.weights.shape[1]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.deallocate = deallocate
        self.activation = activation
        self.groups = 1
        self.dtype = dtype
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        if auto_shard:
            self.shard_layout = None
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_weights_double_buffer = enable_weights_double_buffer
        self.output_layout = output_layout
        self.reallocate_halo_output = reallocate_halo_output

    def __call__(self, device, input_tensor):
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            activation=self.activation,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            output_layout=self.output_layout,
            reallocate_halo_output=self.reallocate_halo_output,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_weights_double_buffer=self.enable_weights_double_buffer,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        output_tensor, [_out_height, _out_width], [self.weights, self.bias] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weights,
            bias_tensor=self.bias,
            in_channels=self.in_channels,  # input_tensor.shape[3],
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
            slice_config=ttnn.Conv2dL1FullSliceConfig,
        )

        output_tensor = ttnn.reshape(
            output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[3])
        )

        del _out_height, _out_width

        return output_tensor


class ConvTranspose:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        dtype=ttnn.bfloat16,
        auto_shard=False,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
    ) -> None:
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[1]
        self.act_block_h = act_block_h
        self.reshard = reshard
        self.deallocate = deallocate
        self.groups = 1
        self.dtype = dtype
        self.output_layout = output_layout
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        if auto_shard:
            self.shard_layout = None

    def __call__(self, device, input_tensor):
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            output_layout=self.output_layout,
            reallocate_halo_output=True,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        output_tensor, [_out_height, _out_width], [self.weights, self.bias] = ttnn.conv_transpose2d(
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
            output_padding=(0, 0),
            dilation=(1, 1),
            mirror_kernel=True,
            dtype=self.dtype,
        )

        output_tensor = ttnn.reshape(
            output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[3])
        )

        del _out_height, _out_width

        return output_tensor


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = None
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer
