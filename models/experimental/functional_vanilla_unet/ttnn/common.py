# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

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
        activation="relu",
        dtype=ttnn.bfloat16,
        auto_shard=False,
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
        self.groups = 1
        self.dtype = dtype
        self.shard_layout = (
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED if height_sharding else ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        if auto_shard:
            self.shard_layout = None

    def __call__(self, device, input_tensor):
        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        conv_config = ttnn.Conv2dConfig(
            dtype=self.dtype,
            weights_dtype=ttnn.bfloat16,
            activation=self.activation,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
            reallocate_halo_output=False,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h

        output_tensor, [_out_height, _out_width], [self.weights, self.bias] = ttnn.conv2d(
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
            dtype=self.dtype,
            weights_dtype=ttnn.bfloat16,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            output_layout=ttnn.ROW_MAJOR_LAYOUT,
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
        )

        output_tensor = ttnn.reshape(
            output_tensor, (input_tensor.shape[0], _out_height, _out_width, output_tensor.shape[3])
        )

        del _out_height, _out_width

        return output_tensor


class ConvSplit:
    def __init__(
        self,
        conv_params,
        parameters,
        *,
        act_block_h=None,
        reshard=False,
        deallocate=False,
        height_sharding=True,
        activation="",
        dtype=ttnn.bfloat16,
        auto_shard=True,
        split_factor=2,
    ) -> None:
        self.split_factor = split_factor
        self.weights = parameters["weight"]
        self.bias = parameters["bias"]

        self.kernel_size = (self.weights.shape[2], self.weights.shape[3])
        self.conv_params = conv_params
        self.out_channels = self.weights.shape[0]
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

    def __call__(self, device, input_tensor):
        input_channels = input_tensor.shape[3]
        assert input_channels % self.split_factor == 0
        split_input_channels = input_channels // self.split_factor

        split_input_tensors = ttnn.split(input_tensor, self.split_factor, 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(input_tensor)
        split_weight_tensors = ttnn.split(self.weights, self.split_factor, 1)

        compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )
        conv_config = ttnn.Conv2dConfig(
            dtype=self.dtype,
            weights_dtype=ttnn.bfloat16,
            activation=self.activation,
            shard_layout=self.shard_layout,
            reshard_if_not_optimal=self.reshard,
            deallocate_activation=self.deallocate,
            output_layout=ttnn.TILE_LAYOUT,
        )
        if self.act_block_h is not None:
            conv_config.act_block_h_override = self.act_block_h
        tt_weight_tensor = split_weight_tensors
        for i in range(self.split_factor):
            input_tensor = split_input_tensors[i]
            tt_output_tensor_on_device, [_out_height, _out_width], [self.weights, self.bias] = ttnn.conv2d(
                input_tensor=input_tensor,
                weight_tensor=tt_weight_tensor[i],
                bias_tensor=self.bias,
                in_channels=input_tensor.shape[3],
                out_channels=tt_weight_tensor[i].shape[0],
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
            )

            conv_output_tensor = ttnn.reshape(
                tt_output_tensor_on_device,
                (input_tensor.shape[0], _out_height, _out_width, tt_output_tensor_on_device.shape[3]),
            )
            if i == 0:
                output_tensor = conv_output_tensor
            else:
                output_tensor = ttnn.add(output_tensor, conv_output_tensor)

            del _out_height, _out_width

        return output_tensor
