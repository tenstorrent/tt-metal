# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtnnConv1D:
    def __init__(
        self,
        conv,
        parameters,
        device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=None,  # ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        fp32_accum=False,
        packer_l1_acc=False,
        activation=None,
        deallocate_activation=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
    ):
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size[0]
        self.padding = conv.padding[0]
        self.stride = conv.stride[0]
        self.groups = conv.groups
        self.conv_config = ttnn.Conv1dConfig(
            # dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
            activation=activation,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_accum,
            packer_l1_acc=packer_l1_acc,
        )
        if "bias" in parameters and parameters["bias"] is not None:
            bias = ttnn.from_device(parameters.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(parameters.weight)
        self.weight = weight

    def __call__(self, x):
        input_length = x.shape[1]  # self.conv.input_length
        batch_size = 1  # self.conv.batch_size
        [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch_size,
            input_length=input_length,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        # print(f"{out_length=}")
        # tt_output_tensor_on_device = ttnn.reshape(
        #     tt_output_tensor_on_device,
        #     (1, out_length, tt_output_tensor_on_device.shape[-2] // out_length, tt_output_tensor_on_device.shape[-1]),
        # )
        # print(f"{tt_output_tensor_on_device.shape=}")
        return tt_output_tensor_on_device


class TtnnConv2D:
    def __init__(
        self,
        conv,
        conv_pth,
        device=None,
        cache={},
        activation=None,
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        shard_layout=None,
        # shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        is_dealloc_act=False,
        return_dims=False,
    ):
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.cache = cache
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        self.is_dealloc_act = is_dealloc_act
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.is_dealloc_act,
            enable_act_double_buffer=False,
            # enable_split_reader=False,
            # enable_subblock_padding=False,
            reshard_if_not_optimal=False,
            activation=activation,
        )
        # if self.conv_config.shard_layout is None:
        #     self.input_memory_config = ttnn.L1_MEMORY_CONFIG
        # elif self.conv_config.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        #     self.input_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        # elif self.conv_config.shard_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED:
        #     self.input_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

        if conv_pth.bias is not None:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        self.return_dims = return_dims

        self.weight = ttnn.from_device(conv_pth.weight)

    def __call__(self, x, shape=None):
        if shape is not None:
            batch_size = shape[0]
            input_height = shape[1]
            input_width = shape[2]
        else:
            batch_size = x.shape[0]
            input_height = x.shape[1]
            input_width = x.shape[2]

        # self.conv_kwargs = {
        #     "in_channels": self.conv.in_channels,
        #     "out_channels": self.conv.out_channels,
        #     "batch_size": x.shape[0],
        #     "input_height": x.shape[1],
        #     "input_width": x.shape[2],
        #     "kernel_size": self.conv.kernel_size,
        #     "stride": self.conv.stride,
        #     "padding": self.conv.padding,
        #     "dilation": self.conv.dilation,
        #     "groups": self.conv.groups,
        #     "device": self.device,
        #     "conv_config": self.conv_config,
        # }
        # if not ttnn.is_tensor_storage_on_device(self.weight):
        #     self.weight = ttnn.prepare_conv_weights(
        #         weight_tensor=self.weight,
        #         weights_format="OIHW",
        #         input_memory_config=self.input_memory_config,
        #         input_layout=ttnn.TILE_LAYOUT,
        #         has_bias=True,
        #         **self.conv_kwargs,
        #     )

        #     self.bias = ttnn.prepare_conv_bias(
        #         bias_tensor=self.bias,
        #         input_memory_config=self.input_memory_config,
        #         input_layout=ttnn.TILE_LAYOUT,
        #         **self.conv_kwargs,
        #     )
        #     self.weight = ttnn.to_device(self.weight, self.device)
        #     self.bias = ttnn.to_device(self.bias, self.device)

        [x, [_out_height, _out_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            # **self.conv_kwargs,
            in_channels=self.conv.in_channels,
            out_channels=self.conv.out_channels,
            device=self.device,
            kernel_size=self.conv.kernel_size,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        shape = (batch_size, _out_height, _out_width, x.shape[-1])
        if self.return_dims:
            return x, shape
        else:
            del shape
            return x
