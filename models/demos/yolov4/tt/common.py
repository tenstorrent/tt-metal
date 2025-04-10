# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class Conv:
    def __init__(
        self,
        device,
        conv_param,
        conv_pth,
        activation="",
    ) -> None:
        self.conv_param = conv_param
        self.conv_pth = conv_pth
        self.device = device
        self.cache = {}

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=False,
        )
        self.conv_config = ttnn.Conv2dConfig(
            dtype=conv_param.dtype,
            weights_dtype=ttnn.bfloat8_b,
            activation=activation,
            shard_layout=conv_param.shard_layout,
            input_channels_alignment=16 if conv_param.in_channels < 16 else 32,
            reshard_if_not_optimal=conv_param.reshard_if_not_optimal,
            deallocate_activation=conv_param.deallocate_activation,
            enable_act_double_buffer=conv_param.enable_act_double_buffer,
            enable_split_reader=conv_param.enable_split_reader,
            output_layout=ttnn.TILE_LAYOUT,
            transpose_shards=conv_param.transpose_shards,
        )
        config_override = None
        if conv_param.act_block_h is not None:
            self.conv_config.act_block_h_override = conv_param.act_block_h

        if "bias" in conv_pth:
            bias = ttnn.from_device(conv_pth.bias)
            self.bias = bias
        else:
            self.bias = None

        weight = ttnn.from_device(conv_pth.weight)
        self.weight = weight

        if conv_param.shard_layout is None:
            self.input_memory_config = ttnn.L1_MEMORY_CONFIG
        elif (
            conv_param.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            and conv_param.shard_layout != ttnn.TensorMemoryLayout.WIDTH_SHARDED
        ):
            self.input_memory_config = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        else:
            self.input_memory_config = ttnn.L1_BLOCK_SHARDED_MEMORY_CONFIG

        self.conv_kwargs = {
            "in_channels": conv_param.in_channels,
            "out_channels": conv_param.out_channels,
            "batch_size": conv_param.batch_size,
            "input_height": conv_param.input_height,
            "input_width": conv_param.input_width,
            "kernel_size": conv_param.kernel_size,
            "stride": conv_param.stride,
            "padding": conv_param.padding,
            "dilation": conv_param.dilation,
            "groups": conv_param.groups,
            "device": device,
            "conv_config": self.conv_config,
        }

        if not ttnn.is_tensor_storage_on_device(self.weight):
            self.weight = ttnn.prepare_conv_weights(
                weight_tensor=self.weight,
                weights_format="OIHW",
                input_memory_config=self.input_memory_config,
                input_layout=ttnn.TILE_LAYOUT,
                has_bias=True,
                **self.conv_kwargs,
            )

            self.bias = ttnn.prepare_conv_bias(
                bias_tensor=self.bias,
                input_memory_config=self.input_memory_config,
                input_layout=ttnn.TILE_LAYOUT,
                **self.conv_kwargs,
            )
            self.weight = ttnn.to_device(self.weight, device)
            self.bias = ttnn.to_device(self.bias, device)

    def __str__(self) -> str:
        return f"Conv: {self.weights.shape} {self.bias.shape} {self.kernel_size}"

    def __call__(self, input_tensor):
        [x, [output_height, output_width], [self.weight, self.bias]] = ttnn.conv2d(
            input_tensor=input_tensor,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            **self.conv_kwargs,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
        )
        return x, output_height, output_width
