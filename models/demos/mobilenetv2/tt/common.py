# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn


class TtMobileNetV2Conv2D:
    def __init__(
        self,
        input_params,
        parameters,
        device,
        batch_size,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=False,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        reshard_if_not_optimal=True,
        activation_dtype=ttnn.bfloat8_b,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        activation_function=None,
    ):
        self.device = device
        self.parameters = parameters
        self.activation_dtype = activation_dtype
        self.input_params = input_params
        self.groups = groups
        self.dilation = dilation
        self.act_block_h = act_block_h
        self.block_shard = block_shard
        self.deallocate_activation = deallocate_activation
        self.output_layout = output_layout
        self.width_shard = width_shard
        self.act_blocks = act_blocks
        self.enable_act_double_buffer = enable_act_double_buffer
        self.enable_split_reader = enable_split_reader
        self.reshard_if_not_optimal = reshard_if_not_optimal
        self.batch_size = batch_size
        self.shard_layout = shard_layout
        self.activation_function = activation_function
        if self.block_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        if self.width_shard:
            self.shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            activation=self.activation_function if self.activation_function is not None else "",
            shard_layout=self.shard_layout,
            act_block_w_div=1,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_split_reader=True
            if self.shard_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            else self.enable_split_reader,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=self.reshard_if_not_optimal,
            enable_weights_double_buffer=True,
        )

        if self.act_block_h:
            conv_config.act_block_h_override = self.act_blocks

        if self.block_shard:
            conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

        return conv_config

    def _initialize_compute_config(self):
        return ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def __call__(self, x):
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt((x.shape[2] // self.batch_size)))
            input_width = int(math.sqrt((x.shape[2] // self.batch_size)))
        [x, [h, w], [self.weights, self.bias]] = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weights,
            in_channels=x.shape[3],
            out_channels=self.input_params[3],
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            dilation=(self.dilation, self.dilation),
            batch_size=self.batch_size,
            input_height=input_height,
            input_width=input_width,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=self.activation_dtype,
        )
        return x, h, w


class TtInvertedResidual:
    def __init__(
        self, model_params, device, batchsize, expand_ratio, stride, in_channels, out_channels, id, block_shard=False
    ):
        self.device = device
        self.batchsize = batchsize
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_shard = block_shard
        self.id = id
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        self.conv1 = None
        if expand_ratio != 1:
            self.conv1 = TtMobileNetV2Conv2D(
                [1, 1, 0, hidden_dim],
                (model_params[f"fused_conv_{id * 3 - id}_weight"], model_params[f"fused_conv_{id * 3 - id}_bias"]),
                device,
                batchsize,
                block_shard=False if id == 6 and (11 < id <= 16) else self.block_shard,
                deallocate_activation=True if not self.use_res_connect else False,
                enable_act_double_buffer=True,
                activation_function="relu6",
            )

        self.conv2 = TtMobileNetV2Conv2D(
            [3, stride, 1, hidden_dim],
            (model_params[f"fused_conv_{id * 3 -id +1}_weight"], model_params[f"fused_conv_{id * 3 - id + 1}_bias"]),
            device,
            batchsize,
            groups=hidden_dim,
            block_shard=self.block_shard,
            deallocate_activation=True,
            activation_function="relu6",
        )
        self.conv3 = TtMobileNetV2Conv2D(
            [1, 1, 0, out_channels],
            (model_params[f"conv_{id}_weight"], model_params[f"conv_{id}_bias"]),
            device,
            batchsize,
            block_shard=False if (10 <= id <= 16) else self.block_shard,
            deallocate_activation=True,
            enable_act_double_buffer=True,
        )

    def __call__(self, x):
        identity = x
        if self.conv1 is not None:
            x, h, w = self.conv1(x)
        out, h, w = self.conv2(x)
        out, h, w = self.conv3(out)
        if self.use_res_connect:
            tmp = ttnn.add(identity, out)
            ttnn.deallocate(identity)
            ttnn.deallocate(out)
            out = tmp
        return out
