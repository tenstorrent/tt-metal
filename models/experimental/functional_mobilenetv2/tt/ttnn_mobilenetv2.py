# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math


class MobileNetV2Conv2D:
    def __init__(
        self,
        input_params,
        parameters,
        device,
        batch_size,
        groups=1,
        dilation=1,
        act_block_h=False,
        block_shard=None,
        deallocate_activation=False,
        output_layout=ttnn.TILE_LAYOUT,
        width_shard=False,
        act_blocks=32,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        reshard_if_not_optimal=False,
        use_shallow_covariant=False,
        cache={},
        activation_dtype=ttnn.bfloat8_b,
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
        self.cache = cache
        self.batch_size = batch_size
        self.use_shallow_covariant = use_shallow_covariant
        self.conv_config = self._initialize_conv_config()
        self.compute_config = self._initialize_compute_config()
        self.weights, self.bias = self.parameters

    def _initialize_conv_config(self):
        conv_config = ttnn.Conv2dConfig(
            dtype=self.activation_dtype,
            weights_dtype=ttnn.bfloat8_b,
            activation="",
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            input_channels_alignment=16 if self.use_shallow_covariant else 32,
            act_block_w_div=1,
            transpose_shards=False,
            deallocate_activation=False,
            enable_act_double_buffer=self.enable_act_double_buffer,
            enable_split_reader=self.enable_split_reader,
            enable_subblock_padding=False,
            output_layout=self.output_layout,
            reallocate_halo_output=False,
            reshard_if_not_optimal=True,
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
            conv_op_cache=self.cache,
            debug=False,
            groups=self.groups,
            return_weights_and_bias=True,
            return_output_dim=True,
        )

        return x, h, w


class MobileNetV2:
    def __init__(self, model_params, device, batchsize) -> None:
        self.device = device
        self.model_parameters = model_params
        self.batchsize = batchsize

        self.c1 = MobileNetV2Conv2D([3, 2, 1, 32], model_params[1], device, batchsize, use_shallow_covariant=True)
        self.c2 = MobileNetV2Conv2D([3, 1, 1, 32], model_params[2], device, batchsize, groups=32)
        self.c3 = MobileNetV2Conv2D([1, 1, 0, 16], model_params[3], device, batchsize)
        self.c4 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[4], device, batchsize)
        self.c5 = MobileNetV2Conv2D([3, 2, 1, 96], model_params[5], device, batchsize, groups=96)
        self.c6 = MobileNetV2Conv2D([1, 1, 0, 24], model_params[6], device, batchsize, deallocate_activation=False)
        self.c7 = MobileNetV2Conv2D([1, 1, 0, 144], model_params[7], device, batchsize)
        self.c8 = MobileNetV2Conv2D([3, 1, 1, 144], model_params[8], device, batchsize, groups=144)
        self.c9 = MobileNetV2Conv2D([1, 1, 0, 24], model_params[9], device, batchsize, deallocate_activation=False)
        self.c10 = MobileNetV2Conv2D([1, 1, 0, 144], model_params[10], device, batchsize)
        self.c11 = MobileNetV2Conv2D([3, 2, 1, 144], model_params[11], device, batchsize, groups=144)
        self.c12 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[12], device, batchsize)
        self.c13 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[13], device, batchsize)
        self.c14 = MobileNetV2Conv2D([3, 1, 1, 192], model_params[14], device, batchsize, groups=192)
        self.c15 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[15], device, batchsize)
        self.c16 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[16], device, batchsize)
        self.c17 = MobileNetV2Conv2D([3, 1, 1, 192], model_params[17], device, batchsize, groups=192)
        self.c18 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[18], device, batchsize)
        self.c19 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[19], device, batchsize)
        self.c20 = MobileNetV2Conv2D([3, 2, 1, 192], model_params[20], device, batchsize, groups=192)
        self.c21 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[21], device, batchsize)
        self.c22 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[22], device, batchsize)
        self.c23 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[23], device, batchsize, groups=384)
        self.c24 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[24], device, batchsize)
        self.c25 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[25], device, batchsize)
        self.c26 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[26], device, batchsize, groups=384)
        self.c27 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[27], device, batchsize)
        self.c28 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[28], device, batchsize)
        self.c29 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[29], device, batchsize, groups=384)
        self.c30 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[30], device, batchsize)
        self.c31 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[31], device, batchsize)
        self.c32 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[32], device, batchsize, groups=384)
        self.c33 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[33], device, batchsize)
        self.c34 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[34], device, batchsize)
        self.c35 = MobileNetV2Conv2D(
            [3, 1, 1, 576],
            model_params[35],
            device,
            batchsize,
            groups=576,
            block_shard=True,
        )
        self.c36 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[36], device, batchsize)
        self.c37 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[37], device, batchsize)
        self.c38 = MobileNetV2Conv2D([3, 1, 1, 576], model_params[38], device, batchsize, groups=576, block_shard=True)
        self.c39 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[39], device, batchsize)
        self.c40 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[40], device, batchsize)
        self.c41 = MobileNetV2Conv2D([3, 2, 1, 576], model_params[41], device, batchsize, groups=576, block_shard=True)
        self.c42 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[42], device, batchsize)
        self.c43 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[43], device, batchsize)
        self.c44 = MobileNetV2Conv2D([3, 1, 1, 960], model_params[44], device, batchsize, groups=960, block_shard=True)
        self.c45 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[45], device, batchsize)
        self.c46 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[46], device, batchsize)
        self.c47 = MobileNetV2Conv2D([3, 1, 1, 960], model_params[47], device, batchsize, groups=960, block_shard=True)
        self.c48 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[48], device, batchsize)
        self.c49 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[49], device, batchsize)
        self.c50 = MobileNetV2Conv2D([3, 1, 1, 960], model_params[50], device, batchsize, groups=960, block_shard=True)
        self.c51 = MobileNetV2Conv2D([1, 1, 0, 320], model_params[51], device, batchsize)
        self.c52 = MobileNetV2Conv2D([1, 1, 0, 1280], model_params[52], device, batchsize, block_shard=True)
        self.l1_weight = model_params["l1"]["weight"]
        self.l1_bias = model_params["l1"]["bias"]

    def __call__(
        self,
        x,
    ):
        output_tensor, h, w = self.c1(x)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c2(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c3(output_tensor)

        output_tensor, h, w = self.c4(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c5(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c6(output_tensor)

        output_tensor_c6 = output_tensor

        output_tensor, h, w = self.c7(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c8(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c9(output_tensor)

        output_tensor = ttnn.add(output_tensor_c6, output_tensor)
        output_tensor, h, w = self.c10(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c11(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c12(output_tensor)
        output_tensor_c12 = output_tensor

        output_tensor, h, w = self.c13(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c14(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c15(output_tensor)
        output_tensor_c15 = output_tensor
        output_tensor = output_tensor_c15 + output_tensor_c12
        output_tensor_a2 = output_tensor

        output_tensor, h, w = self.c16(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c17(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c18(output_tensor)
        output_tensor = output_tensor_a2 + output_tensor

        output_tensor, h, w = self.c19(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c20(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c21(output_tensor)
        output_tensor_c21 = output_tensor

        output_tensor, h, w = self.c22(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c23(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c24(output_tensor)
        output_tensor = output_tensor_c21 + output_tensor
        output_tensor_a4 = output_tensor

        output_tensor, h, w = self.c25(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c26(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c27(output_tensor)
        output_tensor = output_tensor_a4 + output_tensor
        output_tensor_a5 = output_tensor

        output_tensor, h, w = self.c28(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c29(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c30(output_tensor)
        output_tensor = ttnn.add(output_tensor_a5, output_tensor)

        output_tensor, h, w = self.c31(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c32(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c33(output_tensor)
        output_tensor_c33 = output_tensor

        output_tensor, h, w = self.c34(output_tensor_c33)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c35(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c36(output_tensor)
        output_tensor = output_tensor_c33 + output_tensor
        output_tensor_a7 = output_tensor

        output_tensor, h, w = self.c37(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c38(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c39(output_tensor)
        output_tensor = output_tensor_a7 + output_tensor

        output_tensor, h, w = self.c40(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c41(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c42(output_tensor)
        output_tensor_c42 = output_tensor

        output_tensor, h, w = self.c43(output_tensor_c42)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c44(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c45(output_tensor)
        output_tensor = output_tensor_c42 + output_tensor
        output_tensor_a9 = output_tensor

        output_tensor, h, w = self.c46(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c47(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c48(output_tensor)
        output_tensor = output_tensor + output_tensor_a9

        output_tensor, h, w = self.c49(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c50(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor, h, w = self.c51(output_tensor)

        output_tensor, h, w = self.c52(output_tensor)

        output_tensor = ttnn.relu6(output_tensor)
        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (self.batchsize, h, w, output_tensor.shape[3]))
        output_tensor = ttnn.global_avg_pool2d(output_tensor)

        output_tensor = ttnn.reshape(output_tensor, (self.batchsize, -1))

        output_tensor = ttnn.linear(output_tensor, self.l1_weight, bias=self.l1_bias)

        return output_tensor
