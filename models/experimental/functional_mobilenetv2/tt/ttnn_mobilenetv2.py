# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
from ttnn.model_preprocessing import ParameterDict


class MobileNetV2Conv2D:
    def __init__(
        self,
        input_params,
        parameters,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        groups=1,
        deallocate_activation=False,
    ):
        self.device = device
        self.batch_size = 1
        self.input_params = input_params
        self.groups = groups
        self.parameters = parameters
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = deallocate_activation
        self.cache = cache

        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        self.weight, self.bias = self.parameters

    def __call__(self, x):
        if x.shape[1] != 1:
            input_height = x.shape[1]
            input_width = x.shape[2]
        else:
            input_height = int(math.sqrt(x.shape[2]) // self.batch_size)
            input_width = int(math.sqrt(x.shape[2]) // self.batch_size)
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=x.shape[3],
            out_channels=self.input_params[3],
            input_height=input_height,
            input_width=input_width,
            batch_size=self.batch_size,
            kernel_size=(self.input_params[0], self.input_params[0]),
            stride=(self.input_params[1], self.input_params[1]),
            padding=(self.input_params[2], self.input_params[2]),
            conv_config=self.conv_config,
            conv_op_cache=self.cache,
            groups=self.groups,
            compute_config=self.compute_config,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        return x


class MobileNetV2:
    def __init__(self, model_params, device) -> None:
        self.device = device
        self.model_parameters = model_params

        self.c1 = MobileNetV2Conv2D([3, 2, 1, 32], model_params[1], device)
        self.c2 = MobileNetV2Conv2D([3, 1, 1, 32], model_params[2], device, groups=32)
        self.c3 = MobileNetV2Conv2D([1, 1, 0, 16], model_params[3], device)
        self.c4 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[4], device)
        self.c5 = MobileNetV2Conv2D([3, 2, 1, 96], model_params[5], device, groups=96)
        self.c6 = MobileNetV2Conv2D([1, 1, 0, 24], model_params[6], device)
        self.c7 = MobileNetV2Conv2D([1, 1, 0, 144], model_params[7], device)
        self.c8 = MobileNetV2Conv2D([3, 1, 1, 144], model_params[8], device, groups=144)
        self.c9 = MobileNetV2Conv2D([1, 1, 0, 24], model_params[9], device)
        self.c10 = MobileNetV2Conv2D([1, 1, 0, 144], model_params[10], device)
        self.c11 = MobileNetV2Conv2D([3, 2, 1, 144], model_params[11], device, groups=144)
        self.c12 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[12], device)
        self.c13 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[13], device)
        self.c14 = MobileNetV2Conv2D([3, 1, 1, 192], model_params[14], device, groups=192)
        self.c15 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[15], device)
        self.c16 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[16], device)
        self.c17 = MobileNetV2Conv2D([3, 1, 1, 192], model_params[17], device, groups=192)
        self.c18 = MobileNetV2Conv2D([1, 1, 0, 32], model_params[18], device)
        self.c19 = MobileNetV2Conv2D([1, 1, 0, 192], model_params[19], device)
        self.c20 = MobileNetV2Conv2D([3, 2, 1, 192], model_params[20], device, groups=192)
        self.c21 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[21], device)
        self.c22 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[22], device)
        self.c23 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[23], device, groups=384)
        self.c24 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[24], device)
        self.c25 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[25], device)
        self.c26 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[26], device, groups=384)
        self.c27 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[27], device)
        self.c28 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[28], device)
        self.c29 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[29], device, groups=384)
        self.c30 = MobileNetV2Conv2D([1, 1, 0, 64], model_params[30], device)
        self.c31 = MobileNetV2Conv2D([1, 1, 0, 384], model_params[31], device)
        self.c32 = MobileNetV2Conv2D([3, 1, 1, 384], model_params[32], device, groups=384)
        self.c33 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[33], device)
        self.c34 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[34], device)
        self.c35 = MobileNetV2Conv2D(
            [3, 1, 1, 576],
            model_params[35],
            device,
            groups=576,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            use_1d_systolic_array=True,
        )
        self.c36 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[36], device)
        self.c37 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[37], device)
        self.c38 = MobileNetV2Conv2D(
            [3, 1, 1, 576], model_params[38], device, groups=576, shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.c39 = MobileNetV2Conv2D([1, 1, 0, 96], model_params[39], device)
        self.c40 = MobileNetV2Conv2D([1, 1, 0, 576], model_params[40], device)
        self.c41 = MobileNetV2Conv2D(
            [3, 2, 1, 576], model_params[41], device, groups=576, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )
        self.c42 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[42], device)
        self.c43 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[43], device)
        self.c44 = MobileNetV2Conv2D(
            [3, 1, 1, 960], model_params[44], device, groups=960, shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.c45 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[45], device)
        self.c46 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[46], device)
        self.c47 = MobileNetV2Conv2D(
            [3, 1, 1, 960], model_params[47], device, groups=960, shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.c48 = MobileNetV2Conv2D([1, 1, 0, 160], model_params[48], device)
        self.c49 = MobileNetV2Conv2D([1, 1, 0, 960], model_params[49], device)
        self.c50 = MobileNetV2Conv2D(
            [3, 1, 1, 960], model_params[50], device, groups=960, shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.c51 = MobileNetV2Conv2D([1, 1, 0, 320], model_params[51], device)
        self.c52 = MobileNetV2Conv2D(
            [1, 1, 0, 1280], model_params[52], device, shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED
        )
        self.l1_weight = model_params["l1"]["weight"]
        self.l1_bias = model_params["l1"]["bias"]

    def __call__(
        self,
        device,
        x,
    ):
        output_tensor = self.c1(x)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c3(output_tensor)
        output_tensor = self.c4(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c5(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c6(output_tensor)
        output_tensor_c6 = output_tensor
        output_tensor = self.c7(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c8(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c9(output_tensor)
        output_tensor = ttnn.add(output_tensor_c6, output_tensor)
        output_tensor = self.c10(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c11(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c12(output_tensor)
        output_tensor_c12 = output_tensor
        output_tensor = self.c13(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c14(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c15(output_tensor)
        output_tensor_c15 = output_tensor
        output_tensor = output_tensor_c15 + output_tensor_c12
        output_tensor_a2 = output_tensor
        output_tensor = self.c16(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c17(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c18(output_tensor)
        output_tensor = output_tensor_a2 + output_tensor
        output_tensor = self.c19(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c20(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c21(output_tensor)
        output_tensor_c21 = output_tensor
        output_tensor = self.c22(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c23(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c24(output_tensor)
        output_tensor = output_tensor_c21 + output_tensor
        output_tensor_a4 = output_tensor
        output_tensor = self.c25(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c26(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c27(output_tensor)
        output_tensor = output_tensor_a4 + output_tensor
        output_tensor_a5 = output_tensor
        output_tensor = self.c28(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c29(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c30(output_tensor)
        output_tensor = ttnn.add(output_tensor_a5, output_tensor)
        output_tensor = self.c31(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c32(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c33(output_tensor)
        output_tensor_c33 = output_tensor
        output_tensor = self.c34(output_tensor_c33)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c35(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c36(output_tensor)
        output_tensor = output_tensor_c33 + output_tensor
        output_tensor_a7 = output_tensor
        output_tensor = self.c37(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c38(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c39(output_tensor)
        output_tensor = output_tensor_a7 + output_tensor
        output_tensor = self.c40(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c41(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c42(output_tensor)
        output_tensor_c42 = output_tensor
        output_tensor = self.c43(output_tensor_c42)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c44(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c45(output_tensor)
        output_tensor = output_tensor_c42 + output_tensor
        output_tensor_a9 = output_tensor
        output_tensor = self.c46(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c47(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c48(output_tensor)
        output_tensor = output_tensor + output_tensor_a9
        output_tensor = self.c49(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c50(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = self.c51(output_tensor)
        output_tensor = self.c52(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.global_avg_pool2d(output_tensor)
        output_tensor = ttnn.reshape(output_tensor, (output_tensor.shape[0], -1))
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.linear(output_tensor, self.l1_weight, bias=self.l1_bias)

        return output_tensor
