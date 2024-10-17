# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from ttnn.model_preprocessing import ParameterDict

from torch import nn
from ttnn.model_preprocessing import preprocess_linear_weight, preprocess_linear_bias


class MobileNetV2Conv2D:
    def fold_batch_norm2d_into_conv2d(self, conv, bn):
        if not bn.track_running_stats:
            raise RuntimeError("BatchNorm2d must have track_running_stats=True to be folded into Conv2d")
        weight = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        eps = bn.eps
        scale = bn.weight
        shift = bn.bias
        weight = weight * (scale / torch.sqrt(running_var + eps))[:, None, None, None]
        bias = shift - running_mean * (scale / torch.sqrt(running_var + eps))
        return weight, bias

    def __init__(
        self,
        conv,
        bn=None,
        device=None,
        cache={},
        activation="",
        activation_dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat8_b,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.device = device
        self.batch_size = conv.batch_size
        self.input_height = conv.input_height
        self.input_width = conv.input_width
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size
        self.padding = conv.padding
        self.stride = conv.stride
        self.groups = conv.groups
        self.use_1d_systolic_array = use_1d_systolic_array
        self.deallocate_activation = True
        self.cache = cache

        self.conv_config = ttnn.Conv2dConfig(
            dtype=activation_dtype,
            weights_dtype=weights_dtype,
            math_fidelity=ttnn.MathFidelity.LoFi,
            shard_layout=shard_layout,
            deallocate_activation=self.deallocate_activation,
            fp32_dest_acc_enabled=True,
            packer_l1_accum_enabled=False,
            enable_act_double_buffer=False,
            enable_split_reader=False,
            enable_subblock_padding=False,
            reshard_if_not_optimal=True if self.use_1d_systolic_array else False,
            activation=activation,
        )
        config_override = conv.conv_blocking_and_parallelization_config_override
        if config_override and "act_block_h" in config_override:
            self.conv_config.act_block_h_override = config_override["act_block_h"]

        if bn is not None:
            weight, bias = self.fold_batch_norm2d_into_conv2d(conv.module, bn.module)
        else:
            weight, bias = conv.module.weight, conv.module.bias

        weight = weight
        bias = torch.reshape(bias, (1, 1, 1, -1))
        self.weight = ttnn.from_torch(weight, dtype=ttnn.float32)
        self.bias = ttnn.from_torch(bias, dtype=ttnn.float32)

    def __call__(self, x):
        x, output_height, output_width, self.weight, self.bias = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            input_height=self.input_height,
            input_width=self.input_width,
            batch_size=self.batch_size,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            conv_config=self.conv_config,
            conv_op_cache=self.cache,
            groups=self.groups,
        )
        return x


class MobileNetV2:
    def input_preprocessor(self, tensor, n, c, h, w):
        tensor = ttnn.to_torch(tensor).to(torch.float32)
        tensor = torch.reshape(tensor, (n, h, w, c))
        tensor = torch.permute(tensor, (0, 3, 1, 2))
        return tensor

    def __init__(self, parameters: ParameterDict, device, model) -> None:
        self.device = device
        self.model = model
        self.parameters = parameters

        self.c1 = MobileNetV2Conv2D(parameters.c1, parameters.b1, device)
        self.c2 = MobileNetV2Conv2D(parameters.c2, parameters.b2, device)

        self.c3 = MobileNetV2Conv2D(parameters.c3, parameters.b3, device)

        self.c4 = MobileNetV2Conv2D(parameters.c4, parameters.b4, device)

        self.c5 = MobileNetV2Conv2D(parameters.c5, parameters.b5, device)

        self.c6 = MobileNetV2Conv2D(parameters.c6, parameters.b6, device)

        self.c7 = MobileNetV2Conv2D(parameters.c7, parameters.b7, device)

        self.c8 = MobileNetV2Conv2D(parameters.c8, parameters.b8, device)

        self.c9 = MobileNetV2Conv2D(parameters.c9, parameters.b9, device)

        self.c10 = MobileNetV2Conv2D(parameters.c10, parameters.b10, device)

        self.c11 = MobileNetV2Conv2D(parameters.c11, parameters.b11, device)

        self.c12 = MobileNetV2Conv2D(parameters.c12, parameters.b12, device)

        self.c13 = MobileNetV2Conv2D(parameters.c13, parameters.b13, device)
        self.c14 = MobileNetV2Conv2D(parameters.c14, parameters.b14, device)
        self.c15 = MobileNetV2Conv2D(parameters.c15, parameters.b15, device)
        self.c16 = MobileNetV2Conv2D(parameters.c16, parameters.b16, device)
        self.c17 = MobileNetV2Conv2D(parameters.c17, parameters.b17, device)
        self.c18 = MobileNetV2Conv2D(parameters.c18, parameters.b18, device)
        self.c19 = MobileNetV2Conv2D(parameters.c19, parameters.b19, device)
        self.c20 = MobileNetV2Conv2D(parameters.c20, parameters.b20, device)
        self.c21 = MobileNetV2Conv2D(parameters.c21, parameters.b21, device)
        self.c22 = MobileNetV2Conv2D(parameters.c22, parameters.b22, device)
        self.c23 = MobileNetV2Conv2D(parameters.c23, parameters.b23, device)
        self.c24 = MobileNetV2Conv2D(parameters.c24, parameters.b24, device)
        self.c25 = MobileNetV2Conv2D(parameters.c25, parameters.b25, device)
        self.c26 = MobileNetV2Conv2D(parameters.c26, parameters.b26, device)
        self.c27 = MobileNetV2Conv2D(parameters.c27, parameters.b27, device)
        self.c28 = MobileNetV2Conv2D(parameters.c28, parameters.b28, device)
        self.c29 = MobileNetV2Conv2D(parameters.c29, parameters.b29, device)
        self.c30 = MobileNetV2Conv2D(parameters.c30, parameters.b30, device)
        self.c31 = MobileNetV2Conv2D(parameters.c31, parameters.b31, device)
        self.c32 = MobileNetV2Conv2D(parameters.c32, parameters.b32, device)
        self.c33 = MobileNetV2Conv2D(parameters.c33, parameters.b33, device)
        self.c34 = MobileNetV2Conv2D(parameters.c34, parameters.b34, device)

        self.c35 = MobileNetV2Conv2D(
            parameters.c35, parameters.b35, device, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )

        self.c36 = MobileNetV2Conv2D(parameters.c36, parameters.b36, device)
        self.c37 = MobileNetV2Conv2D(parameters.c37, parameters.b37, device)

        self.c38 = MobileNetV2Conv2D(
            parameters.c38, parameters.b38, device, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )

        self.c39 = MobileNetV2Conv2D(parameters.c39, parameters.b39, device)
        self.c40 = MobileNetV2Conv2D(parameters.c40, parameters.b40, device)

        self.c41 = MobileNetV2Conv2D(
            parameters.c41, parameters.b41, device, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )

        self.c42 = MobileNetV2Conv2D(parameters.c42, parameters.b42, device)
        self.c43 = MobileNetV2Conv2D(parameters.c43, parameters.b43, device)

        self.c44 = MobileNetV2Conv2D(
            parameters.c44, parameters.b44, device, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )

        self.c45 = MobileNetV2Conv2D(parameters.c45, parameters.b45, device)
        self.c46 = MobileNetV2Conv2D(parameters.c46, parameters.b46, device)

        self.c47 = MobileNetV2Conv2D(
            parameters.c47, parameters.b47, device, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )

        self.c48 = MobileNetV2Conv2D(parameters.c48, parameters.b48, device)
        self.c49 = MobileNetV2Conv2D(parameters.c49, parameters.b49, device)

        self.c50 = MobileNetV2Conv2D(
            parameters.c50, parameters.b50, device, shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED
        )

        self.c51 = MobileNetV2Conv2D(parameters.c51, parameters.b51, device)
        self.c52 = MobileNetV2Conv2D(parameters.c52, parameters.b52, device)

        self.l1_weight = parameters.l1["weight"]
        self.l1_bias = parameters.l1["bias"]

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

        if output_tensor_c6.is_sharded():
            output_tensor_c6 = ttnn.sharded_to_interleaved(output_tensor_c6, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c7(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c8(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c9(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.add(output_tensor_c6, output_tensor)

        output_tensor = self.c10(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c11(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c12(output_tensor)
        output_tensor_c12 = output_tensor

        if output_tensor_c12.is_sharded():
            output_tensor_c12 = ttnn.sharded_to_interleaved(output_tensor_c12, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c13(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c14(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c15(output_tensor)
        output_tensor_c15 = output_tensor

        if output_tensor_c15.is_sharded():
            output_tensor_c15 = ttnn.sharded_to_interleaved(output_tensor_c15, ttnn.L1_MEMORY_CONFIG)

        output_tensor = output_tensor_c15 + output_tensor_c12
        output_tensor_a2 = output_tensor

        if output_tensor_a2.is_sharded():
            output_tensor_a2 = ttnn.sharded_to_interleaved(output_tensor_a2, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c16(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c17(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c18(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = output_tensor_a2 + output_tensor

        output_tensor = self.c19(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c20(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c21(output_tensor)

        output_tensor_c21 = output_tensor
        if output_tensor_c21.is_sharded():
            output_tensor_c21 = ttnn.sharded_to_interleaved(output_tensor_c21, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c22(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c23(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c24(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = output_tensor_c21 + output_tensor

        output_tensor_a4 = output_tensor

        if output_tensor_a4.is_sharded():
            output_tensor_a4 = ttnn.sharded_to_interleaved(output_tensor_a4, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c25(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c26(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c27(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = output_tensor_a4 + output_tensor
        output_tensor_a5 = output_tensor
        if output_tensor_a5.is_sharded():
            output_tensor_a5 = ttnn.sharded_to_interleaved(output_tensor_a5, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c28(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c29(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c30(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.add(output_tensor_a5, output_tensor)

        output_tensor = self.c31(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c32(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c33(output_tensor)

        output_tensor_c33 = output_tensor
        if output_tensor_c33.is_sharded():
            output_tensor_c33 = ttnn.sharded_to_interleaved(output_tensor_c33, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c34(output_tensor_c33)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c35(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c36(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = output_tensor_c33 + output_tensor

        output_tensor_a7 = output_tensor

        if output_tensor_a7.is_sharded():
            output_tensor_a7 = ttnn.sharded_to_interleaved(output_tensor_a7, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c37(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c38(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c39(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = output_tensor_a7 + output_tensor

        output_tensor = self.c40(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c41(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c42(output_tensor)

        output_tensor_c42 = output_tensor
        if output_tensor_c42.is_sharded():
            output_tensor_c42 = ttnn.sharded_to_interleaved(output_tensor_c42, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c43(output_tensor_c42)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)

        output_tensor = self.c44(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c45(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = output_tensor_c42 + output_tensor
        output_tensor_a9 = output_tensor

        if output_tensor_a9.is_sharded():
            output_tensor_a9 = ttnn.sharded_to_interleaved(output_tensor_a9, ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c46(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = self.c47(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c48(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = output_tensor + output_tensor_a9

        output_tensor = self.c49(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = self.c50(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c51(output_tensor)

        output_tensor = self.c52(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 1280, 4, 4)

        x = nn.functional.adaptive_avg_pool2d(output_tensor, (1, 1))
        x = torch.flatten(x, 1)

        output_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)

        self.l1_weight = preprocess_linear_weight(self.l1_weight, dtype=ttnn.bfloat16)
        self.l1_bias = preprocess_linear_bias(self.l1_bias, dtype=ttnn.bfloat16)
        self.l1_weight = ttnn.to_device(self.l1_weight, device)
        self.l1_bias = ttnn.to_device(self.l1_bias, device)

        output_tensor = ttnn.linear(output_tensor, self.l1_weight, bias=self.l1_bias)

        return ttnn.from_device(output_tensor)
