# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch

from loguru import logger


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
            shard_layout=(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED
                if self.use_1d_systolic_array
                else ttnn.TensorMemoryLayout.BLOCK_SHARDED
            ),
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

        self.c1 = MobileNetV2Conv2D(parameters.c1, parameters.b1, device, activation="")
        self.c2 = MobileNetV2Conv2D(parameters.c2, parameters.b2, device, activation="")

        self.c3 = model.c3
        self.b3 = model.b3

        self.c4 = model.c4
        self.b4 = model.b4

        self.c5 = MobileNetV2Conv2D(parameters.c5, parameters.b5, device, activation="")

        self.c6 = model.c6
        self.b6 = model.b6

        self.c7 = model.c7
        self.b7 = model.b7

        self.c8 = MobileNetV2Conv2D(parameters.c8, parameters.b8, device, activation="")

        self.c9 = model.c9
        self.b9 = model.b9

        self.c10 = model.c10
        self.b10 = model.b10

        self.c11 = MobileNetV2Conv2D(parameters.c11, parameters.b11, device, activation="")

        self.c12 = model.c12
        self.b12 = model.b12

        self.c13 = MobileNetV2Conv2D(parameters.c13, parameters.b13, device, activation="")
        self.c14 = MobileNetV2Conv2D(parameters.c14, parameters.b14, device, activation="")
        self.c15 = MobileNetV2Conv2D(parameters.c15, parameters.b15, device)
        self.c16 = MobileNetV2Conv2D(parameters.c16, parameters.b16, device, activation="")
        self.c17 = MobileNetV2Conv2D(parameters.c17, parameters.b17, device, activation="")
        self.c18 = MobileNetV2Conv2D(parameters.c18, parameters.b18, device)
        self.c19 = MobileNetV2Conv2D(parameters.c19, parameters.b19, device, activation="")
        self.c20 = MobileNetV2Conv2D(parameters.c20, parameters.b20, device, activation="")
        self.c21 = MobileNetV2Conv2D(parameters.c21, parameters.b21, device)
        self.c22 = MobileNetV2Conv2D(parameters.c22, parameters.b22, device, activation="")
        self.c23 = MobileNetV2Conv2D(parameters.c23, parameters.b23, device, activation="")
        self.c24 = MobileNetV2Conv2D(parameters.c24, parameters.b24, device)
        self.c25 = MobileNetV2Conv2D(parameters.c25, parameters.b25, device, activation="")
        self.c26 = MobileNetV2Conv2D(parameters.c26, parameters.b26, device, activation="")
        self.c27 = MobileNetV2Conv2D(parameters.c27, parameters.b27, device)
        self.c28 = MobileNetV2Conv2D(parameters.c28, parameters.b28, device, activation="")
        self.c29 = MobileNetV2Conv2D(parameters.c29, parameters.b29, device, activation="")
        self.c30 = MobileNetV2Conv2D(parameters.c30, parameters.b30, device)
        self.c31 = MobileNetV2Conv2D(parameters.c31, parameters.b31, device, activation="")
        self.c32 = MobileNetV2Conv2D(parameters.c32, parameters.b32, device, activation="")
        self.c33 = MobileNetV2Conv2D(parameters.c33, parameters.b33, device)
        self.c34 = MobileNetV2Conv2D(parameters.c34, parameters.b34, device, activation="")

        self.c35 = model.c35
        self.b35 = model.b35

        self.c36 = MobileNetV2Conv2D(parameters.c36, parameters.b36, device)
        self.c37 = MobileNetV2Conv2D(parameters.c37, parameters.b37, device, activation="")

        self.c38 = model.c38
        self.b38 = model.b38

        self.c39 = MobileNetV2Conv2D(parameters.c39, parameters.b39, device)
        self.c40 = MobileNetV2Conv2D(parameters.c40, parameters.b40, device, activation="")

        self.c41 = model.c41
        self.b41 = model.b41

        self.c42 = MobileNetV2Conv2D(parameters.c42, parameters.b42, device)
        self.c43 = MobileNetV2Conv2D(parameters.c43, parameters.b43, device, activation="")

        self.c44 = model.c44
        self.b44 = model.b44

        self.c45 = MobileNetV2Conv2D(parameters.c45, parameters.b45, device)
        self.c46 = MobileNetV2Conv2D(parameters.c46, parameters.b46, device, activation="")

        self.c47 = model.c47
        self.b47 = model.b47

        self.c48 = MobileNetV2Conv2D(parameters.c48, parameters.b48, device)
        self.c49 = MobileNetV2Conv2D(parameters.c49, parameters.b49, device, activation="")

        self.c50 = model.c50
        self.b50 = model.b50

        self.c51 = MobileNetV2Conv2D(parameters.c51, parameters.b51, device)
        self.c52 = MobileNetV2Conv2D(parameters.c52, parameters.b52, device, activation="")

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

        output_tensor = self.input_preprocessor(output_tensor, 1, 32, 64, 64)

        output_tensor = self.c3(output_tensor)
        output_tensor = self.b3(output_tensor)

        output_tensor = self.c4(output_tensor)
        output_tensor = self.b4(output_tensor)

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c5(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 96, 32, 32)
        output_tensor = self.c6(output_tensor)
        output_tensor = self.b6(output_tensor)
        output_tensor_c6 = output_tensor

        output_tensor = self.c7(output_tensor)
        output_tensor = self.b7(output_tensor)

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c8(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 144, 32, 32)
        output_tensor = self.c9(output_tensor)
        output_tensor = self.b9(output_tensor)

        output_tensor = output_tensor_c6 + output_tensor

        output_tensor = self.c10(output_tensor)
        output_tensor = self.b10(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c11(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 144, 16, 16)
        output_tensor = self.c12(output_tensor)
        output_tensor = self.b12(output_tensor)

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor_c12 = output_tensor

        output_tensor = self.c13(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c14(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c15(output_tensor)

        output_tensor_c15 = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
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

        output_tensor_c21 = self.c21(output_tensor)
        output_tensor_21_torch = self.input_preprocessor(output_tensor_c21, 1, 64, 8, 8)

        output_tensor = self.c22(output_tensor_c21)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c23(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c24(output_tensor)
        output_tensor_24_torch = self.input_preprocessor(output_tensor, 1, 64, 8, 8)

        output_tensor = output_tensor_21_torch + output_tensor_24_torch
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
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

        output_tensor = output_tensor_a5 + output_tensor

        output_tensor = self.c31(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c32(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor_c33 = self.c33(output_tensor)
        output_tensor_33_torch = self.input_preprocessor(output_tensor_c33, 1, 96, 8, 8)

        output_tensor = self.c34(output_tensor_c33)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 576, 8, 8)
        output_tensor = self.c35(output_tensor)
        output_tensor = self.b35(output_tensor)

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c36(output_tensor)
        output_tensor_36_torch = self.input_preprocessor(output_tensor, 1, 96, 8, 8)

        output_tensor = output_tensor_33_torch + output_tensor_36_torch

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor_a7 = output_tensor

        output_tensor = self.c37(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 576, 8, 8)
        output_tensor = self.c38(output_tensor)
        output_tensor = self.b38(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c39(output_tensor)

        output_tensor = output_tensor_a7 + output_tensor

        output_tensor = self.c40(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 576, 8, 8)
        output_tensor = self.c41(output_tensor)
        output_tensor = self.b41(output_tensor)

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor_c42 = self.c42(output_tensor)
        output_tensor_c42_torch = self.input_preprocessor(output_tensor_c42, 1, 160, 4, 4)
        output_tensor = self.c43(output_tensor_c42)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 960, 4, 4)
        output_tensor = self.c44(output_tensor)
        output_tensor = self.b44(output_tensor)

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c45(output_tensor)

        output_tensor_c45_torch = self.input_preprocessor(output_tensor, 1, 160, 4, 4)
        output_tensor = output_tensor_c42_torch + output_tensor_c45_torch

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor_a9 = output_tensor

        output_tensor = self.c46(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 960, 4, 4)
        output_tensor = self.c47(output_tensor)
        output_tensor = self.b47(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.c48(output_tensor)
        output_tensor = output_tensor + output_tensor_a9

        output_tensor = self.c49(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)

        output_tensor = self.input_preprocessor(output_tensor, 1, 960, 4, 4)
        output_tensor = self.c50(output_tensor)
        output_tensor = self.b50(output_tensor)

        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = output_tensor.reshape(
            1,
            1,
            output_tensor.shape[0] * output_tensor.shape[1] * output_tensor.shape[2],
            output_tensor.shape[3],
        )
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
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
