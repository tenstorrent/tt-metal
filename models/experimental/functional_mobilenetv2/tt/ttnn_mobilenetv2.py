# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import math
from tt_lib.fallback_ops import fallback_ops
import ttnn
import tt_lib
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)


class TtMobilenetv2:
    def output_preprocessing(self, output_tensor, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = torch.reshape(
            output_tensor,
            [
                output_tensor.shape[0],
                output_tensor.shape[1],
                int(math.sqrt(output_tensor.shape[3])),
                int(math.sqrt(output_tensor.shape[3])),
            ],
        )
        output_tensor = torch_to_tt_tensor_rm(output_tensor, device, put_on_device=True)
        return output_tensor

    def input_preprocessing(self, input_tensor, device):
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        input_tensor = ttnn.permute(input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.reshape(
            input_tensor,
            (input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]),
        )
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
        return input_tensor

    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c2 = fallback_ops.Conv2d(
            parameters.c2["weight"], parameters.c2["bias"], 32, 32, 3, 1, 1, groups=32, bias=False
        )
        self.c3 = fallback_ops.Conv2d(parameters.c3["weight"], parameters.c3["bias"], 32, 16, 1, 1, bias=False)
        self.c4 = parameters.c4
        self.c5 = fallback_ops.Conv2d(
            parameters.c5["weight"], parameters.c5["bias"], 96, 96, 3, 2, 1, groups=96, bias=False
        )
        self.c6 = fallback_ops.Conv2d(parameters.c6["weight"], parameters.c6["bias"], 96, 24, 1, 1, bias=False)
        self.c7 = fallback_ops.Conv2d(parameters.c7["weight"], parameters.c7["bias"], 24, 144, 1, 1, bias=False)
        self.c8 = tt_lib.fallback_ops.Conv2d(
            parameters.c8["weight"], parameters.c8["bias"], 144, 144, 3, 1, 1, groups=144, bias=False
        )
        self.c9 = fallback_ops.Conv2d(parameters.c9["weight"], parameters.c9["bias"], 144, 24, 1, 1, bias=False)
        self.c10 = fallback_ops.Conv2d(parameters.c10["weight"], parameters.c10["bias"], 24, 144, 1, 1, bias=False)
        self.c11 = tt_lib.fallback_ops.Conv2d(
            parameters.c11["weight"], parameters.c11["bias"], 144, 144, 3, 2, 1, groups=144, bias=False
        )
        self.c12 = parameters.c12
        self.c13 = parameters.c13
        self.c14 = tt_lib.fallback_ops.Conv2d(
            parameters.c14["weight"], parameters.c14["bias"], 192, 192, 3, 1, 1, groups=192, bias=False
        )
        self.c15 = parameters.c15
        self.c16 = parameters.c16
        self.c17 = tt_lib.fallback_ops.Conv2d(
            parameters.c17["weight"], parameters.c17["bias"], 192, 192, 3, 1, 1, groups=192, bias=False
        )
        self.c18 = parameters.c18
        self.c19 = parameters.c19
        self.c20 = tt_lib.fallback_ops.Conv2d(
            parameters.c20["weight"], parameters.c20["bias"], 192, 192, 3, 2, 1, groups=192, bias=False
        )
        self.c21 = parameters.c21
        self.c22 = parameters.c22
        self.c23 = tt_lib.fallback_ops.Conv2d(
            parameters.c23["weight"], parameters.c23["bias"], 384, 384, 3, 1, 1, groups=384, bias=False
        )
        self.c24 = parameters.c24
        self.c25 = parameters.c25
        self.c26 = tt_lib.fallback_ops.Conv2d(
            parameters.c26["weight"], parameters.c26["bias"], 384, 384, 3, 1, 1, groups=384, bias=False
        )
        self.c27 = parameters.c27
        self.c28 = parameters.c28
        self.c29 = tt_lib.fallback_ops.Conv2d(
            parameters.c29["weight"], parameters.c29["bias"], 384, 384, 3, 1, 1, groups=384, bias=False
        )
        self.c30 = parameters.c30
        self.c31 = parameters.c31
        self.c32 = tt_lib.fallback_ops.Conv2d(
            parameters.c32["weight"], parameters.c32["bias"], 384, 384, 3, 1, 1, groups=384, bias=False
        )
        self.c33 = parameters.c33
        self.c34 = parameters.c34
        self.c35 = tt_lib.fallback_ops.Conv2d(
            parameters.c35["weight"], parameters.c35["bias"], 576, 576, 3, 1, 1, groups=576, bias=False
        )
        self.c36 = parameters.c36
        self.c37 = parameters.c37
        self.c38 = tt_lib.fallback_ops.Conv2d(
            parameters.c38["weight"], parameters.c38["bias"], 576, 576, 3, 1, 1, groups=576, bias=False
        )
        self.c39 = parameters.c39
        self.c40 = parameters.c40
        self.c41 = tt_lib.fallback_ops.Conv2d(
            parameters.c41["weight"], parameters.c41["bias"], 576, 576, 3, 2, 1, groups=576, bias=False
        )
        self.c42 = parameters.c42
        self.c43 = parameters.c43
        self.c44 = tt_lib.fallback_ops.Conv2d(
            parameters.c44["weight"], parameters.c44["bias"], 960, 960, 3, 1, 1, groups=960, bias=False
        )
        self.c45 = parameters.c45
        self.c46 = parameters.c46
        self.c47 = tt_lib.fallback_ops.Conv2d(
            parameters.c47["weight"], parameters.c47["bias"], 960, 960, 3, 1, 1, groups=960, bias=False
        )
        self.c48 = parameters.c48
        self.c49 = parameters.c49

        self.c50 = tt_lib.fallback_ops.Conv2d(
            parameters.c50["weight"], parameters.c50["bias"], 960, 960, 3, 1, 1, groups=960, bias=False
        )
        self.c51 = parameters.c51
        self.c52 = parameters.c52
        # self.l1_bias = parameters.l1.bias
        self.l1 = nn.Linear(in_features=1280, out_features=1000)

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)
        output_tensor = self.c1(input_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c3(output_tensor)

        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = self.c4(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c5(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c6(output_tensor)
        output_tensor_c6 = self.input_preprocessing(output_tensor, device)

        output_tensor = self.c7(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)
        output_tensor = self.output_preprocessing(output_tensor, device)

        output_tensor = self.c8(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)
        output_tensor = self.output_preprocessing(output_tensor, device)

        output_tensor = self.c9(output_tensor)

        output_tensor_c9 = self.input_preprocessing(output_tensor, device)
        output_tensor = output_tensor_c6 + output_tensor_c9

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c10(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)
        output_tensor = self.output_preprocessing(output_tensor, device)

        output_tensor = self.c11(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c12(output_tensor)
        output_tensor_c12 = output_tensor

        output_tensor = self.c13(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c14(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c15(output_tensor)

        output_tensor = output_tensor + output_tensor_c12
        output_tensor_a2 = output_tensor

        output_tensor = self.c16(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c17(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c18(output_tensor)

        output_tensor = output_tensor_a2 + output_tensor

        output_tensor = self.c19(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c20(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor_c21 = self.c21(output_tensor)

        output_tensor = self.c22(output_tensor_c21)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c23(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c24(output_tensor)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(
            output_tensor,
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            ),
        )

        output_tensor = output_tensor_c21 + output_tensor
        output_tensor_a4 = output_tensor

        output_tensor = self.c25(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c26(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c27(output_tensor)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(
            output_tensor,
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            ),
        )
        output_tensor = output_tensor_a4 + output_tensor
        output_tensor_a5 = output_tensor

        output_tensor = self.c28(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c29(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c30(output_tensor)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(
            output_tensor,
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            ),
        )

        output_tensor = output_tensor_a5 + output_tensor

        output_tensor = self.c31(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c32(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor_c33 = self.c33(output_tensor)

        output_tensor_c33 = tt_lib.tensor.sharded_to_interleaved(
            output_tensor_c33,
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            ),
        )
        output_tensor = self.c34(output_tensor_c33)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c35(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c36(output_tensor)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(
            output_tensor,
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            ),
        )
        output_tensor = output_tensor_c33 + output_tensor
        output_tensor_a7 = output_tensor

        output_tensor = self.c37(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c38(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c39(output_tensor)
        output_tensor_39 = output_tensor

        output_tensor = tt_lib.tensor.sharded_to_interleaved(
            output_tensor_39,
            output_mem_config=tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.L1
            ),
        )

        output_tensor = output_tensor_a7 + output_tensor

        output_tensor = self.c40(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c41(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor_c42 = self.c42(output_tensor)
        output_tensor = self.c43(output_tensor_c42)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c44(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c45(output_tensor)
        output_tensor = output_tensor_c42 + output_tensor
        output_tensor_a9 = output_tensor
        output_tensor = self.c46(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c47(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c48(output_tensor)
        output_tensor = output_tensor + output_tensor_a9

        output_tensor = self.c49(output_tensor)

        output_tensor = self.output_preprocessing(output_tensor, device)
        output_tensor = self.c50(output_tensor)
        output_tensor = self.input_preprocessing(output_tensor, device)
        output_tensor = ttnn.relu(output_tensor)

        output_tensor = self.c51(output_tensor)

        output_tensor = self.c52(output_tensor)
        output_tensor_t = output_tensor

        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = torch.reshape(
            output_tensor,
            [
                output_tensor.shape[0],
                output_tensor.shape[1],
                int(math.sqrt(output_tensor.shape[3])),
                int(math.sqrt(output_tensor.shape[3])),
            ],
        )
        output_tensor = nn.functional.adaptive_avg_pool2d(output_tensor, (1, 1))
        output_tensor = torch.flatten(output_tensor, 1)

        # output_tensor = self.l1(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

        return output_tensor
