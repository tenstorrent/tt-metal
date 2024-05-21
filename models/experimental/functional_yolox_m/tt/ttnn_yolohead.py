# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import ttnn
import tt_lib
from typing import List


class TtYOLOXHead:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.c4 = parameters.c4

        self.c5 = parameters.c5
        self.c6 = parameters.c6
        self.c7 = parameters.c7
        self.c8 = parameters.c8

        self.c9 = parameters.c9
        self.c10 = parameters.c10
        self.c11 = parameters.c11
        self.c12 = parameters.c12

        self.c13 = parameters.c13
        self.c14 = parameters.c14
        self.c15 = parameters.c15

        self.c16 = parameters.c16
        self.c17 = parameters.c17
        self.c18 = parameters.c18

        self.c19 = parameters.c19
        self.c20 = parameters.c20

        self.c21 = parameters.c21
        self.c22 = parameters.c22
        self.c23 = parameters.c23
        self.c24 = parameters.c24

    def __call__(self, device, input_tensor: List[ttnn.Tensor]):
        outputs = []
        # output 1 ops
        input_tensor0 = input_tensor[0].to(device, self.c22.conv.input_sharded_memory_config)
        output_tensor = self.c22(input_tensor0)
        output_tensor = ttnn.silu(output_tensor)
        reg_x = output_tensor
        reg_x = reg_x.to(device, self.c7.conv.input_sharded_memory_config)

        output_tensor = self.c1(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = self.c13(output_tensor)
        output_tensor = self.c13.copy_output_from_device(output_tensor)
        output_tensor = output_tensor.reshape(
            output_tensor.shape[0], 1, output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]
        )
        output_tensor = output_tensor.to(device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.sigmoid(output_tensor)
        cls_output = output_tensor

        output_tensor = self.c7(reg_x)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = self.c8(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        reg_feat = output_tensor
        reg_feat = reg_feat.to(device, self.c19.conv.input_sharded_memory_config)

        reg_output = self.c16(output_tensor)
        reg_output = self.c16.copy_output_from_device(reg_output)
        reg_output = reg_output.reshape(
            reg_output.shape[0], 1, reg_output.shape[1] * reg_output.shape[2], reg_output.shape[3]
        )

        output_tensor = self.c19(reg_feat)
        output_tensor = self.c19.copy_output_from_device(output_tensor)
        output_tensor = output_tensor.reshape(
            output_tensor.shape[0], 1, output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]
        )
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        output_tensor = output_tensor.to(device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.sigmoid(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        output_tensor = ttnn.to_torch(output_tensor)
        reg_output = ttnn.to_torch(reg_output)
        cls_output = ttnn.to_torch(cls_output)

        output_tensor = torch.concat([reg_output, output_tensor, cls_output], dim=3)
        output_tensor = ttnn.from_torch(output_tensor)
        outputs.append(output_tensor)

        input_tensor1 = input_tensor[1].to(device, self.c23.conv.input_sharded_memory_config)
        output_tensor = self.c23(input_tensor1)
        output_tensor = ttnn.silu(output_tensor)
        reg_x = output_tensor
        reg_x = reg_x.to(device, self.c9.conv.input_sharded_memory_config)

        output_tensor = self.c3(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = self.c4(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c14.conv.input_sharded_memory_config)
        output_tensor = self.c14(output_tensor)
        output_tensor = self.c14.copy_output_from_device(output_tensor)
        output_tensor = output_tensor.reshape(
            output_tensor.shape[0], 1, output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]
        )
        output_tensor = output_tensor.to(device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.sigmoid(output_tensor)
        cls_output = output_tensor

        output_tensor = self.c9(reg_x)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = self.c10(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        reg_feat = output_tensor
        reg_feat = reg_feat.to(device, self.c20.conv.input_sharded_memory_config)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c17.conv.input_sharded_memory_config)
        reg_output = self.c17(output_tensor)
        reg_output = self.c17.copy_output_from_device(reg_output)
        reg_output = reg_output.reshape(
            reg_output.shape[0], 1, reg_output.shape[1] * reg_output.shape[2], reg_output.shape[3]
        )

        reg_feat = tt_lib.tensor.sharded_to_interleaved(reg_feat, ttnn.L1_MEMORY_CONFIG)
        reg_feat = tt_lib.tensor.interleaved_to_sharded(reg_feat, self.c20.conv.input_sharded_memory_config)
        output_tensor = self.c20(reg_feat)
        output_tensor = self.c20.copy_output_from_device(output_tensor)
        output_tensor = output_tensor.reshape(
            output_tensor.shape[0], 1, output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]
        )
        tt_lib.device.DumpDeviceProfiler(device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        output_tensor = output_tensor.to(device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.sigmoid(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        output_tensor = ttnn.to_torch(output_tensor)
        reg_output = ttnn.to_torch(reg_output)
        cls_output = ttnn.to_torch(cls_output)
        output_tensor = torch.concat([reg_output, output_tensor, cls_output], dim=3)
        output_tensor = ttnn.from_torch(output_tensor)
        outputs.append(output_tensor)

        input_tensor2 = input_tensor[2].to(device, self.c24.conv.input_sharded_memory_config)
        output_tensor = self.c24(input_tensor2)
        output_tensor = ttnn.silu(output_tensor)
        reg_x = output_tensor
        reg_x = reg_x.to(device, self.c11.conv.input_sharded_memory_config)

        output_tensor = self.c5(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = self.c6(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c15.conv.input_sharded_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = self.c15(output_tensor)
        output_tensor = self.c15.copy_output_from_device(output_tensor)
        output_tensor = output_tensor.reshape(
            output_tensor.shape[0], 1, output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]
        )
        output_tensor = output_tensor.to(device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.sigmoid(output_tensor)
        cls_output = output_tensor

        output_tensor = self.c11(reg_x)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = self.c12(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        reg_feat = output_tensor
        reg_feat = reg_feat.to(device, self.c20.conv.input_sharded_memory_config)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c18.conv.input_sharded_memory_config)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, device=device, layout=ttnn.TILE_LAYOUT)
        reg_output = self.c18(output_tensor)
        reg_output = self.c18.copy_output_from_device(reg_output)
        reg_output = reg_output.reshape(
            reg_output.shape[0], 1, reg_output.shape[1] * reg_output.shape[2], reg_output.shape[3]
        )

        reg_feat = tt_lib.tensor.sharded_to_interleaved(reg_feat, ttnn.L1_MEMORY_CONFIG)
        reg_feat = tt_lib.tensor.interleaved_to_sharded(reg_feat, self.c21.conv.input_sharded_memory_config)
        reg_feat = ttnn.to_torch(reg_feat)
        reg_feat = ttnn.from_torch(reg_feat, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = self.c21(reg_feat)
        output_tensor = self.c21.copy_output_from_device(output_tensor)
        output_tensor = output_tensor.reshape(
            output_tensor.shape[0], 1, output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]
        )
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b)
        output_tensor = output_tensor.to(device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.sigmoid(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_memory_config(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)

        output_tensor = ttnn.to_torch(output_tensor)
        reg_output = ttnn.to_torch(reg_output)
        cls_output = ttnn.to_torch(cls_output)
        output_tensor = torch.concat([reg_output, output_tensor, cls_output], dim=3)
        output_tensor = ttnn.from_torch(output_tensor)
        outputs.append(output_tensor)

        return outputs
