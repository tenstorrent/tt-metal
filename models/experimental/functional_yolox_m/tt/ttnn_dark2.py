# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import tt_lib
from models.experimental.functional_yolox_m.tt.ttnn_bottleneck_block import TtBottleneckBlock


class TtFocus:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1

    def __call__(self, device, input_tensor: ttnn.Tensor):
        input_tensor0 = input_tensor[0].to(device)
        input_tensor1 = input_tensor[1].to(device)
        input_tensor2 = input_tensor[2].to(device)
        input_tensor3 = input_tensor[3].to(device)

        input_tensor0 = ttnn.to_torch(input_tensor0)
        input_tensor1 = ttnn.to_torch(input_tensor1)
        input_tensor2 = ttnn.to_torch(input_tensor2)
        input_tensor3 = ttnn.to_torch(input_tensor3)
        output_tensor = torch.concat(
            (
                input_tensor0,
                input_tensor2,
                input_tensor1,
                input_tensor3,
            ),
            dim=3,
        )
        output_tensor = ttnn.from_torch(output_tensor, device=device, layout=ttnn.TILE_LAYOUT)

        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c1.conv.input_sharded_memory_config)
        output_tensor = self.c1(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        return ttnn.from_device(output_tensor)


class TtDark2:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.bblock = TtBottleneckBlock(parameters.bblock, 2, True)
        self.c4 = parameters.c4

    def __call__(self, device, input_tensor: ttnn.Tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        output_tensor = self.c1(input_tensor)

        output_tensor = ttnn.silu(output_tensor)

        output_tensor_c1 = output_tensor
        output_tensor_c1 = output_tensor_c1.to(device, self.c3.conv.input_sharded_memory_config)
        output_tensor = output_tensor.to(device, self.c2.conv.input_sharded_memory_config)
        output_tensor = self.c2(output_tensor)
        output_tensor = self.c2.copy_output_from_device(output_tensor)

        output_tensor = output_tensor.reshape(
            output_tensor.shape[0], 1, output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]
        )

        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

        output_tensor = output_tensor.to(device, self.c2.conv.output_sharded_memory_config)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor_c2 = output_tensor

        output_tensor = self.c3(output_tensor_c1)
        output_tensor = self.c3.copy_output_from_device(output_tensor)
        output_tensor = output_tensor.reshape(
            output_tensor.shape[0], 1, output_tensor.shape[1] * output_tensor.shape[2], output_tensor.shape[3]
        )
        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor = output_tensor.to(device, self.c3.conv.output_sharded_memory_config)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor_c3 = output_tensor

        output_tensor = self.bblock(device, output_tensor_c2)

        output_tensor = output_tensor.to(device)

        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        output_tensor_c3 = ttnn.to_torch(output_tensor_c3)
        output_tensor_c3 = ttnn.from_torch(
            output_tensor_c3, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT
        )

        output_tensor = ttnn.concat([output_tensor, output_tensor_c3], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.TILE_LAYOUT
        )
        output_tensor = output_tensor.to(
            device,
            self.c4.conv.input_sharded_memory_config,
        )

        output_tensor = self.c4(output_tensor)
        output_tensor = output_tensor.to(device, self.c4.conv.output_sharded_memory_config)
        output_tensor = ttnn.silu(output_tensor)
        return ttnn.from_device(output_tensor)
