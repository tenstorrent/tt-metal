# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import tt_lib
from models.experimental.functional_yolox_m.tt.ttnn_bottleneck_block import TtBottleneckBlock


class TtDark3:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.bblock = TtBottleneckBlock(parameters.bblock, 6, True)
        self.c4 = parameters.c4

    def __call__(self, device, input_tensor: ttnn.Tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)
        output_tensor = self.c1(input_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor_c1 = output_tensor
        output_tensor_c1 = output_tensor_c1.to(device, self.c3.conv.input_sharded_memory_config)
        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c2 = output_tensor

        output_tensor = self.c3(output_tensor_c1)
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c3 = output_tensor

        output_tensor = self.bblock(device, output_tensor_c2)
        output_tensor = output_tensor.to(device)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_c3 = tt_lib.tensor.sharded_to_interleaved(output_tensor_c3, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_c3], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = output_tensor.to(device, self.c4.conv.input_sharded_memory_config)
        output_tensor = self.c4(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        return ttnn.from_device(output_tensor)
