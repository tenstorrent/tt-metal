# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
import ttnn
import tt_lib
import tt_lib.fallback_ops
from models.experimental.functional_yolox_m.tt.ttnn_bottleneck_block import TtBottleneckBlock


class TtDark5:
    def output_preprocessing(self, output_tensor):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        return output_tensor

    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.c4 = parameters.c4
        self.c4 = tt_lib.fallback_ops.Conv2d(
            parameters.c4["weight"], parameters.c4["bias"], 768, 384, 1, 1, 0, bias=True
        )
        # self.c5 = parameters.c5
        self.c5 = tt_lib.fallback_ops.Conv2d(
            parameters.c5["weight"], parameters.c5["bias"], 768, 384, 1, 1, 0, bias=True
        )
        self.bblock = TtBottleneckBlock(parameters.bblock, 2, False)
        self.c6 = parameters.c6

        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}

        max_pool_parallel_config_override["grid_size"] = self.c2.conv.grid_size
        max_pool_parallel_config_override["num_cores_nhw"] = self.c2.conv.sliding_window_op_params.num_cores_nhw

        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.p3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)

    def __call__(self, device, input_tensor: ttnn.Tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        output_tensor = self.c1(input_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor_c2 = output_tensor

        output_tensor_c2 = ttnn.from_device(output_tensor_c2)
        output_tensor_c2 = ttnn.to_torch(output_tensor_c2)
        output_tensor_c2 = torch.reshape(output_tensor_c2, (1, 20, 20, 384))
        output_tensor_c2 = torch.permute(output_tensor_c2, (0, 3, 1, 2))

        output_tensor_p1 = self.p1(output_tensor_c2)
        output_tensor_p2 = self.p2(output_tensor_c2)
        output_tensor_p3 = self.p3(output_tensor_c2)

        output_tensor_p1 = torch.reshape(output_tensor_p1, (1, 384, 1, 400))
        output_tensor_p1 = torch.permute(output_tensor_p1, (0, 2, 3, 1))
        output_tensor_p1 = ttnn.from_torch(output_tensor_p1, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor_p1 = output_tensor_p1.to(device, ttnn.L1_MEMORY_CONFIG)

        output_tensor_p2 = torch.reshape(output_tensor_p2, (1, 384, 1, 400))
        output_tensor_p2 = torch.permute(output_tensor_p2, (0, 2, 3, 1))
        output_tensor_p2 = ttnn.from_torch(output_tensor_p2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor_p2 = output_tensor_p2.to(device, ttnn.L1_MEMORY_CONFIG)

        output_tensor_p3 = torch.reshape(output_tensor_p3, (1, 384, 1, 400))
        output_tensor_p3 = torch.permute(output_tensor_p3, (0, 2, 3, 1))
        output_tensor_p3 = ttnn.from_torch(output_tensor_p3, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor_p3 = output_tensor_p3.to(device, ttnn.L1_MEMORY_CONFIG)

        output_tensor_c2 = torch.reshape(output_tensor_c2, (1, 384, 1, 400))
        output_tensor_c2 = torch.permute(output_tensor_c2, (0, 2, 3, 1))
        output_tensor_c2 = ttnn.from_torch(output_tensor_c2, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        output_tensor_c2 = output_tensor_c2.to(device, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat(
            [output_tensor_c2] + [output_tensor_p1, output_tensor_p2, output_tensor_p3],
            dim=3,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output_tensor = self.c3(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c3 = output_tensor

        output_tensor = self.output_preprocessing(output_tensor)
        output_tensor = self.c4(output_tensor)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.silu(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(
            output_tensor, self.bblock.module_list[0][0].conv.input_sharded_memory_config
        )
        output_tensor_c4 = output_tensor

        output_tensor_c3 = self.output_preprocessing(output_tensor_c3)
        output_tensor = self.c5(output_tensor_c3)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.permute(output_tensor, (0, 2, 3, 1))
        output_tensor = ttnn.silu(output_tensor)
        output_tensor = output_tensor.to(device, ttnn.L1_MEMORY_CONFIG)
        output_tensor_c5 = output_tensor

        output_tensor = self.bblock(device, output_tensor_c4)
        output_tensor = output_tensor.to(device, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        output_tensor_c5 = ttnn.to_torch(output_tensor_c5)
        output_tensor_c5 = ttnn.from_torch(
            output_tensor_c5, device=device, memory_config=ttnn.L1_MEMORY_CONFIG, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        output_tensor = ttnn.concat([output_tensor, output_tensor_c5], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = output_tensor.to(device, self.c6.conv.input_sharded_memory_config)
        output_tensor = self.c6(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        return ttnn.from_device(output_tensor)
