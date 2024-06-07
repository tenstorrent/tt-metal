# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
##### ttnneck

import torch
import torch.nn as nn

from ttnn.model_preprocessing import preprocess_model

import ttnn
import tt_lib
import tt_lib.fallback_ops


class TtNeck:
    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.device = device
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.c4 = parameters.c4
        self.c5 = parameters.c5
        self.c6 = parameters.c6
        self.c7 = parameters.c7
        self.c7_2 = parameters.c7_2
        self.c7_3 = parameters.c7_3
        self.c7_4 = parameters.c7_4
        self.c7_5 = parameters.c7_5
        self.c8 = parameters.c8
        self.c8_2 = parameters.c8_2
        self.c9 = parameters.c9
        self.c9_2 = parameters.c9_2
        self.c9_3 = parameters.c9_3
        self.c9_4 = parameters.c9_4
        self.c9_5 = parameters.c9_5
        self.c10 = parameters.c10
        self.c10_2 = parameters.c10_2
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)

        self.max_pool_reader_patterns_cache = {}
        max_pool_parallel_config_override = {}

        max_pool_parallel_config_override["grid_size"] = self.c3.conv.grid_size
        max_pool_parallel_config_override["num_cores_nhw"] = self.c3.conv.sliding_window_op_params.num_cores_nhw

        self.p1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        self.p2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        self.p3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)

    def __call__(self, device, input_tensors):
        input_tensor0 = input_tensors[0].to(device, self.c1.conv.input_sharded_memory_config)

        output_tensor = self.c1(input_tensor0)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c2.conv.input_sharded_memory_config)

        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c3.conv.input_sharded_memory_config)

        output_tensor = self.c3(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c7.conv.input_sharded_memory_config)

        output_tensorc3 = output_tensor

        output_tensorc3 = tt_lib.tensor.sharded_to_interleaved(output_tensorc3, ttnn.L1_MEMORY_CONFIG)
        custom_sharded_memory_config = ttnn.experimental.tensor.MemoryConfig(
            memory_layout=ttnn.experimental.tensor.TensorMemoryLayout.HEIGHT_SHARDED,
            buffer_type=ttnn.experimental.tensor.BufferType.L1,
        )
        output_tensorc3 = ttnn.to_layout(output_tensorc3, ttnn.ROW_MAJOR_LAYOUT)

        output_tensorc3 = ttnn.from_device(output_tensorc3)
        output_tensorc3 = ttnn.to_torch(output_tensorc3)
        output_tensorc3 = torch.reshape(output_tensorc3, (1, 10, 10, 512))
        output_tensorc3 = torch.permute(output_tensorc3, (0, 3, 1, 2))

        output_tensor = self.p1(output_tensorc3)
        output_tensorp1 = output_tensor
        output_tensor = self.p2(output_tensorc3)

        output_tensorp2 = output_tensor
        output_tensor = self.p3(output_tensorc3)

        output_tensorp3 = output_tensor
        output_tensorp1 = torch.reshape(output_tensorp1, (1, 512, 1, 100))
        output_tensorp2 = torch.reshape(output_tensorp2, (1, 512, 1, 100))
        output_tensorp3 = torch.reshape(output_tensorp3, (1, 512, 1, 100))
        output_tensorc3 = torch.reshape(output_tensorc3, (1, 512, 1, 100))
        output_tensorp1 = torch.permute(output_tensorp1, (0, 2, 3, 1))
        output_tensorp2 = torch.permute(output_tensorp2, (0, 2, 3, 1))
        output_tensorp3 = torch.permute(output_tensorp3, (0, 2, 3, 1))
        output_tensorc3 = torch.permute(output_tensorc3, (0, 2, 3, 1))

        output_tensorp1 = ttnn.from_torch(
            output_tensorp1,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensorp2 = ttnn.from_torch(
            output_tensorp2,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensorp3 = ttnn.from_torch(
            output_tensorp3,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensorc3 = ttnn.from_torch(
            output_tensorc3,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensorp1 = output_tensorp1.to(device)
        output_tensorp2 = output_tensorp2.to(device)
        output_tensorp3 = output_tensorp3.to(device)
        output_tensorc3 = output_tensorc3.to(device)
        output_tensor = ttnn.concat(
            [output_tensorp3, output_tensorp2, output_tensorp1, output_tensorc3],
            dim=3,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        output_tensor = output_tensor.to(device, self.c4.conv.input_sharded_memory_config)
        output_tensor = self.c4(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c5.conv.input_sharded_memory_config)

        output_tensor = self.c5(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c6.conv.input_sharded_memory_config)

        output_tensor = self.c6(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c7.conv.input_sharded_memory_config)

        output_tensor_9m = output_tensor
        output_tensor = self.c7(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, (1, 4, 1), memory_config=output_tensor.memory_config())

        outDownSample4 = input_tensors[1].to(device, self.c7_2.conv.input_sharded_memory_config)
        # CBR block for conc2
        outDownSample4_c7 = self.c7_2(outDownSample4)
        outDownSample4_c7 = ttnn.to_torch(outDownSample4_c7)
        outDownSample4_c7 = self.leaky_relu(outDownSample4_c7)
        outDownSample4_c7 = ttnn.from_torch(
            outDownSample4_c7,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        outDownSample4_c7 = ttnn.to_layout(outDownSample4_c7, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([outDownSample4_c7, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c7_3.conv.input_sharded_memory_config)
        output_tensor = self.c7_3(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c8.conv.input_sharded_memory_config)

        output_tensor = self.c8(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c7_4.conv.input_sharded_memory_config)

        output_tensor = self.c7_4(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c8_2.conv.input_sharded_memory_config)

        output_tensor = self.c8_2(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c7_5.conv.input_sharded_memory_config)

        output_tensor = self.c7_5(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9.conv.input_sharded_memory_config)

        output_tensor_16m = output_tensor
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = output_tensor.to(device, self.c9.conv.input_sharded_memory_config)

        output_tensor = self.c9(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9_2.conv.input_sharded_memory_config)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.upsample(output_tensor, (1, 4, 1), memory_config=output_tensor.memory_config())

        outDownSample3 = input_tensors[2].to(device, self.c9_2.conv.input_sharded_memory_config)
        outDownSample3_c9 = self.c9_2(outDownSample3)
        outDownSample3_c9 = ttnn.to_torch(outDownSample3_c9)
        outDownSample3_c9 = self.leaky_relu(outDownSample3_c9)
        outDownSample3_c9 = ttnn.from_torch(
            outDownSample3_c9,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([outDownSample3_c9, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = output_tensor.to(device, self.c9_3.conv.input_sharded_memory_config)
        output_tensor = self.c9_3(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c10.conv.input_sharded_memory_config)
        output_tensor = self.c10(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9_4.conv.input_sharded_memory_config)
        output_tensor = self.c9_4(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c10_2.conv.input_sharded_memory_config)
        output_tensor = self.c10_2(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )

        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9_5.conv.input_sharded_memory_config)
        output_tensor = self.c9_5(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = self.leaky_relu(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
        )
        return ttnn.from_device(output_tensor), ttnn.from_device(output_tensor_9m), ttnn.from_device(output_tensor_16m)
