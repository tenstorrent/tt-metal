# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import tt_lib


class TtDownSample1:
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

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.c1.conv.input_sharded_memory_config)

        output_tensor = self.c1(input_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c1.conv.input_sharded_memory_config)

        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c3.conv.input_sharded_memory_config)

        output_tensor_c2 = output_tensor
        output_tensor = self.c3(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c3.conv.input_sharded_memory_config)

        output_tensor_c3 = output_tensor
        output_tensor_c3 = tt_lib.tensor.sharded_to_interleaved(output_tensor_c3, ttnn.L1_MEMORY_CONFIG)

        output_tensor_c2 = ttnn.to_torch(output_tensor_c2)
        output_tensor_c2 = ttnn.from_torch(
            output_tensor_c2, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT
        )
        output_tensor = self.c4(output_tensor_c2)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c3.conv.input_sharded_memory_config)

        output_tensor_c4 = output_tensor
        output_tensor = self.c5(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c6.conv.input_sharded_memory_config)

        output_tensor = self.c6(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c7.conv.input_sharded_memory_config)

        output_tensor = output_tensor + output_tensor_c4
        output_tensor = self.c7(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.concat([output_tensor, output_tensor_c3], dim=3)

        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c8.conv.input_sharded_memory_config)
        output_tensor = self.c8(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        output_tensor = ttnn.mish(output_tensor)
        return ttnn.from_device(output_tensor)
