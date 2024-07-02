# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import tt_lib


class TtBottleneckBlock:
    def __init__(self, parameters, nblocks, shortcut) -> None:
        self.shortcut = shortcut
        self.nblocks = nblocks
        self.module_list = []
        keys = list(parameters.keys())
        for i in range(0, len(keys), 2):
            conv1 = parameters[keys[i]]
            conv2 = parameters[keys[i + 1]]
            resblock_one = [conv1, conv2]
            self.module_list.append(resblock_one)

    def __call__(self, device, input_tensor: ttnn.Tensor):
        input_tensor = tt_lib.tensor.sharded_to_interleaved(input_tensor, ttnn.L1_MEMORY_CONFIG)
        input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
        for i in range(self.nblocks):
            output_tensor_h = input_tensor
            output_tensor_h = output_tensor_h.to(device, self.module_list[i][0].conv.input_sharded_memory_config)
            output_tensor_1 = self.module_list[i][0](output_tensor_h)
            output_tensor_1 = ttnn.silu(output_tensor_1)
            output_tensor_h = self.module_list[i][1](output_tensor_1)
            output_tensor_h = ttnn.silu(output_tensor_h)

            output_tensor_h = ttnn.to_layout(output_tensor_h, layout=ttnn.TILE_LAYOUT)
            input_tensor = (input_tensor + output_tensor_h) if self.shortcut else output_tensor_h
        return ttnn.from_device(input_tensor)
