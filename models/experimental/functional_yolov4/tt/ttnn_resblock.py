# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TtResBlock:
    def __init__(self, parameters, nblocks, shortcut) -> None:
        self.shortcut = shortcut
        self.nblocks = nblocks
        self.module_list = []
        for i in range(nblocks):
            conv1 = parameters[f"resblock_{i}_conv1"]
            conv2 = parameters[f"resblock_{i}_conv2"]
            resblock_one = [conv1, conv2]
            self.module_list.append(resblock_one)

    def __call__(self, device, input_tensor):
        input_tensor = input_tensor.to(device, self.module_list[0][0].conv.input_sharded_memory_config)
        for i in range(self.nblocks):
            output_tensor_h = input_tensor
            output_tensor_1 = self.module_list[i][0](output_tensor_h)
            output_tensor_h = self.module_list[i][1](output_tensor_1)
            input_tensor = ttnn.add(input_tensor, output_tensor_h) if self.shortcut else output_tensor_h
        return ttnn.from_device(input_tensor)
