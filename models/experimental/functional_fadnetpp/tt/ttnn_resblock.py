# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
import tt_lib


class TtResBlock:
    def __init__(self, parameters, n_in, n_out, stride=1) -> None:
        self.conv1 = parameters["resblock_1_conv1"]
        self.conv2 = parameters["resblock_2_conv2"]
        self.sc = False
        if stride != 1 or n_out != n_in:
            self.sc = True
            self.shortcut = parameters["resblock_sc_conv"]

    def __call__(self, device, input_tensor):
        if self.sc:
            input_tensor = input_tensor.to(device)
            input_tensor = tt_lib.tensor.interleaved_to_sharded(
                input_tensor, self.shortcut.conv.input_sharded_memory_config
            )
            residual = input_tensor
            residual = self.shortcut(input_tensor)
        else:
            input_tensor = input_tensor.to(device, self.conv1.conv.input_sharded_memory_config)
            residual = input_tensor
        output_tensor_h = input_tensor
        output_tensor_1 = self.conv1(output_tensor_h)
        output_tensor_1 = ttnn.relu(output_tensor_1)

        if output_tensor_1.shape[3] > 256:
            memory_config = tt_lib.tensor.MemoryConfig(
                tt_lib.tensor.TensorMemoryLayout.INTERLEAVED,
                tt_lib.tensor.BufferType.L1,
            )
            output_tensor_1 = tt_lib.tensor.sharded_to_interleaved(output_tensor_1, memory_config)
            residual = tt_lib.tensor.sharded_to_interleaved(residual, memory_config)
            output_tensor_1 = tt_lib.tensor.interleaved_to_sharded(
                output_tensor_1, self.conv2.conv.input_sharded_memory_config
            )
            residual = tt_lib.tensor.interleaved_to_sharded(residual, self.conv2.conv.input_sharded_memory_config)

        else:
            output_tensor_1 = output_tensor_1.to(device, self.conv2.conv.input_sharded_memory_config)

        output_tensor_h = self.conv2(output_tensor_1)

        output_tensor_h += residual
        output_tensor_h = ttnn.relu(output_tensor_h)
        return ttnn.from_device(output_tensor_h)
