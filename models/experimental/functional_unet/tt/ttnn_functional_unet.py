# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class UNet:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.c1 = parameters.c1
        self.c1_2 = parameters.c1_2
        self.p1 = parameters.p1

    def __call__(self, x):
        identity = x

        # Relu and bn1 are fused with conv1
        out = self.c1(x)

        # Relu and bn2 are fused with conv1
        out = self.c1_2(out)
        out = self.p1(out)

        # out = ttnn.add(out, identity, memory_config=ttnn.get_memory_config(out))
        # out = ttnn.to_memory_config(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # out = self.relu(out)

        return out

    def torch_call(self, torch_input_tensor):
        input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat16)

        input_tensor = self.c1.copy_input_to_device(input_tensor)
        output_tensor = self(input_tensor)
        output_tensor = self.p1.copy_output_from_device(output_tensor)
        # output_tensor = self.c1_2.copy_output_from_device(input_tensor)

        output_tensor = ttnn.to_torch(output_tensor)
        print("the shape before change is: ", output_tensor.size())
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        print("the shape after change is: ", output_tensor.size())
        print(" torch_input_tensor.shape: ", torch_input_tensor.shape)
        # output_tensor = torch.reshape(output_tensor, torch_input_tensor.shape)
        output_tensor = output_tensor.to(torch_input_tensor.dtype)
        return output_tensor
