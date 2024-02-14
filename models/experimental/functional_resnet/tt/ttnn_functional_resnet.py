# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class BasicBlock:
    def __init__(
        self,
        parameters,
    ) -> None:
        self.conv1 = parameters.conv1
        self.conv2 = parameters.conv2
        if "downsample" in parameters:
            self.downsample = parameters.downsample
        else:
            self.downsample = None

    def __call__(self, x):
        identity = x

        # Relu and bn1 are fused with conv1
        out = self.conv1(x)

        # Relu and bn2 are fused with conv1
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # out = ttnn.add(out, identity, memory_config=ttnn.get_memory_config(out))
        # out = ttnn.to_memory_config(out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # out = self.relu(out)

        return out
