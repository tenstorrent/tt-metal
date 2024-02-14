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
        conv1 = self.conv1(x)

        # Relu and bn2 are fused with conv1
        conv2 = self.conv2(conv1)
        ttnn.deallocate(conv1)

        if self.downsample is not None:
            identity = self.downsample(x)
            ttnn.deallocate(x)

        out = ttnn.add_and_apply_activation(
            conv2, identity, activation="relu", memory_config=ttnn.get_memory_config(conv2)
        )
        ttnn.deallocate(conv2)
        if x != identity:
            ttnn.deallocate(identity)

        return out
