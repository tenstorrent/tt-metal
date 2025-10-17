# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.experimental.detr3d.ttnn.common import TtnnConv2D


class TtnnSharedMLP(LightweightModule):
    def __init__(self, module, parameters, device):
        super().__init__()
        self.device = device
        self.parameters = parameters
        shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        self.conv1 = TtnnConv2D(
            module.layer0.conv,
            parameters.layer0.conv,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_dealloc_act=True,
            return_dims=True,
            shard_layout=shard_layout,
        )
        self.conv2 = TtnnConv2D(
            module.layer1.conv,
            parameters.layer1.conv,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_dealloc_act=True,
            return_dims=True,
            shard_layout=shard_layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.conv3 = TtnnConv2D(
            module.layer2.conv,
            parameters.layer2.conv,
            device,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            is_dealloc_act=True,
            return_dims=True,
            shard_layout=shard_layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, features):
        shape = features.shape
        conv1, shape = self.conv1(features, shape)
        conv2, shape = self.conv2(conv1, shape)
        conv3, shape = self.conv3(conv2, shape)
        conv3 = ttnn.reshape(conv3, shape)
        return conv3
