# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.ufld_v2.ttnn.common import TtnnUFLDV2Conv2D


class TtnnBasicBlock:
    def __init__(self, conv_args, conv_pth, device, is_downsample=False, blk_sharded=False):
        self.is_downsample = is_downsample

        self.conv1 = TtnnUFLDV2Conv2D(
            conv_args.conv1, conv_pth.conv1, device=device, activation="relu", is_blk=blk_sharded
        )
        self.conv2 = TtnnUFLDV2Conv2D(conv_args.conv2, conv_pth.conv2, device=device, activation="", is_blk=blk_sharded)
        if is_downsample:
            self.downsample = TtnnUFLDV2Conv2D(
                conv_args.downsample[0], conv_pth.downsample, device=device, activation="", is_blk=blk_sharded
            )

    def __call__(self, input):
        x_identity = input
        x, out_ht, out_wdth = self.conv1(input)
        x, out_ht, out_wdth = self.conv2(x)
        if self.is_downsample:
            x_identity, out_ht, out_wdth = self.downsample(input)
        if x_identity.is_sharded():
            if x.memory_config().shard_spec.grid != x_identity.memory_config().shard_spec.grid:
                memory_config_req = x.memory_config()
                memory_config_req.shard_spec.shape = x_identity.memory_config().shard_spec.shape
                memory_config_req.shard_spec.grid = x_identity.memory_config().shard_spec.grid
                x = ttnn.reshard(x, memory_config_req)
        x = ttnn.add(x, x_identity, memory_config=x.memory_config())
        x = ttnn.relu(x)
        ttnn.deallocate(x_identity)
        return x
