# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.vadv2.tt.common import TtnnConv2D


class TtnnBottleneck:
    def __init__(
        self,
        conv_args,
        conv_pth,
        device,
        is_downsample=False,
        blk_sharded=False,
        activation_dtype=ttnn.bfloat16,
        conv3_blk_sharded=False,
    ):
        self.is_downsample = is_downsample
        self.activation_dtype = activation_dtype

        self.conv1 = TtnnConv2D(conv_args.conv1, conv_pth.conv1, device=device, activation="relu")
        self.conv2 = TtnnConv2D(conv_args.conv2, conv_pth.conv2, device=device, activation="relu", act_block_h=32)
        self.conv3 = TtnnConv2D(conv_args.conv3, conv_pth.conv3, device=device, activation="", is_blk=conv3_blk_sharded)

        if is_downsample:
            self.downsample = TtnnConv2D(
                conv_args.downsample[0],
                conv_pth.downsample,
                device=device,
                activation="",
                is_blk=blk_sharded,
                activation_dtype=activation_dtype,
            )

    def __call__(self, x_identity):
        x, out_ht, out_wdth = self.conv1(x_identity)
        if self.activation_dtype == ttnn.bfloat8_b:
            x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.bfloat8_b)
            x_identity = ttnn.add(x_identity, 0.0, dtype=ttnn.bfloat8_b)

        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x, out_ht, out_wdth = self.conv2(x)
        x, out_ht, out_wdth = self.conv3(x)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)

        if self.is_downsample:
            x_identity, _, _ = self.downsample(x_identity)
        x_identity = ttnn.to_memory_config(x_identity, ttnn.DRAM_MEMORY_CONFIG)

        x = ttnn.add(x, x_identity)
        x = ttnn.relu(x)

        ttnn.deallocate(x_identity)
        return x
