# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math


from models.experimental.yolov11.tt.common import (
    determine_num_cores_for_upsample,
    get_core_grid_from_num_cores,
    deallocate_tensors,
    sharded_concat,
    Conv,
)
from models.experimental.yolov11.tt.ttnn_sppf import SPPF
from models.experimental.yolov11.tt.ttnn_c3k2 import C3k2
from models.experimental.yolov11.tt.ttnn_c2psa import C2PSA
from models.experimental.yolov11.tt.ttnn_detect import Detect


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class YoloV11:
    def __init__(self, device, parameters):
        self.device = device

        self.conv1 = Conv(device, parameters.conv_args[0], parameters.model[0])
        self.conv2 = Conv(device, parameters.conv_args[1], parameters.model[1])
        self.c3k2_1 = C3k2(device, parameters.conv_args[2], parameters.model[2], is_bk_enabled=True)
        self.conv3 = Conv(device, parameters.conv_args[3], parameters.model[3])
        self.c3k2_2 = C3k2(device, parameters.conv_args[4], parameters.model[4], is_bk_enabled=True)
        self.conv5 = Conv(device, parameters.conv_args[5], parameters.model[5])
        self.c3k2_3 = C3k2(device, parameters.conv_args[6], parameters.model[6], is_bk_enabled=False)
        self.conv6 = Conv(device, parameters.conv_args[7], parameters.model[7])
        self.c3k2_4 = C3k2(device, parameters.conv_args[8], parameters.model[8], is_bk_enabled=False)
        self.sppf = SPPF(device, parameters.conv_args[9], parameters.model[9])
        self.c2psa = C2PSA(device, parameters.conv_args[10], parameters.model[10])
        self.c3k2_5 = C3k2(
            device,
            parameters.conv_args[13],
            parameters.model[13],
            is_bk_enabled=True,
        )
        self.c3k2_6 = C3k2(
            device,
            parameters.conv_args[16],
            parameters.model[16],
            is_bk_enabled=True,
        )
        self.conv7 = Conv(device, parameters.conv_args[17], parameters.model[17])
        self.c3k2_7 = C3k2(
            device,
            parameters.conv_args[19],
            parameters.model[19],
            is_bk_enabled=True,
        )
        self.conv8 = Conv(device, parameters.conv_args[20], parameters.model[20])
        self.c3k2_8 = C3k2(
            device,
            parameters.conv_args[22],
            parameters.model[22],
            is_bk_enabled=False,
        )
        self.detect = Detect(device, parameters.model_args.model[23], parameters.model[23])

    def __call__(self, x):
        x = self.conv1(self.device, x)
        x = self.conv2(self.device, x)
        x = self.c3k2_1(self.device, x)
        x = self.conv3(self.device, x)
        x = self.c3k2_2(self.device, x)
        x4 = x
        x = self.conv5(self.device, x)
        x = self.c3k2_3(self.device, x)
        x6 = x
        x = self.conv6(self.device, x)
        x = self.c3k2_4(self.device, x)
        x = self.sppf(self.device, x)
        x = self.c2psa(self.device, x)
        x10 = x
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores_for_upsample(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)
        x = ttnn.upsample(x, scale_factor=2, memory_config=x.memory_config())  # 11
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        x6 = ttnn.to_layout(x6, layout=ttnn.ROW_MAJOR_LAYOUT)
        print("x is", x, x[0].shape)
        shard_height = (x.shape[2] + 64 - 1) // 64
        print("sahrd h is", shard_height)
        input_sharded_memory_config_1 = ttnn.create_sharded_memory_config(
            (shard_height, x.shape[-1]),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        input_sharded_memory_config_2 = ttnn.create_sharded_memory_config(
            (shard_height, x6.shape[-1]),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        p(x, "inpu1")
        p(x6, "input2")
        print("sahr1", input_sharded_memory_config_1)
        print("sahr2", input_sharded_memory_config_2)
        x = ttnn.to_memory_config(x, input_sharded_memory_config_1)
        x6 = ttnn.to_memory_config(x6, input_sharded_memory_config_2)
        out_sharded_memory_config_ = ttnn.create_sharded_memory_config(
            (shard_height, x.shape[-1] + x6.shape[-1]),
            core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))}),
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.concat((x, x6), -1, memory_config=out_sharded_memory_config_)

        ttnn.deallocate(x6)
        if x.shape[2] == 196:
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.c3k2_5(self.device, x)  # 13
        x13 = x
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores_for_upsample(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)
        x = ttnn.upsample(x, scale_factor=2, memory_config=x.memory_config())
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        x4 = ttnn.to_layout(x4, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = sharded_concat([x, x4])
        ttnn.deallocate(x4)
        x = self.c3k2_6(self.device, x)  # 16
        x16 = x
        x = self.conv7(self.device, x)  # 17
        x = ttnn.concat((x, x13), -1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 18
        ttnn.deallocate(x13)
        x = self.c3k2_7(self.device, x)  # 19
        x19 = x
        x = self.conv8(self.device, x)
        x = ttnn.concat((x, x10), -1, memory_config=ttnn.L1_MEMORY_CONFIG)  # 21
        ttnn.deallocate(x10)
        x = self.c3k2_8(self.device, x)  # 22
        x22 = x
        x = self.detect(self.device, x16, x19, x22)
        deallocate_tensors(x16, x19, x22)

        return x
