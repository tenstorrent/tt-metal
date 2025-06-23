# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn

from models.experimental.yolo_common.yolo_utils import concat

from models.experimental.yolov12x.tt.c3k2 import TtnnC3k2
from models.experimental.yolov12x.tt.a2c2f import TtnnA2C2f
from models.experimental.yolov12x.tt.detect import TtnnDetect
from models.experimental.yolov12x.tt.common import TtYOLOv12xConv2D, interleaved_to_sharded


class YoloV12x:
    def __init__(self, device, parameters):
        self.device = device
        self.conv1 = TtYOLOv12xConv2D(
            device=device,
            conv=parameters.conv_args[0].conv,
            conv_pth=parameters.model[0].conv,
            config_override={"act_block_h": 32},
            activation="silu",
            deallocate_activation=True,
        )  # 0
        self.conv2 = TtYOLOv12xConv2D(
            device=device,
            conv=parameters.conv_args[1].conv,
            conv_pth=parameters.model[1].conv,
            config_override={"act_block_h": 32},
            activation="silu",
            deallocate_activation=True,
        )  # 1
        self.c3k2_1 = TtnnC3k2(device, parameters.conv_args[2], parameters.model[2])  # 2
        self.conv3 = TtYOLOv12xConv2D(
            device=device,
            conv=parameters.conv_args[3].conv,
            conv_pth=parameters.model[3].conv,
            activation="silu",
            config_override={"act_block_h": 32},
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )  # 3
        self.c3k2_2 = TtnnC3k2(device, parameters.conv_args[4], parameters.model[4])  # 4
        self.conv4 = TtYOLOv12xConv2D(
            device=device,
            conv=parameters.conv_args[5].conv,
            conv_pth=parameters.model[5].conv,
            activation="silu",
            config_override={"act_block_h": 32},
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
        )  # 5
        self.a2c2f_1 = TtnnA2C2f(
            device,
            parameters.conv_args[6],
            parameters.model[6],
            c1=768,
            c2=768,
            n=4,
            a2=True,
            area=4,
            residual=True,
            mlp_ratio=1.2,
            e=0.5,
            g=1,
            shortcut=True,
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            config_override={"act_block_h": 32},
        )  # 6
        self.conv5 = TtYOLOv12xConv2D(
            device=device,
            conv=parameters.conv_args[7].conv,
            conv_pth=parameters.model[7].conv,
            activation="silu",
            config_override={"act_block_h": 32},
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )  # 7
        self.a2c2f_2 = TtnnA2C2f(
            device,
            parameters.conv_args[8],
            parameters.model[8],
            c1=768,
            c2=768,
            n=4,
            a2=True,
            area=4,
            residual=True,
            mlp_ratio=1.2,
            e=0.5,
            g=1,
            shortcut=True,
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            config_override={"act_block_h": 32},
        )  # 8
        self.a2c2f_3 = TtnnA2C2f(
            device,
            parameters.conv_args[11],
            parameters.model[11],
            c1=1536,
            c2=768,
            n=2,
            a2=False,
            area=-1,
            residual=True,
            mlp_ratio=1.2,
            e=0.5,
            g=1,
            shortcut=True,
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            config_override={"act_block_h": 32},
        )  # 11
        self.a2c2f_4 = TtnnA2C2f(
            device,
            parameters.conv_args[14],
            parameters.model[14],
            c1=1536,
            c2=384,
            n=2,
            a2=False,
            area=-1,
            residual=True,
            mlp_ratio=1.2,
            e=0.5,
            g=1,
            shortcut=True,
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            config_override={"act_block_h": 32},
        )  # 14
        self.conv6 = TtYOLOv12xConv2D(
            device=device,
            conv=parameters.conv_args[15].conv,
            conv_pth=parameters.model[15].conv,
            activation="silu",
            config_override={"act_block_h": 32},
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )  # 15
        self.a2c2f_5 = TtnnA2C2f(
            device,
            parameters.conv_args[17],
            parameters.model[17],
            c1=1152,
            c2=768,
            n=2,
            a2=False,
            area=-1,
            residual=True,
            mlp_ratio=1.2,
            e=0.5,
            g=1,
            shortcut=True,
        )  # 17
        self.conv7 = TtYOLOv12xConv2D(
            device=device,
            conv=parameters.conv_args[18].conv,
            conv_pth=parameters.model[18].conv,
            activation="silu",
            config_override={"act_block_h": 32},
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )  # 18

        self.c3k2_3 = TtnnC3k2(
            device,
            parameters.conv_args[20],
            parameters.model[20],
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )  # 20
        self.detect = TtnnDetect(device, parameters.model_args.model[21], parameters.model[21])  # 21

    def __call__(self, x):
        x = self.conv1(x)  # 0
        x = self.conv2(x)  # 1
        x = self.c3k2_1(x, i=4)  # 2
        x = self.conv3(x)  # 3
        x = self.c3k2_2(x, i=6)  # 4
        x4 = x
        x = self.conv4(x)  # 5
        x = self.a2c2f_1(x, i=8)  # 6
        x6 = x
        x = self.conv5(x)  # 7
        x = self.a2c2f_2(x, i=10)  # 8
        x8 = x
        x8 = ttnn.to_memory_config(x8, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.DumpDeviceProfiler(self.device)

        x = interleaved_to_sharded(x)
        x = ttnn.upsample(x, scale_factor=2)  # 9
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]))
        x = concat(-1, True, x, x6)  # 10
        ttnn.deallocate(x6)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.a2c2f_3(x, i=13)  # 11
        x11 = x
        x11 = ttnn.to_memory_config(x11, ttnn.DRAM_MEMORY_CONFIG)

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = interleaved_to_sharded(x)
        x = ttnn.upsample(x, scale_factor=2)  # 12
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]))

        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        if x4.is_sharded():
            x4 = ttnn.sharded_to_interleaved(x4, ttnn.L1_MEMORY_CONFIG)
        x = concat(-1, True, x, x4)  # 13
        ttnn.deallocate(x4)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self.a2c2f_4(x, i=16)  # 14
        x14 = x
        x14 = ttnn.to_memory_config(x14, ttnn.DRAM_MEMORY_CONFIG)
        x = self.conv6(x)  # 15

        if x.layout == ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x11 = ttnn.to_memory_config(x11, ttnn.L1_MEMORY_CONFIG)
        if x11.layout == ttnn.TILE_LAYOUT:
            x11 = ttnn.to_layout(x11, ttnn.ROW_MAJOR_LAYOUT)
        x = concat(-1, False, x, x11)  # 16
        ttnn.deallocate(x11)
        ttnn.DumpDeviceProfiler(self.device)
        x = self.a2c2f_5(x, i=19)  # 17
        x17 = x
        x17 = ttnn.to_memory_config(x17, ttnn.DRAM_MEMORY_CONFIG)
        x = self.conv7(x)  # 18
        x8 = ttnn.to_memory_config(x8, ttnn.L1_MEMORY_CONFIG)
        x = concat(-1, False, x, x8)  # 19
        ttnn.deallocate(x8)
        x = self.c3k2_3(x, i=22)  # 20
        x20 = x
        x14 = ttnn.to_memory_config(x14, ttnn.L1_MEMORY_CONFIG)
        x17 = ttnn.to_memory_config(x17, ttnn.L1_MEMORY_CONFIG)
        ttnn.DumpDeviceProfiler(self.device)

        x = self.detect(x14, x17, x20)  # 21
        return x
