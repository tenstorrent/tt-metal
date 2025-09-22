# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov7.tt.common import TtYOLOv7Matmul
from models.demos.yolov11.tt.common import deallocate_tensors, reshard_if_possible, sharded_concat
from models.demos.yolov11.tt.ttnn_yolov11_bottleneck import TtnnBottleneck


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnC3K:
    def __init__(self, device, parameter, conv_pt):
        # self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        # self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2)
        # self.cv3 = TtnnConv(device, parameter.cv3, conv_pt.cv3)  # needed as input is RM
        self.cv1 = TtYOLOv7Matmul(device, conv_pt.cv1.conv)
        self.cv2 = TtYOLOv7Matmul(device, conv_pt.cv2.conv)
        self.cv3 = TtYOLOv7Matmul(device, conv_pt.cv3.conv)  # needed as input is RM
        self.k1 = TtnnBottleneck(device, parameter.m[0], conv_pt.m[0])
        self.k2 = TtnnBottleneck(device, parameter.m[1], conv_pt.m[1])

    def __call__(self, device, x, use_shard_concat=True):
        x1 = self.cv1(device, x)
        x2 = self.cv2(device, x)
        k1 = self.k1(device, x1)
        k2 = self.k2(device, k1)
        if use_shard_concat:
            x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
            k2 = ttnn.to_layout(k2, ttnn.ROW_MAJOR_LAYOUT)
            p(k2, "k2")
            p(x2, "x2")
            x = sharded_concat([k2, x2], to_interleaved=False)
        else:
            x = ttnn.concat((k2, x2), 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        p(x, "cv3 in")

        x = reshard_if_possible(x)
        # if x.is_sharded() and (x.memory_config().shard_spec.shape[0] % 32 != 0 or x.memory_config().shard_spec.shape[1] % 32 != 0):
        #     print("BEFORE IS", x.memory_config().shard_spec.shape)
        #     aligned_h, aligned_w = roundup32(x.memory_config().shard_spec.shape[0]), roundup32(
        #         x.memory_config().shard_spec.shape[1]
        #     )
        #     print("after IS", aligned_h, aligned_w)
        #     resharded_memory_config = ttnn.create_sharded_memory_config(
        #         shape=(aligned_h, aligned_w),
        #         core_grid=x.memory_config().shard_spec.grid,
        #         strategy=ttnn.ShardStrategy.HEIGHT,
        #         orientation=x.memory_config().shard_spec.orientation,
        #         use_height_and_width_as_shard_shape=True,
        #     )
        #     x = ttnn.to_memory_config(x, resharded_memory_config)
        x = self.cv3(device, x)
        deallocate_tensors(x1, x2, k1, k2)
        return x
