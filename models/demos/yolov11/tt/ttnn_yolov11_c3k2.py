# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors, sharded_concat
from models.demos.yolov11.tt.ttnn_yolov11_bottleneck import TtnnBottleneck
from models.demos.yolov11.tt.ttnn_yolov11_c3k import TtnnC3K


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape,x.padded_shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnC3k2:
    def __init__(self, device, parameter, conv_pt, is_bk_enabled=False, reshard=False):
        self.is_bk_enabled = is_bk_enabled
        self.parameter = parameter
        self.cv1_a = TtnnConv(
            device,
            parameter.cv1,
            conv_pt.cv1.a,
            reshard=reshard,
            split_weights=True,
        )  # matmul
        self.cv1_b = TtnnConv(
            device,
            parameter.cv1,
            conv_pt.cv1.b,
            reshard=reshard,
            split_weights=True,
        )  # matmul
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2, reshard=True)  # need resahrd as input is in RM
        if is_bk_enabled:
            self.k = TtnnBottleneck(device, parameter[0], conv_pt.m[0])
        else:
            self.c3k = TtnnC3K(device, parameter[0], conv_pt.m[0])

    def __call__(self, device, x, use_shard_concat=True, tile_shape=32):
        cv1_a = self.cv1_a(device, x)
        p(cv1_a, "cv1_a out is")
        print("cv1_a out is", cv1_a.shape, cv1_a.memory_config())
        if x.shape[2] == 400 and x.shape[-1] == 384:
            print("spcl case true is triggered")
            spcl_case = True
        else:
            spcl_case = False
        cv1_b = self.cv1_b(device, x, output_rm_needed=False, spcl_case=spcl_case)
        p(cv1_b, "cv1_b out is")
        print("cv1_b out is", cv1_b.shape, cv1_b.memory_config())
        # p(cv1_a, "c3k2 cv1_a")
        # p(cv1_b, "c3k2 cv1_b")
        # x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        # x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        # y1 = x[:, :, :, : x.shape[-1] // 2]
        # y2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        # print("c3k2 cv1 convs out",x.shape,y1.shape,y2.shape)
        if self.is_bk_enabled:
            y3 = self.k(device, cv1_b)
        else:
            y3 = self.c3k(device, cv1_b)
        p(y3, "y3 out is ")
        if cv1_b.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            cv1_b = ttnn.to_layout(cv1_b, ttnn.ROW_MAJOR_LAYOUT)
        if cv1_a.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            cv1_a = ttnn.to_layout(cv1_a, ttnn.ROW_MAJOR_LAYOUT)
        if y3.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            y3 = ttnn.to_layout(y3, ttnn.ROW_MAJOR_LAYOUT)
        if use_shard_concat:
            to_interleaved = True if (cv1_a.shape[3] < tile_shape) else False
            # p(cv1_a, "1st")
            # p(cv1_b, "2nd")
            # p(y3, "3rd")
            x = sharded_concat([cv1_a, cv1_b, y3], to_interleaved=to_interleaved)
        else:
            cv1_a = ttnn.sharded_to_interleaved(cv1_a, ttnn.L1_MEMORY_CONFIG)
            cv1_b = ttnn.sharded_to_interleaved(cv1_b, ttnn.L1_MEMORY_CONFIG)
            y3 = ttnn.sharded_to_interleaved(y3, ttnn.L1_MEMORY_CONFIG)
            x = ttnn.concat((cv1_a, cv1_b, y3), 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        # p(x, "concat out is")

        x = self.cv2(device, x, output_rm_needed=False)
        deallocate_tensors(cv1_a, cv1_b, y3)
        return x
