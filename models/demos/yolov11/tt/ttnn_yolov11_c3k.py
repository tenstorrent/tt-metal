# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors, sharded_concat
from models.demos.yolov11.tt.ttnn_yolov11_bottleneck import TtnnBottleneck


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape,x.padded_shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnC3K:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2)
        self.cv3 = TtnnConv(device, parameter.cv3, conv_pt.cv3, reshard=True)  # needed as input is RM
        self.k1 = TtnnBottleneck(device, parameter.m[0], conv_pt.m[0])
        self.k2 = TtnnBottleneck(device, parameter.m[1], conv_pt.m[1])

    def __call__(self, device, x, use_shard_concat=True):
        p(x, "c3k block input")
        x1 = self.cv1(device, x, output_rm_needed=False)
        # if x1.shape[2]==416:
        #     x1 = x1[:,:,:400,:]
        p(x1, "x1  out")
        x2 = self.cv2(device, x)
        p(x2, "x2  out")
        k1 = self.k1(device, x1)
        p(k1, "k1  out")
        k2 = self.k2(device, k1)
        p(k2, "k2 block out")
        if use_shard_concat:
            x2 = ttnn.to_layout(x2, ttnn.ROW_MAJOR_LAYOUT)
            k2 = ttnn.to_layout(k2, ttnn.ROW_MAJOR_LAYOUT)
            # p(k2, "k2")
            # p(x2, "x2")
            x = sharded_concat([k2, x2], to_interleaved=False)
            p(x, "c3k concat out ")
        else:
            x = ttnn.concat((k2, x2), 3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv3(device, x, output_rm_needed=True)
        p(x, "c3k cv3 out")
        deallocate_tensors(x1, x2, k1, k2)
        return x
