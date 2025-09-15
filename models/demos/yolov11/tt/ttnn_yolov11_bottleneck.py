# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


class TtnnBottleneck:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1, reshard=False)  # core_count=64 and resahrd at foward
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2, reshard=False)  # core_count=64

    def __call__(self, device, x, tile_shape=32):
        input = x
        if use_signpost:
            signpost(header="bottleneck 1conv")
        x = self.cv1(device, x)
        if use_signpost:
            signpost(header="bottleneck 2conv")
        x = self.cv2(device, x)
        print("bot222", x.memory_config(), x.shape)
        # p(x, "x")
        # p(input, "input is")
        if input.memory_config() != x.memory_config():
            print("it is execcc")
            input = ttnn.reshard(input, x.memory_config())
        x = ttnn.add(input, x, memory_config=x.memory_config())
        # if x.shape[3] < tile_shape:
        #     input = ttnn.to_layout(input, layout=ttnn.TILE_LAYOUT)
        #     x = ttnn.add(input, x, memory_config=x.memory_config())
        # else:
        #     x = ttnn.add(input, x, memory_config=x.memory_config(), use_legacy=False)
        return x
