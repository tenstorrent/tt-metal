# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors
from models.demos.yolov11.tt.ttnn_yolov11_psa import TtnnPSABlock

try:
    from tracy import signpost

    use_signpost = True
except ModuleNotFoundError:
    use_signpost = False


def p(x, a="x"):
    print(f"{a}'s  shape: {x.shape,x.padded_shape}")
    print(f"{a}'s  layout: {x.layout}")
    print(f"{a}'s  dtype: {x.dtype}")
    print(f"{a}'s config: {x.memory_config()}")


class TtnnC2PSA:
    def __init__(self, device, parameter, conv_pt):
        self.out_channel_0 = parameter.cv1.conv.out_channels
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2)
        self.psablock = TtnnPSABlock(device, parameter.m[0], conv_pt.m[0])

    def __call__(self, device, x):
        p(x, "input to c2psa is")
        x = self.cv1(device, x, output_rm_needed=False)
        p(x, "output of c2psa 1st conv is")
        if x.get_layout() != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        p(x, "after layout change")
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        a, b = x[:, :, :400, : int(self.out_channel_0 / 2)], x[:, :, :400, int(self.out_channel_0 / 2) :]
        p(a, "a")
        p(b, "b")
        if use_signpost:
            signpost(header="psablock")
        x = self.psablock(device, b)
        p(x, "psa whole block out")
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.concat((a, x), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = self.cv2(device, x)
        deallocate_tensors(a, b)
        return x
