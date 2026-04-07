# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11l.tt.common import TtnnConv


class TtnnBottleneck:
    def __init__(self, device, parameter, conv_pt):
        inner_slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=8)
        self.cv1 = TtnnConv(
            device,
            parameter.cv1,
            conv_pt.cv1,
            shard_layout=None,
            reshard=True,
            slice_config=inner_slice_config,
        )
        self.cv2 = TtnnConv(
            device,
            parameter.cv2,
            conv_pt.cv2,
            shard_layout=None,
            reshard=True,
            slice_config=inner_slice_config,
        )

    def __call__(self, device, x, tile_shape=32):
        input = x
        x = self.cv1(device, x)
        x = self.cv2(device, x)
        if x.shape[3] < tile_shape:
            input = ttnn.to_layout(input, layout=ttnn.TILE_LAYOUT)
            x = ttnn.add(input, x, memory_config=x.memory_config())
        else:
            x = ttnn.add(input, x, memory_config=x.memory_config(), use_legacy=False)
        return x
