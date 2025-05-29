# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.yolo_common.yolo_utils import concat
from models.experimental.yolov12x.tt.c3k2 import C3k
from models.experimental.yolov12x.tt.ablock import ABlock
from models.experimental.yolov12x.tt.common import Yolov12x_Conv2D


class A2C2f:
    def __init__(
        self,
        device,
        parameter,
        conv_pt,
        c1,
        c2,
        n=1,
        a2=True,
        area=1,
        residual=False,
        mlp_ratio=2.0,
        e=0.5,
        g=1,
        shortcut=True,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        config_override=None,
    ):
        residual = True
        self.m = []

        self.n = n
        self.a2 = a2
        self.area = area
        self.cv1 = Yolov12x_Conv2D(
            conv=parameter.cv1.conv,
            conv_pth=conv_pt.cv1.conv,
            device=device,
            activation="silu",
            config_override=config_override,
        )
        self.cv2 = Yolov12x_Conv2D(
            conv=parameter.cv2.conv,
            conv_pth=conv_pt.cv2.conv,
            device=device,
            activation="silu",
            use_1d_systolic_array=use_1d_systolic_array,
            shard_layout=shard_layout,
            config_override=config_override,
        )
        if a2 and residual:
            self.gamma = 0.01 * ttnn.ones(
                [c2], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG
            )
        else:
            self.gamma = None
        if a2:
            for i in range(2):
                layers = ABlock(
                    device,
                    parameter[i][i],
                    conv_pt.m[i][i],
                    dim=384,
                    num_heads=12,
                    mlp_ratio=1.2,
                    area=area,
                    is_bk_enabled=False,
                )
                self.m.append(layers)
        else:
            self.m.append(C3k(device, parameter[0], conv_pt.m[0]))

    def __call__(self, x):
        # y = self.cv1(x)
        # y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
        y = [self.cv1(x)]
        for _ in range(self.n):
            if self.a2:
                for m in self.m:
                    a0 = m(y[-1])
                    a1 = m(a0)
                y.append(a1)
            else:
                for m in self.m:
                    y.append(m(y[-1]))
        y_concat = concat(-1, True, *y)

        y_concat = ttnn.sharded_to_interleaved(y_concat, ttnn.L1_MEMORY_CONFIG)

        y = self.cv2(y_concat)
        if self.gamma is not None:
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            if y.is_sharded():
                y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
            gamma = ttnn.unsqueeze_to_4D(self.gamma)
            y = gamma * y
            ttnn.deallocate(gamma)
            return x + y

        return y
