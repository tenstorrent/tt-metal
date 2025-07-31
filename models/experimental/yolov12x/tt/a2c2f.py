# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.yolo_common.yolo_utils import concat
from models.experimental.yolov12x.tt.c3k2 import TtnnC3k
from models.experimental.yolov12x.tt.ablock import TtnnABlock
from models.experimental.yolov12x.tt.common import TtYOLOv12xConv2D


class TtnnA2C2f:
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
        self.cv1 = TtYOLOv12xConv2D(
            conv=parameter.cv1.conv,
            conv_pth=conv_pt.cv1.conv,
            device=device,
            activation="silu",
            config_override=config_override,
            shard_layout=shard_layout,
        )
        self.cv2 = TtYOLOv12xConv2D(
            conv=parameter.cv2.conv,
            conv_pth=conv_pt.cv2.conv,
            device=device,
            activation="silu",
            use_1d_systolic_array=use_1d_systolic_array,
            shard_layout=shard_layout,
            config_override=config_override,
            deallocate_activation=True,
        )
        if a2 and residual:
            self.gamma = 0.01 * ttnn.ones(
                [c2], dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        else:
            self.gamma = None

        self.m = {}
        for i in range(self.n):
            if a2:
                self.m[i] = {}
                for j in range(2):
                    self.m[i][j] = TtnnABlock(
                        device,
                        parameter[i][j],
                        conv_pt.m[i][j],
                        dim=384,
                        num_heads=12,
                        mlp_ratio=1.2,
                        area=area,
                        is_bk_enabled=False,
                    )
            else:
                self.m[i] = TtnnC3k(device, parameter[i], conv_pt.m[i])

    def __call__(self, x, i=0):
        y = [self.cv1(x)]
        if self.gamma is None:
            ttnn.deallocate(x)

        out = y[-1]
        for i in range(self.n):
            if self.a2:
                out = y[-1]
                for j in range(2):
                    out = self.m[i][j](out, i=i, j=j)
                y.append(out)
            else:
                out = self.m[i](y[-1])
                y.append(out)

        y_concat = concat(-1, False, *y)
        y_concat = ttnn.sharded_to_interleaved(y_concat, ttnn.L1_MEMORY_CONFIG)

        y = self.cv2(y_concat)
        ttnn.deallocate(y_concat)

        if self.gamma is not None:
            if x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
            if y.is_sharded():
                y = ttnn.sharded_to_interleaved(y, ttnn.L1_MEMORY_CONFIG)
            gamma = ttnn.unsqueeze_to_4D(self.gamma)
            y = gamma * y
            return x + y

        return y
