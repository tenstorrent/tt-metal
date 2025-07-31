# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolo_common.yolo_utils import concat
from models.experimental.yolov12x.tt.common import TtYOLOv12xConv2D, TtnnBottleneck


class TtnnC3k:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = TtYOLOv12xConv2D(
            conv=parameter.cv1.conv, conv_pth=conv_pt.cv1.conv, device=device, activation="silu"
        )
        self.cv2 = TtYOLOv12xConv2D(
            conv=parameter.cv2.conv, conv_pth=conv_pt.cv2.conv, device=device, activation="silu"
        )
        self.cv3 = TtYOLOv12xConv2D(
            conv=parameter.cv3.conv,
            conv_pth=conv_pt.cv3.conv,
            device=device,
            activation="silu",
            deallocate_activation=True,
        )
        self.k1 = TtnnBottleneck(device, parameter.m[0], conv_pt.m[0])
        self.k2 = TtnnBottleneck(device, parameter.m[1], conv_pt.m[1])

    def __call__(self, x, i=0, j=0):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        k1 = self.k1(x1)
        ttnn.deallocate(x1)

        k2 = self.k2(k1)
        ttnn.deallocate(k1)

        x = concat(-1, True, k2, x2)
        ttnn.deallocate(x2)
        ttnn.deallocate(k2)

        x = self.cv3(x)
        return x


class TtnnC3k2:
    def __init__(
        self,
        device,
        parameter,
        conv_pt,
        is_bk_enabled=False,
        use_1d_systolic_array=True,
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ):
        self.parameter = parameter
        self.is_bk_enabled = is_bk_enabled
        conv_config_override = True if conv_pt.cv1.conv.weight.shape[1] == 1536 else False
        config_override = {"act_block_h": 32} if conv_config_override else None

        self.cv1 = TtYOLOv12xConv2D(
            conv=parameter.cv1.conv,
            conv_pth=conv_pt.cv1.conv,
            device=device,
            config_override=config_override,
            activation="silu",
            use_1d_systolic_array=use_1d_systolic_array,
            shard_layout=shard_layout,
            deallocate_activation=True,
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

        if is_bk_enabled:
            self.m = Bottleneck(device, parameter[0], conv_pt.m[0])
        else:
            self.m_0 = TtnnC3k(device, parameter[0], conv_pt.m[0])
            self.m_1 = TtnnC3k(device, parameter[1], conv_pt.m[1])

    def __call__(self, x, i=0):
        x = self.cv1(x)
        if x.is_sharded():
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)

        y1 = x[:, :, :, : x.shape[-1] // 2]
        y2 = x[:, :, :, x.shape[-1] // 2 : x.shape[-1]]
        if self.is_bk_enabled:
            y3 = self.m(y2)
            x = concat(-1, False, y1, y2, y3)
        else:
            y3 = self.m_0(y2, i=i, j=0)
            y4 = self.m_1(y3, i=i, j=1)
            x = concat(-1, False, y1, y2, y3, y4)
            ttnn.deallocate(y4)

        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)

        x = self.cv2(x)
        return x
