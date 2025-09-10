# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.experimental.yolov5x.tt.common import (
    TtYOLOv5xConv2D,
    TtnnBottleneck,
    deallocate_tensors,
    interleaved_to_sharded,
)
from models.experimental.yolo_common.yolo_utils import concat


class TtnnC3:
    def __init__(self, shortcut=True, n=4, device=None, parameters=None, conv_pt=None, use_block_shard=False):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = TtYOLOv5xConv2D(
            device,
            parameters.cv1.conv,
            self.conv_pt.cv1.conv,
            activation="silu",
        )

        self.cv2 = TtYOLOv5xConv2D(
            device,
            parameters.cv2.conv,
            self.conv_pt.cv2.conv,
            activation="silu",
        )

        self.cv3 = TtYOLOv5xConv2D(
            device,
            parameters.cv3.conv,
            self.conv_pt.cv3.conv,
            activation="silu",
            auto_shard=True,
        )

        self.m = [
            TtnnBottleneck(
                self.shortcut,
                device=self.device,
                parameters=self.parameters.m[i],
                conv_pt=self.conv_pt.m[i],
                label=(i == 0),
                use_block_shard=use_block_shard,
            )
            for i in range(n)
        ]

    def __call__(self, input_tensor):
        m_out = self.cv1(input_tensor)

        for m in self.m:
            m_out = m(m_out)

        cv2_out = self.cv2(input_tensor)

        if cv2_out.shape[2] != m_out.shape[2]:
            cv2_out = ttnn.sharded_to_interleaved(cv2_out, memory_config=ttnn.L1_MEMORY_CONFIG)
            cv2_out = cv2_out[:, :, : m_out.shape[2], :]
            cv2_out = interleaved_to_sharded(cv2_out)

        concat_out = concat(-1, True, m_out, cv2_out)
        concat_out = ttnn.sharded_to_interleaved(concat_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv3(concat_out)
        deallocate_tensors(m_out, cv2_out)

        return out
