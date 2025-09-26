# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn
from models.demos.yolov5x.tt.common import TtnnBottleneck, TtYOLOv5xConv2D, deallocate_tensors
from models.experimental.yolo_common.yolo_utils import concat


class TtnnC3:
    def __init__(self, shortcut=True, n=4, device=None, parameters=None, conv_pt=None, use_block_shard=False):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.use_block_shard = use_block_shard
        if use_block_shard:
            shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED
        else:
            shard_layout = None

        self.cv1 = TtYOLOv5xConv2D(
            device,
            parameters.cv1.conv,
            self.conv_pt.cv1.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            shard_layout=shard_layout,
        )

        self.cv2 = TtYOLOv5xConv2D(
            device,
            parameters.cv2.conv,
            self.conv_pt.cv2.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            shard_layout=shard_layout,
        )

        self.cv3 = TtYOLOv5xConv2D(
            device,
            parameters.cv3.conv,
            self.conv_pt.cv3.conv,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU),
            auto_shard=True if use_block_shard else False,
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

        if self.use_block_shard:
            concat_out = concat(-1, False, m_out, cv2_out)
        else:
            concat_out = concat(-1, True, m_out, cv2_out)

        if self.use_block_shard:
            concat_out = ttnn.sharded_to_interleaved(concat_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        out = self.cv3(concat_out)
        deallocate_tensors(m_out, cv2_out)

        return out
