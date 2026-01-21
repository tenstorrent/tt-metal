# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov10x.tt.attention import TtnnAttention
from models.demos.yolov10x.tt.common import Conv, deallocate_tensors
from models.experimental.yolo_common.yolo_utils import concat


class TtnnPSA:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(device, parameters.cv1, self.conv_pt.cv1, deallocate_activation=True)

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
        )

        self.attn = TtnnAttention(
            dim=320,
            num_heads=5,
            attn_ratio=0.5,
            device=self.device,
            parameters=self.parameters.attn,
            conv_pt=self.conv_pt.attn,
        )

        self.ffn_0 = Conv(
            device,
            parameters.ffn[0],
            self.conv_pt.ffn[0],
        )

        self.ffn_1 = Conv(
            device,
            parameters.ffn[1],
            self.conv_pt.ffn[1],
            enable_identity=True,
        )

    def __call__(self, input_tensor):
        cv1 = self.cv1(input_tensor)
        cv1 = ttnn.sharded_to_interleaved(cv1, ttnn.L1_MEMORY_CONFIG)
        a = cv1[:, :, :, : cv1.shape[-1] // 2]
        b = cv1[:, :, :, cv1.shape[-1] // 2 :]

        out = self.attn(b)

        b = b + out

        out = self.ffn_0(b)
        out = self.ffn_1(out)

        b = b + out

        out = concat(-1, True, a, b)
        out = ttnn.sharded_to_interleaved(out, memory_config=ttnn.L1_MEMORY_CONFIG)

        output = self.cv2(out)
        deallocate_tensors(a, b)

        return output
