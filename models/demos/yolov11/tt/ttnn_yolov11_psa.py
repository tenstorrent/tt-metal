# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11.tt.common import TtnnConv
from models.demos.yolov11.tt.ttnn_yolov11_attention import TtnnAttention


class TtnnPSABlock:
    def __init__(self, device, parameter, conv_pt):
        self.attn = TtnnAttention(device=device, parameter=parameter.attn, conv_pt=conv_pt.attn)
        self.ffn_conv1 = TtnnConv(device, parameter.ffn[0], conv_pt.ffn[0])
        self.ffn_conv2 = TtnnConv(device, parameter.ffn[1], conv_pt.ffn[1], enable_act=False)

    def __call__(self, device, x):
        x1 = x
        x = self.attn(device, x)
        x = ttnn.add(x1, x, memory_config=x.memory_config())
        x1 = x
        x = self.ffn_conv1(device, x)
        x = self.ffn_conv2(device, x)
        x = ttnn.add(x, x1, memory_config=x1.memory_config())
        return x
