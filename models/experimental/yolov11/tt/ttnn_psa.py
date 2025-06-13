# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov11.tt.common import Conv
from models.experimental.yolov11.tt.ttnn_attention import Attention


class PSABlock:
    def __init__(self, device, parameter, conv_pt):
        self.attn = Attention(device=device, parameter=parameter.attn, conv_pt=conv_pt.attn)
        self.ffn_conv1 = Conv(device, parameter.ffn[0], conv_pt.ffn[0])
        self.ffn_conv2 = Conv(device, parameter.ffn[1], conv_pt.ffn[1], enable_act=False)

    def __call__(self, device, x):
        x1 = x
        x = self.attn(device, x)
        x = ttnn.add(x1, x, memory_config=x.memory_config())
        x1 = x
        x = self.ffn_conv1(device, x)
        x = self.ffn_conv2(device, x)
        x = ttnn.add(x, x1, memory_config=x1.memory_config())
        return x
