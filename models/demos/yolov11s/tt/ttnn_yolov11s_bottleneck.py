# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11s.tt.common import TtnnConv


def _to_l1_interleaved(x):
    return ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG) if x.is_sharded() else x


class TtnnBottleneck:
    def __init__(self, device, parameter, conv_pt):
        self.cv1 = TtnnConv(device, parameter.cv1, conv_pt.cv1, reshard=False)
        self.cv2 = TtnnConv(device, parameter.cv2, conv_pt.cv2, reshard=False)

    def __call__(self, device, x):
        cfg = self.cv1.conv.conv
        hw = int(cfg.input_height) * int(cfg.input_width)
        residual = x
        x = self.cv1(device, x)
        x = self.cv2(device, x)
        if residual.shape[2] == x.shape[2] == hw and residual.is_sharded() == x.is_sharded():
            x = ttnn.add(residual, x, memory_config=x.memory_config())
        else:
            residual = _to_l1_interleaved(residual)
            x = _to_l1_interleaved(x)
            if int(residual.shape[2]) > hw:
                residual = residual[:, :, :hw, :]
            if int(x.shape[2]) > hw:
                x = x[:, :, :hw, :]
            x = ttnn.add(residual, x, memory_config=ttnn.L1_MEMORY_CONFIG)
        return x
