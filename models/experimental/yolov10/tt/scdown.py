# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.yolov10.tt.common import Conv
import ttnn


class TtnnSCDown:
    def __init__(self, device=None, parameters=None, conv_pt=None, auto_shard=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
            enable_identity=True,
            use_1d_systolic_array=False,
            auto_shard=auto_shard,
            deallocate_activation=True,
        )

    def __call__(self, input_tensor):
        cv1 = self.cv1(input_tensor)
        cv1 = ttnn.sharded_to_interleaved(
            cv1, ttnn.L1_MEMORY_CONFIG
        )  # needed since cv2 uses block_sharding and input is in height sharding
        output = self.cv2(cv1)
        return output
