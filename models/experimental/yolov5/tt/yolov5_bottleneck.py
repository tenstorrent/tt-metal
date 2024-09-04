# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from models.experimental.yolov5.tt.yolov5_conv import TtYolov5Conv


class TtYolov5Bottleneck(torch.nn.Module):
    # Standard bottleneck
    def __init__(
        self, state_dict, base_address, device, c1, c2, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()

        self.device = device
        c_ = int(c2 * e)  # hidden channels

        self.cv1 = TtYolov5Conv(
            state_dict=state_dict,
            base_address=f"{base_address}.cv1",
            device=device,
            c1=c1,
            c2=c_,
            k=1,
            s=1,
        )

        self.cv2 = TtYolov5Conv(
            state_dict=state_dict,
            base_address=f"{base_address}.cv2",
            device=device,
            c1=c_,
            c2=c2,
            k=3,
            s=1,
            g=g,
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        tmp = self.cv1(x)
        conv_res = self.cv2(tmp)

        res = ttnn.add(x, conv_res) if self.add else conv_res
        return res
