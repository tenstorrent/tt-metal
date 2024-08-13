# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn


class TtYolov5Concat(torch.nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, state_dict, base_address, device, dimension=1):
        super().__init__()
        self.device = device
        self.base_address = base_address

        self.d = dimension

    def forward(self, x):
        return ttnn.concat(x, self.d)
