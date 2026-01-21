# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

import ttnn


class TtConcat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, device, state_dict, base_address, dimension=1):
        super().__init__()
        self.device = device
        self.base_address = base_address

        self.d = dimension

    def forward(self, x):
        return ttnn.concat(x, self.d)
