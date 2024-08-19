# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import tt_lib
from loguru import logger
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch2tt_tensor, tt2torch_tensor


class TtUpsample(torch.nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        device,
        state_dict,
        base_address,
        size=None,
        scale_factor=None,
        mode="nearest",
    ):
        super().__init__()
        self.device = device
        self.upsample = torch.nn.Upsample(size=size, scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        x = tt2torch_tensor(x)
        x = self.upsample(x)
        x = torch2tt_tensor(x, self.device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)
        return x
