# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import Union, Tuple
import ttnn


class TtSelectAdaptivePool2d:
    """Selectable global pooling layer with dynamic input kernel size"""

    def __init__(
        self,
        device=None,
    ):
        self.device = device

    def forward(self, x):
        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.global_avg_pool2d(x)
        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.reshape(x, [x.shape[0], 1, 1, x.shape[1] * x.shape[2] * x.shape[3]])
        return x
