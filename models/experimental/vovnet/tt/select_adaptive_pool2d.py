# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
from typing import Union, Tuple

import tt_lib.fallback_ops


class TtSelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size"""

    def __init__(
        self,
        output_size: Union[int, Tuple[int, int]] = 1,
        pool_type: str = "fast",
        flatten: bool = False,
        input_fmt: str = "NCHW",
        device=None,
    ):
        super(TtSelectAdaptivePool2d, self).__init__()
        self.device = device
        assert input_fmt in ("NCHW", "NHWC")
        self.pool_type = pool_type or ""  # convert other falsy values to empty string for consistent TS typing
        if not pool_type:
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Flatten(1)
        else:
            assert input_fmt == "NCHW"
            if pool_type != "max":
                self.pool = tt_lib.fallback_ops.AdaptiveAvgPool2d(output_size)
        self.shape = (1, 1, 1, 1024)

    def forward(self, x):
        x = self.pool(x)
        x = tt_lib.fallback_ops.reshape(x, *self.shape)
        return x
