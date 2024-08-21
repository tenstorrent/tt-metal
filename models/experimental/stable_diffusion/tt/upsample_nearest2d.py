# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import ttnn


class TtUpsampleNearest2d(nn.Module):
    def __init__(self, scale_factor=2.0):
        super().__init__()

        assert scale_factor % 1 == 0 and scale_factor > 0, "We only support scaling by positive integer values"
        self.scale_factor = int(scale_factor)

    def forward(self, input: ttnn.Tensor) -> ttnn.Tensor:
        input_shape = input.get_legacy_shape()
        output_shape = list(input.get_legacy_shape())
        output_shape[-1] *= self.scale_factor
        output_shape[-2] *= self.scale_factor
        input = ttnn.repeat_interleave(input, self.scale_factor, dim=3)
        input = ttnn.repeat_interleave(input, self.scale_factor, dim=2)
        return input
