# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

import tt_lib as ttl
from tt_lib.fallback_ops import fallback_ops


class TtUpsampleNearest2d(nn.Module):
    def __init__(self, scale_factor=2.0):
        super().__init__()

        assert scale_factor % 1 == 0 and scale_factor > 0, "We only support scaling by positive integer values"
        self.scale_factor = int(scale_factor)

    def forward(self, input: ttl.tensor.Tensor) -> ttl.tensor.Tensor:
        input_shape = input.get_legacy_shape()
        output_shape = list(input.get_legacy_shape())
        output_shape[-1] *= self.scale_factor
        output_shape[-2] *= self.scale_factor
        input = ttl.tensor.repeat_interleave(input, self.scale_factor, dim=3)
        input = ttl.tensor.repeat_interleave(input, self.scale_factor, dim=2)
        return input
