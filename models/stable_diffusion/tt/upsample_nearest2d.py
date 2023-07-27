from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

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
        input_shape = input.shape()
        output_shape = list(input.shape())
        output_shape[-1] *= self.scale_factor
        output_shape[-2] *= self.scale_factor
        input =  fallback_ops.repeat_interleave(input, repeats= self.scale_factor, dim=-1)
        input =  fallback_ops.repeat_interleave(input, repeats= self.scale_factor, dim=-2)


        return input
