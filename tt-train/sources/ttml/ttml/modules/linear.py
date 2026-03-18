# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import ml_dtypes

import ttnn
import ttml

from .module_base import AbstractModuleBase
from .parameter import Parameter


def _create_weight(in_features: int, out_features: int, zero_init: bool = False):
    # Shape matches C++ convention: (1, 1, out_features, in_features)
    device = ttml.autograd.AutoContext.get_instance().get_device()
    shape = (1, 1, out_features, in_features)
    if zero_init:
        weight_ttnn = ttnn.zeros(
            shape, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE
        )
    else:
        init_k = math.sqrt(1.0 / in_features)
        weight_ttnn = ttnn.rand(
            shape, device=device, dtype=ttnn.DataType.BFLOAT16, low=-init_k, high=init_k
        )
    return ttml.autograd.create_tensor(weight_ttnn)


def _create_bias(in_features: int, out_features: int, zero_init: bool = False):
    device = ttml.autograd.AutoContext.get_instance().get_device()
    shape = (1, 1, 1, out_features)
    if zero_init:
        bias_ttnn = ttnn.zeros(
            shape, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE
        )
    else:
        init_k = math.sqrt(1.0 / in_features)
        bias_ttnn = ttnn.rand(
            shape, device=device, dtype=ttnn.DataType.BFLOAT16, low=-init_k, high=init_k
        )
    return ttml.autograd.create_tensor(bias_ttnn)


class LinearLayer(AbstractModuleBase):
    """Fully-connected linear layer: y = x @ W^T + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        zero_init: bool = False,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(_create_weight(in_features, out_features, zero_init))
        self.bias = (
            Parameter(_create_bias(in_features, out_features, zero_init))
            if has_bias
            else None
        )

    def __reduce__(self):
        return (
            self.__class__,
            (self.in_features, self.out_features, self.bias is not None),
            self.__getstate__(),
        )

    def __getstate__(self):
        return {
            "weight": self.weight.tensor.to_numpy(ttnn.DataType.FLOAT32),
            "bias": self.bias.tensor.to_numpy(ttnn.DataType.FLOAT32)
            if self.bias is not None
            else None,
        }

    def __setstate__(self, state):
        self.weight.tensor.set_value(
            ttml.autograd.Tensor.from_numpy(
                state["weight"].astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ).get_value()
        )
        if state["bias"] is not None:
            if self.bias is None:
                raise ValueError(
                    "LinearLayer bias was improperly initialized when deserializing from Pickle"
                )
            self.bias.tensor.set_value(
                ttml.autograd.Tensor.from_numpy(
                    state["bias"].astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
                ).get_value()
            )

    def forward(self, x):
        """Compute linear projection of x."""
        bias = self.bias.tensor if self.bias is not None else None
        return ttml.ops.linear.linear(x, self.weight.tensor, bias)
