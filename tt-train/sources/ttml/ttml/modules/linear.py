# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import math

import ml_dtypes
import ttnn
import ttml

from .module_base import AbstractModuleBase
from .parameter import Parameter, TensorMetadata


class LinearLayer(AbstractModuleBase):
    """Fully-connected linear layer: y = x @ W^T + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        **kwargs,
    ) -> None:
        self.in_features = in_features
        self.out_features = out_features

        k = math.sqrt(1.0 / in_features)

        self.weight = Parameter(
            TensorMetadata(
                shape=(1, 1, out_features, in_features),
                init_fn=ttml.init.uniform(-k, k),
            )
        )

        if has_bias:
            self.bias = Parameter(
                TensorMetadata(
                    shape=(1, 1, 1, out_features),
                    init_fn=ttml.init.uniform(-k, k),
                )
            )
        else:
            self.bias = None

        super().__init__(**kwargs)

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
