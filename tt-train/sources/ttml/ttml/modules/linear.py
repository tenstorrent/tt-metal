# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import math

import ml_dtypes

import ttnn
import ttml

from .module_base import AbstractModuleBase
from .parameter import Parameter


class LinearLayer(AbstractModuleBase):
    """Fully-connected linear layer: y = x @ W^T + b."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        if weight_init is None:
            k = math.sqrt(1.0 / in_features)
            weight_init = ttml.init.uniform(-k, k)
        if bias_init is None:
            k = math.sqrt(1.0 / in_features)
            bias_init = ttml.init.uniform(-k, k)

        weight_shape = (1, 1, out_features, in_features)
        self.weight = Parameter(weight_init(weight_shape))

        if has_bias:
            bias_shape = (1, 1, 1, out_features)
            self.bias = Parameter(bias_init(bias_shape))
        else:
            self.bias = None

    def __reduce__(self):
        return (
            self.__class__,
            (self.in_features, self.out_features, self.bias is not None),
            self.__getstate__(),
        )

    def __getstate__(self):
        return {
            "weight": self.weight.tensor.to_numpy(ttnn.DataType.FLOAT32),
            "bias": self.bias.tensor.to_numpy(ttnn.DataType.FLOAT32) if self.bias is not None else None,
        }

    def __setstate__(self, state):
        self.weight.tensor.set_value(
            ttml.autograd.Tensor.from_numpy(
                state["weight"].astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
            ).get_value()
        )
        if state["bias"] is not None:
            if self.bias is None:
                raise ValueError("LinearLayer bias was improperly initialized when deserializing from Pickle")
            self.bias.tensor.set_value(
                ttml.autograd.Tensor.from_numpy(
                    state["bias"].astype(ml_dtypes.bfloat16), layout=ttnn.Layout.TILE
                ).get_value()
            )

    def forward(self, x):
        """Compute linear projection of x."""
        bias_t = self.bias.tensor if self.bias is not None else None
        return ttml.ops.linear.linear(x, self.weight.tensor, bias_t)


class ColumnParallelLinear(AbstractModuleBase):
    """Column-parallel linear layer. Shards output features across TP devices."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
        gather_output: bool = False,
        axis_name: str = "tp",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.axis_name = axis_name
        self.cluster_axis = ttml.current_mesh().axis_pos(axis_name)

        if weight_init is None:
            k = math.sqrt(1.0 / in_features)
            weight_init = ttml.init.uniform(-k, k)
        if bias_init is None:
            k = math.sqrt(1.0 / in_features)
            bias_init = ttml.init.uniform(-k, k)

        weight_shape = (1, 1, out_features, in_features)
        weight_mapper = ttml.current_mesh().axis_mapper(axis_name, tdim=2)
        self.weight = Parameter(weight_init(weight_shape, mapper=weight_mapper))

        if has_bias:
            bias_shape = (1, 1, 1, out_features)
            bias_mapper = ttml.current_mesh().axis_mapper(axis_name, tdim=3)
            self.bias = Parameter(bias_init(bias_shape, mapper=bias_mapper))
        else:
            self.bias = None

    def forward(self, x):
        """Compute linear projection of x."""
        bias_t = self.bias.tensor if self.bias is not None else None
        x = ttml.ops.distributed.broadcast(x, self.cluster_axis)
        x = ttml.ops.linear.linear(x, self.weight.tensor, bias_t)
        if self.gather_output:
            x = ttml.ops.distributed.all_gather(x, 3, self.cluster_axis, ttml.ops.distributed.GradOutputType.REPLICATED)
        return x


class RowParallelLinear(AbstractModuleBase):
    """Row-parallel linear layer. Shards input features across TP devices."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
        input_is_parallel: bool = False,
        axis_name: str = "tp",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        # TODO: use ttnn.Tensor.tensor_topology() once CCL ops will properly update this property
        self.input_is_parallel = input_is_parallel
        self.axis_name = axis_name
        self.cluster_axis = ttml.current_mesh().axis_pos(axis_name)

        if weight_init is None:
            k = math.sqrt(1.0 / in_features)
            weight_init = ttml.init.uniform(-k, k)
        if bias_init is None:
            k = math.sqrt(1.0 / in_features)
            bias_init = ttml.init.uniform(-k, k)

        weight_shape = (1, 1, out_features, in_features)
        weight_mapper = ttml.current_mesh().axis_mapper(axis_name, tdim=3)
        self.weight = Parameter(weight_init(weight_shape, mapper=weight_mapper))

        if has_bias:
            bias_shape = (1, 1, 1, out_features)
            self.bias = Parameter(bias_init(bias_shape))
        else:
            self.bias = None

    def forward(self, x):
        """Compute linear projection of x."""
        if not self.input_is_parallel:
            x = ttml.ops.distributed.scatter(x, 3, self.cluster_axis)
        x = ttml.ops.linear.linear(x, self.weight.tensor, None)
        x = ttml.ops.distributed.all_reduce(x, self.input_is_parallel, self.cluster_axis)
        if self.bias is not None:
            x = ttml.ops.binary.add(x, self.bias.tensor)
        return x
