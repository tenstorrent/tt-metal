# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import math

import ttnn
import ttml

from ttml.parallel import placement, sum_partials, whole

from .module_base import AbstractModuleBase
from .parameter import Parameter


def feature_shard(x, cluster_axis: int):
    """The input of a row-parallel matmul: this rank's slice of the features, sliced here from a replicated input."""
    if isinstance(placement(x, cluster_axis), ttnn.PlacementReplicate):
        return ttml.ops.distributed.scatter(x, 3, cluster_axis)
    return x


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
        ttml.autograd.assign_numpy(self.weight.tensor, state["weight"])
        if state["bias"] is not None:
            if self.bias is None:
                raise ValueError("LinearLayer bias was improperly initialized when deserializing from Pickle")
            ttml.autograd.assign_numpy(self.bias.tensor, state["bias"])

    def forward(self, x):
        """Compute linear projection of x."""
        bias_t = self.bias.tensor if self.bias is not None else None
        return ttml.ops.linear.linear(x, self.weight.tensor, bias_t)


class ColumnParallelLinear(AbstractModuleBase):
    """Column-parallel linear layer.

    The weight matrix ``(out_features, in_features)`` is sharded along the
    *output-feature* (row) dimension — i.e. dim 2 in the 4-D layout
    ``(1, 1, out_features, in_features)`` — so each TP device holds
    ``out_features // tp_size`` rows.  The bias is similarly sharded on dim 3.

    The input is made whole on every rank (:func:`ttml.parallel.whole`): gathered if it arrives
    sharded across the TP axis (along the sequence, under sequence parallelism), used as is if replicated.

    Args:
        gather_output: If ``True`` an all-gather is inserted after the matmul so
            the output is replicated across TP devices (needed when the consumer
            expects an un-sharded tensor, e.g. the final LM-head projection).
        axis_name: Mesh axis used for tensor parallelism.
    """

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
        self.cluster_axis = ttml.mesh().axis_index(axis_name)

        if weight_init is None:
            k = math.sqrt(1.0 / in_features)
            weight_init = ttml.init.uniform(-k, k)
        if bias_init is None:
            k = math.sqrt(1.0 / in_features)
            bias_init = ttml.init.uniform(-k, k)

        weight_shape = (1, 1, out_features, in_features)
        mesh = ttml.mesh()
        weight_mapper = mesh.axis_mapper(axis_name, tdim=2)
        self.weight = Parameter(weight_init(weight_shape, mapper=weight_mapper))

        if has_bias:
            bias_shape = (1, 1, 1, out_features)
            bias_mapper = mesh.axis_mapper(axis_name, tdim=3)
            self.bias = Parameter(bias_init(bias_shape, mapper=bias_mapper))
        else:
            self.bias = None

    def forward(self, x):
        x = whole(x, self.cluster_axis)
        bias_t = self.bias.tensor if self.bias is not None else None
        x = ttml.ops.linear.linear(x, self.weight.tensor, bias_t)
        if self.gather_output:
            # Reconstitute the full output across TP devices (dim 3 = last feature dim).
            x = ttml.ops.distributed.all_gather(x, 3, self.cluster_axis)
        return x


class RowParallelLinear(AbstractModuleBase):
    """Row-parallel linear layer.

    The weight matrix ``(out_features, in_features)`` is sharded along the
    *input-feature* (column) dimension — i.e. dim 3 in the 4-D layout
    ``(1, 1, out_features, in_features)`` — so each TP device holds
    ``in_features // tp_size`` columns.  The bias is replicated (not sharded)
    because it is added after the partial products are summed.

    The input is this rank's slice of the features, as a ``ColumnParallelLinear`` without
    ``gather_output`` leaves it; a replicated input is sliced here. The partial products are summed
    into the layout of ``like`` on the TP axis (:func:`ttml.parallel.sum_partials`): pass the tensor
    the region started from, so the result returns in that layout (sharded along the sequence under
    sequence parallelism). Without ``like`` the result is replicated.

    Args:
        axis_name: Mesh axis used for tensor parallelism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
        axis_name: str = "tp",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.axis_name = axis_name
        self.cluster_axis = ttml.mesh().axis_index(axis_name)

        if weight_init is None:
            k = math.sqrt(1.0 / in_features)
            weight_init = ttml.init.uniform(-k, k)
        if bias_init is None:
            k = math.sqrt(1.0 / in_features)
            bias_init = ttml.init.uniform(-k, k)

        weight_shape = (1, 1, out_features, in_features)
        weight_mapper = ttml.mesh().axis_mapper(axis_name, tdim=3)
        self.weight = Parameter(weight_init(weight_shape, mapper=weight_mapper))

        if has_bias:
            bias_shape = (1, 1, 1, out_features)
            self.bias = Parameter(bias_init(bias_shape))
        else:
            self.bias = None

    def forward(self, x, like=None):
        x = feature_shard(x, self.cluster_axis)
        x = ttml.ops.linear.linear(x, self.weight.tensor, None)
        x = sum_partials(x, self.cluster_axis, like)
        if self.bias is not None:
            x = ttml.ops.binary.add(x, self.bias.tensor)
        return x
