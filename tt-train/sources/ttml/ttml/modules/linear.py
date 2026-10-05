# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import math

import ml_dtypes

import ttnn
import ttml

from ttml.parallel import SEQUENCE_DIM, mark_sequence_parallel

from .module_base import AbstractModuleBase
from .parameter import Parameter


def column_parallel_input(x, cluster_axis: int, sequence_parallel: bool):
    """The input of a column-parallel matmul: the full sequence, replicated on every TP rank."""
    if sequence_parallel:
        return ttml.ops.distributed.all_gather(
            x, SEQUENCE_DIM, cluster_axis, ttml.ops.distributed.GradOutputType.SHARDED
        )
    return ttml.ops.distributed.broadcast(x, cluster_axis)


def row_parallel_output(x, cluster_axis: int, sequence_parallel: bool, input_is_parallel: bool):
    """The output of a row-parallel matmul: the partial products summed across TP ranks."""
    if sequence_parallel:
        return ttml.ops.distributed.reduce_scatter(x, SEQUENCE_DIM, cluster_axis)
    return ttml.ops.distributed.all_reduce(x, input_is_parallel, cluster_axis)


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
    """Column-parallel linear layer.

    The weight matrix ``(out_features, in_features)`` is sharded along the
    *output-feature* (row) dimension — i.e. dim 2 in the 4-D layout
    ``(1, 1, out_features, in_features)`` — so each TP device holds
    ``out_features // tp_size`` rows.  The bias is similarly sharded on dim 3.

    Args:
        gather_output: If ``True`` an all-gather is inserted after the matmul so
            the output is replicated across TP devices (needed when the consumer
            expects an un-sharded tensor, e.g. the final LM-head projection).
        sequence_parallel: If ``True`` the input arrives sharded along the sequence
            dimension across the TP axis (Megatron sequence parallelism). An
            all-gather over the sequence dim reconstructs the full sequence before
            the matmul.
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
        sequence_parallel: bool = False,
        axis_name: str = "tp",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel
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
        x = column_parallel_input(x, self.cluster_axis, self.sequence_parallel)
        bias_t = self.bias.tensor if self.bias is not None else None
        x = ttml.ops.linear.linear(x, self.weight.tensor, bias_t)
        if self.gather_output:
            # Reconstitute the full output across TP devices (dim 3 = last feature dim).
            x = ttml.ops.distributed.all_gather(x, 3, self.cluster_axis, ttml.ops.distributed.GradOutputType.REPLICATED)
        return x


class RowParallelLinear(AbstractModuleBase):
    """Row-parallel linear layer.

    The weight matrix ``(out_features, in_features)`` is sharded along the
    *input-feature* (column) dimension — i.e. dim 3 in the 4-D layout
    ``(1, 1, out_features, in_features)`` — so each TP device holds
    ``in_features // tp_size`` columns.  The bias is replicated (not sharded)
    because it is added after the all-reduce that sums partial products.

    Args:
        input_is_parallel: Set to ``True`` when the incoming activation is already
            sharded along the feature dimension (e.g. the output of a
            ``ColumnParallelLinear`` with ``gather_output=False``).  When ``False``
            a scatter is inserted to partition the input.
        sequence_parallel: If ``True`` the partial matmul outputs are combined with a
            sequence reduce-scatter (Megatron sequence parallelism) instead of an
            all-reduce: this sums the partial products across TP *and* shards the
            result along the sequence, leaving the residual stream sequence-sharded.
        axis_name: Mesh axis used for tensor parallelism.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        has_bias: bool = True,
        weight_init: Callable | None = None,
        bias_init: Callable | None = None,
        input_is_parallel: bool = False,
        sequence_parallel: bool = False,
        axis_name: str = "tp",
    ) -> None:
        super().__init__()

        if sequence_parallel and not input_is_parallel:
            # SP RowParallel always follows an SP ColumnParallel, whose output is
            # already feature-sharded. A non-parallel (replicated) SP input would
            # need a simultaneous sequence-gather + feature-scatter that Llama never
            # exercises; reject it rather than emit a subtly wrong graph.
            raise ValueError("RowParallelLinear: sequence_parallel requires input_is_parallel=True")

        self.in_features = in_features
        self.out_features = out_features
        # TODO: use ttnn.Tensor.tensor_topology() once CCL ops will properly update this property
        self.input_is_parallel = input_is_parallel
        self.sequence_parallel = sequence_parallel
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
            if sequence_parallel:
                mark_sequence_parallel(self.bias)
        else:
            self.bias = None

    def forward(self, x):
        if not self.input_is_parallel:
            # Split the input along the feature dimension across TP devices.
            x = ttml.ops.distributed.scatter(x, 3, self.cluster_axis)
        x = ttml.ops.linear.linear(x, self.weight.tensor, None)
        x = row_parallel_output(x, self.cluster_axis, self.sequence_parallel, self.input_is_parallel)
        if self.bias is not None:
            x = ttml.ops.binary.add(x, self.bias.tensor)
        return x
