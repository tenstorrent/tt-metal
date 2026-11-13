# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable
import math

import ttnn
import ttml

from ttml.parallel import SEQUENCE_DIM, mark_partial

from .module_base import AbstractModuleBase
from .parameter import Parameter


# A tensor-parallel region runs from a column-parallel matmul to a row-parallel one. How a tensor is laid out is read
# from its label, but a label cannot say that a tensor is a partial sum still to be added up across ranks: a placement
# is only Replicate or Shard (https://github.com/tenstorrent/tt-metal/issues/62347). The row-parallel matmul's output is
# such a sum, and so is the gradient a column-parallel matmul sends back to its input. So entering and leaving the
# region choose the collectives that add them up.


def _input_layout_error(layer: str, x, expected: str, cluster_axis: int) -> ValueError:
    return ValueError(
        f"{layer} takes its input {expected} across mesh axis {ttml.mesh().axis_names[cluster_axis]!r}, but the input "
        f"is laid out as {ttml.Sharding.from_tensor(x).describe()}"
    )


def enter_tp_region(x, cluster_axis: int, sequence_parallel: bool):
    expected = SEQUENCE_DIM if sequence_parallel else None
    if ttml.mesh().shape[cluster_axis] > 1 and ttml.sharding.shard_dim(x, cluster_axis) != expected:
        layout = "sharded along the sequence" if sequence_parallel else "replicated"
        raise _input_layout_error("A column-parallel layer", x, layout, cluster_axis)
    if sequence_parallel:
        return ttml.ops.distributed.all_gather(
            x, SEQUENCE_DIM, cluster_axis, ttml.ops.distributed.GradOutputType.SHARDED
        )
    return ttml.ops.distributed.broadcast(x, cluster_axis)


def shard_features(x, cluster_axis: int):
    if ttml.mesh().shape[cluster_axis] == 1:
        return x
    feature_dim = len(x.shape()) - 1
    dim = ttml.sharding.shard_dim(x, cluster_axis)
    if dim is None:
        return ttml.ops.distributed.scatter(x, feature_dim, cluster_axis)
    if dim != feature_dim:
        raise _input_layout_error("A row-parallel layer", x, "replicated or sharded along the features", cluster_axis)
    return x


def exit_tp_region(x, cluster_axis: int, sequence_parallel: bool):
    """Sum a tensor-parallel region's partial products across TP ranks."""
    if sequence_parallel:
        return ttml.ops.distributed.reduce_scatter(x, SEQUENCE_DIM, cluster_axis)
    return ttml.ops.distributed.all_reduce(x, noop_backward=True, cluster_axis=cluster_axis)


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

    Args:
        gather_output: If ``True`` an all-gather is inserted after the matmul so
            the output is replicated across TP devices (needed when the consumer
            expects an un-sharded tensor, e.g. the final LM-head projection).
        sequence_parallel: If ``True`` the input must arrive sharded along the sequence
            dimension across the TP axis (Megatron sequence parallelism), and an
            all-gather over the sequence dim reconstructs the full sequence before
            the matmul. Otherwise the input must be replicated.
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
        x = enter_tp_region(x, self.cluster_axis, self.sequence_parallel)
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

    It takes its input sharded along the features, or replicated.

    Args:
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
        sequence_parallel: bool = False,
        axis_name: str = "tp",
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
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
                # Added after the reduce-scatter, it sees only this rank's tokens.
                mark_partial(self.bias, axis_name)
        else:
            self.bias = None

    def forward(self, x):
        x = shard_features(x, self.cluster_axis)
        x = ttml.ops.linear.linear(x, self.weight.tensor, None)
        x = exit_tp_region(x, self.cluster_axis, self.sequence_parallel)
        if self.bias is not None:
            x = ttml.ops.binary.add(x, self.bias.tensor)
        return x
