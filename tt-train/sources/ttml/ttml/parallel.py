# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel strategy for TTML models, and the marker for parameters with partial gradients."""

from __future__ import annotations

from enum import Enum
from typing import Any

from ttml.modules.module_base import AbstractModuleBase
from ttml.modules.parameter import Parameter

# Activations flow as 4-D ``(B, 1, S, H)`` tensors; sequence parallelism shards dim 2 across the tp axis.
SEQUENCE_DIM = 2


class TPStrategy(Enum):
    """How tensor parallelism is applied to a model.

    - ``NONE``            -- no tensor parallelism (single device / DP / FSDP only).
    - ``TENSOR``          -- Megatron tensor parallelism: activations are replicated across
      the ``tp`` axis between the sharded matmuls (all-reduce / broadcast).
    - ``TENSOR_SEQUENCE`` -- tensor parallelism **plus** Megatron sequence parallelism: the
      residual stream is sharded along the sequence across the ``tp`` axis in the
      norm/dropout/residual regions (reduce-scatter / all-gather). Requires a ``tp`` axis.
    """

    NONE = "none"
    TENSOR = "tensor"
    TENSOR_SEQUENCE = "tensor_sequence"

    @classmethod
    def from_flags(cls, enable_tp: bool, *, enable_sp: bool = False) -> "TPStrategy":
        """Map the ``(enable_tp, enable_sp)`` device-config flags to a strategy.

        Sequence parallelism shards the tensor-parallel residual stream, so it requires
        tensor parallelism; ``enable_sp`` without ``enable_tp`` is rejected here.
        """
        if enable_sp and not enable_tp:
            raise ValueError("enable_sp (sequence parallelism) requires enable_tp (tensor parallelism)")
        if not enable_tp:
            return cls.NONE
        return cls.TENSOR_SEQUENCE if enable_sp else cls.TENSOR

    @property
    def tensor_parallel(self) -> bool:
        """True if any tensor parallelism is active (``TENSOR`` or ``TENSOR_SEQUENCE``)."""
        return self is not TPStrategy.NONE

    @property
    def sequence_parallel(self) -> bool:
        """True if Megatron sequence parallelism is active (``TENSOR_SEQUENCE``)."""
        return self is TPStrategy.TENSOR_SEQUENCE


def mark_partial(target: Parameter | AbstractModuleBase, axis_name: str) -> None:
    """Mark a parameter, or every parameter of a module, as having a partial gradient across mesh axis ``axis_name``.

    Such a parameter is replicated across the axis, but each rank sees only part of the data.
    Each rank's gradient is then a partial sum, and :func:`ttml.sync_gradients` adds them up. The mark says what the
    gradient's label cannot, since a placement is only Replicate or Shard
    (https://github.com/tenstorrent/tt-metal/issues/62347).
    Mark a module only if all its parameters are replicated on the axis.
    """
    if isinstance(target, Parameter):
        target.add_post_materialize_callback(lambda p: _add_partial_axis(p.tensor, axis_name))
        return
    for _, module in target.named_modules():
        if isinstance(module, AbstractModuleBase):
            for value in vars(module).values():
                if isinstance(value, Parameter):
                    mark_partial(value, axis_name)


def partial_axes(param_tensor: Any) -> frozenset[str]:
    """The names of the mesh axes :func:`mark_partial` marked this parameter tensor with."""
    return getattr(param_tensor, "_partial_axes", frozenset())


def _add_partial_axis(param_tensor: Any, axis_name: str) -> None:
    param_tensor._partial_axes = partial_axes(param_tensor) | {axis_name}
