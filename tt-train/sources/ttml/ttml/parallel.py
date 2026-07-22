# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel strategy for TTML models."""

from __future__ import annotations

from enum import Enum


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
    def from_flags(cls, enable_tp: bool, enable_sp: bool) -> "TPStrategy":
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
