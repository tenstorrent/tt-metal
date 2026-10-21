# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tensor parallelism: the strategy a model is built with, and the collectives at the edges of a tensor-parallel region.

A tensor-parallel region computes locally with weights sharded across one mesh axis. Its input is made whole on every
rank of the axis (:func:`whole`), whatever layout it arrives in. Its result starts as each rank's partial sum, which
:func:`sum_partials` adds up into the layout the model keeps between regions: replicated for tensor parallelism,
sharded along the sequence for sequence parallelism. The layout is read off the tensors, so a model chooses it once,
where the residual stream starts, and every region follows.
"""

from __future__ import annotations

from enum import Enum

import ttnn
import ttml

# Activations flow as 4-D ``(B, 1, S, H)`` tensors; sequence parallelism shards dim 2 across the tp axis.
SEQUENCE_DIM = 2


def placement(tensor, cluster_axis: int):
    """``tensor``'s placement on mesh axis ``cluster_axis``."""
    return ttml.Sharding.from_tensor(tensor).placements[cluster_axis]


def whole(x, cluster_axis: int):
    """``x`` whole on every rank of ``cluster_axis``: gathered if sharded there, as is if replicated.

    In the backward pass the gradient arrives as each rank's partial sum (the region's sharded weights each see part
    of it), and both ways sum it.
    """
    p = placement(x, cluster_axis)
    if isinstance(p, ttnn.PlacementShard):
        return ttml.ops.distributed.all_gather(x, p.dim, cluster_axis)
    return ttml.ops.distributed.broadcast(x, cluster_axis)


def cross_entropy_loss(logits, targets, reduce=None):
    """Cross-entropy of ``logits`` against ``targets``, computed across the mesh axis the logits' vocabulary (last
    dim) is sharded on, if any: the softmax then normalizes over the whole vocabulary, not each rank's slice."""
    if reduce is None:
        reduce = ttml.ops.ReduceType.MEAN
    sharding = ttml.Sharding.from_tensor(logits)
    vocab_dim = len(logits.shape()) - 1
    axes = [axis for axis in range(len(sharding.placements)) if sharding.shard_dim(axis) == vocab_dim]
    if not axes:
        return ttml.ops.loss.cross_entropy_loss(logits, targets, reduce)
    if len(axes) > 1:
        raise ValueError(f"the logits' vocabulary is sharded on several mesh axes {axes}; expected at most one")
    return ttml.ops.distributed.vocab_parallel_cross_entropy_loss(logits, targets, cluster_axis=axes[0], reduce=reduce)


def sum_partials(partial, cluster_axis: int, like=None):
    """Add up the ranks' partial results across ``cluster_axis`` into the layout of ``like`` there: a placement, or a
    tensor whose placement to match. Replicated by default."""
    if like is not None and not isinstance(like, (ttnn.PlacementShard, ttnn.PlacementReplicate)):
        like = placement(like, cluster_axis)
    if isinstance(like, ttnn.PlacementShard):
        return ttml.ops.distributed.reduce_scatter(partial, like.dim, cluster_axis)
    return ttml.ops.distributed.all_reduce(partial, cluster_axis)


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
