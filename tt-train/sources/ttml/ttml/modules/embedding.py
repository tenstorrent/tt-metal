# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding layer for ttml models."""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

import ttnn
import ttml

from .module_base import AbstractModuleBase
from .parameter import Parameter


class Embedding(AbstractModuleBase):
    """Embedding layer implemented in Python using ttml operations."""

    def __init__(self, num_embeddings: int, embedding_dim: int, weight_init=None) -> None:
        """Initialize embedding layer.

        Args:
            num_embeddings: Size of vocabulary
            embedding_dim: Dimension of embeddings
            weight_init: Initializer for weight tensor. Defaults to normal(0, 0.02).
        """
        super().__init__()

        if weight_init is None:
            weight_init = ttml.init.normal(0.0, 0.02)

        weight_shape = (1, 1, num_embeddings, embedding_dim)
        self.weight = Parameter(weight_init(weight_shape))

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of embedding layer.

        Args:
            x: Input tensor of token indices, shape [batch_size, 1, 1, seq_len]

        Returns:
            Output tensor of embeddings
        """
        return ttml.ops.embedding.embedding(x, self.weight.tensor)


class VocabParallelEmbedding(AbstractModuleBase):
    """Embedding whose table is sharded along the vocabulary dimension.

    Each device shifts the ids into its local window,
    looks up the ids that fall in its slice, zeroes the rest, and an all-reduce
    then sums the per-device contributions — each token is owned by exactly one
    device, so the sum reconstructs the full, replicated embedding.

    Args:
        num_embeddings: Vocabulary size. Must be divisible by ``tp_size``.
        embedding_dim: Dimension of embeddings.
        weight_init: Initializer for the weight tensor. Defaults to normal(0, 0.02).
        axis_name: Mesh axis used for tensor parallelism.

    Note:
        ``num_embeddings`` need only be divisible by ``tp_size`` — the per-device
        shard is not required to be tile-aligned. The embedding *backward* kernel,
        however, requires ``seq_len`` and ``embedding_dim`` to be multiples of 32,
        so for training callers must tile-align the sequence length (the model
        forward paths already pad it).
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_init: Optional[Callable] = None,
        axis_name: str = "tp",
    ) -> None:
        super().__init__()

        mesh = ttml.mesh()
        self.axis_name = axis_name
        self.cluster_axis = mesh.axis_index(axis_name)
        self.tp_size = mesh.axis_size(axis_name)

        if num_embeddings % self.tp_size != 0:
            raise ValueError(
                f"num_embeddings ({num_embeddings}) must be divisible by the tensor-parallel "
                f"size ({self.tp_size}) of mesh axis '{axis_name}'."
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_embeddings_per_partition = num_embeddings // self.tp_size

        if weight_init is None:
            weight_init = ttml.init.normal(0.0, 0.02)

        weight_shape = (1, 1, num_embeddings, embedding_dim)
        weight_mapper = mesh.axis_mapper(axis_name, tdim=2)
        self.weight = Parameter(weight_init(weight_shape, mapper=weight_mapper))

        # Per-device vocab-start offsets (fp32), one scalar shard per TP rank.
        starts_np = (np.arange(self.tp_size, dtype=np.float32) * self.num_embeddings_per_partition).reshape(
            1, 1, self.tp_size, 1
        )
        self._vocab_start = ttml.autograd.Tensor.from_numpy(
            starts_np, ttnn.Layout.TILE, ttnn.DataType.FLOAT32, weight_mapper
        ).get_value(ttml.autograd.PreferredPrecision.FULL)

    def _check_tp_replicated(self, x: ttml.autograd.Tensor) -> None:
        """Reject ids that are sharded along the TP axis.

        The per-rank offset masking is only correct when every TP device sees the
        full ids, so a shard on the TP axis silently produces wrong embeddings.
        """
        placements = ttml.Sharding.from_tensor(x).placements
        if placements is None or self.cluster_axis >= len(placements):
            return
        p = placements[self.cluster_axis]
        if isinstance(p, ttnn.PlacementShard):
            raise ValueError(
                f"VocabParallelEmbedding expects ids replicated across TP axis '{self.axis_name}', "
                f"got PlacementShard(dim={p.dim}). Each TP device must see the full ids for the "
                f"offset masking to be correct."
            )

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Forward pass of the vocab-parallel embedding.

        Args:
            x: Token indices, shape ``[batch_size, 1, 1, seq_len]`` (uint32,
                row-major), replicated across the TP axis.

        Returns:
            Embeddings ``[batch_size, 1, seq_len, embedding_dim]``, replicated
            across the TP axis.
        """
        self._check_tp_replicated(x)

        ids = x.get_value(ttml.autograd.PreferredPrecision.FULL)
        ids_f = ttnn.typecast(ttnn.to_layout(ids, ttnn.Layout.TILE), ttnn.DataType.FLOAT32)

        local = ttnn.subtract(ids_f, self._vocab_start)
        in_range = ttnn.multiply(
            ttnn.ge(local, 0.0),
            ttnn.lt(local, float(self.num_embeddings_per_partition)),
        )

        # Out-of-range ids still need an in-bounds row for the gather: the output
        # mask nulls their value but can't undo an out-of-bounds lookup. where()
        # selects, so in-range ids pass through unperturbed.
        local = ttnn.where(in_range, local, 0.0)
        # +0.5 makes the truncating uint32 cast round to nearest, so any sub-0.5
        # fp drift in `local` snaps back to the intended row.
        local_ids = ttnn.to_layout(ttnn.typecast(ttnn.add(local, 0.5), ttnn.DataType.UINT32), ttnn.Layout.ROW_MAJOR)

        emb = ttml.ops.embedding.embedding(
            ttml.autograd.create_tensor(local_ids, requires_grad=False), self.weight.tensor
        )

        # Zero the rows this device doesn't own before the sum; this also zeroes
        # their backward grad, so nothing accumulates onto row 0. Mask stays fp32 —
        # ttnn.multiply mixes it with the bf16 embedding and outputs bf16.
        mask = ttnn.transpose(in_range, 2, 3)
        emb = ttml.ops.binary.mul(emb, ttml.autograd.create_tensor(mask, requires_grad=False))

        # Each token is nonzero on exactly one device; sum reconstructs the full
        # embedding, replicated across TP. Input ids carry no grad, so backward
        # simply passes the (already replicated) output grad through unchanged.
        return ttml.ops.distributed.all_reduce(emb, noop_backward=True, cluster_axis=self.cluster_axis)


class FeatureParallelEmbedding(AbstractModuleBase):
    """Embedding whose table is sharded along the embedding (hidden) dimension.

    Each device holds the *entire* vocabulary but only an ``embedding_dim //
    tp_size`` slice of every row. Versus :class:`VocabParallelEmbedding` this
    makes the lookup fully local — no vocab offset, range mask, or out-of-range
    handling — and replaces the full-hidden all-reduce with a single all-gather
    over the embedding dim (roughly half the bandwidth), mirroring
    ``ColumnParallelLinear(gather_output=True)``. The trade-off: its layout does
    not match the vocab-parallel LM head, so weight tying is unavailable.

    Args:
        num_embeddings: Vocabulary size (rows). Held in full on every device.
        embedding_dim: Embedding width. Sharded across ``tp_size``.
        weight_init: Initializer for the weight tensor. Defaults to normal(0, 0.02).
        axis_name: Mesh axis used for tensor parallelism.
        gather_output: If ``True`` (default) an all-gather reconstructs the full
            TP-replicated embedding. If ``False`` the hidden-sharded slice
            ``[batch, 1, seq, embedding_dim/tp]`` is returned for a TP-aware consumer.

    Note:
        The shard must be tile-width aligned: ``embedding_dim`` divisible by
        ``tp_size`` and ``embedding_dim // tp_size`` a multiple of 32 (the
        embedding kernel's last-dim requirement). Backward also needs ``seq_len``
        tile-aligned.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_init: Optional[Callable] = None,
        axis_name: str = "tp",
        gather_output: bool = True,
    ) -> None:
        super().__init__()

        mesh = ttml.mesh()
        self.axis_name = axis_name
        self.cluster_axis = mesh.axis_index(axis_name)
        self.tp_size = mesh.axis_size(axis_name)
        self.gather_output = gather_output

        if embedding_dim % self.tp_size != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by the tensor-parallel "
                f"size ({self.tp_size}) of mesh axis '{axis_name}'."
            )
        if (embedding_dim // self.tp_size) % 32 != 0:
            raise ValueError(
                f"embedding_dim per partition ({embedding_dim // self.tp_size} = {embedding_dim} / "
                f"{self.tp_size}) must be a multiple of 32 for the embedding kernel."
            )

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_dim_per_partition = embedding_dim // self.tp_size

        if weight_init is None:
            weight_init = ttml.init.normal(0.0, 0.02)

        weight_shape = (1, 1, num_embeddings, embedding_dim)
        weight_mapper = mesh.axis_mapper(axis_name, tdim=3)
        self.weight = Parameter(weight_init(weight_shape, mapper=weight_mapper))

    def _check_tp_replicated(self, x: ttml.autograd.Tensor) -> None:
        """Reject ids sharded along the TP axis.

        Every device holds a different embedding-dim slice of the *full* vocab,
        so each needs the complete ids; a TP shard would silently drop tokens.
        """
        placements = ttml.Sharding.from_tensor(x).placements
        if placements is None or self.cluster_axis >= len(placements):
            return
        p = placements[self.cluster_axis]
        if isinstance(p, ttnn.PlacementShard):
            raise ValueError(
                f"FeatureParallelEmbedding expects ids replicated across TP axis "
                f"'{self.axis_name}', got PlacementShard(dim={p.dim}). Each TP device must see "
                f"the full ids to look up its embedding-dim slice for every token."
            )

    def forward(self, x: ttml.autograd.Tensor) -> ttml.autograd.Tensor:
        """Look up the local hidden-dim shard for every id.

        Args:
            x: Token indices ``[batch_size, 1, 1, seq_len]``, replicated across TP.

        Returns:
            ``gather_output`` (default): full embedding
            ``[batch_size, 1, seq_len, embedding_dim]`` replicated across TP.
            Otherwise the hidden-sharded slice
            ``[batch_size, 1, seq_len, embedding_dim/tp]``.
        """
        self._check_tp_replicated(x)

        # Every id is valid on every device, so this is a plain local lookup on
        # the hidden-dim shard: output [batch, 1, seq, embedding_dim/tp] (dim-3 sharded).
        emb = ttml.ops.embedding.embedding(x, self.weight.tensor)

        if not self.gather_output:
            return emb

        # All-gather the per-device hidden slices into the full replicated embedding.
        # GradOutputType.REPLICATED divides the backward reduce_scatter by tp_size
        # (output is replicated, consumed replicated), matching
        # ColumnParallelLinear(gather_output=True).
        return ttml.ops.distributed.all_gather(
            emb,
            3,
            self.cluster_axis,
            ttml.ops.distributed.GradOutputType.REPLICATED,
        )
