# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Embedding layer for ttml models."""

from __future__ import annotations

from typing import Callable

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
    """Vocab-parallel embedding (Megatron-LM style).

    The weight matrix ``(num_embeddings, embedding_dim)`` is sharded along the
    *vocabulary* dimension — i.e. dim 2 in the 4-D layout
    ``(1, 1, num_embeddings, embedding_dim)`` — so each TP device holds
    ``num_embeddings // tp_size`` rows.  The sharded layout matches
    ``ColumnParallelLinear``'s output-dim sharding, so the same tensor can
    back both the input embedding and a tied LM-head weight.

    Forward: each device looks up rows in its local vocab shard, zeroes the
    hidden vectors for out-of-range ids, and all-reduces (sum) across the TP
    axis.  Exactly one shard owns any given id, so the sum equals the true
    embedding row.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        weight_init: Callable | None = None,
        axis_name: str = "tp",
    ) -> None:
        """Initialize vocab-parallel embedding layer.

        Args:
            num_embeddings: Full (padded) vocabulary size; must be divisible by
                the TP axis size.
            embedding_dim: Hidden dimension.
            weight_init: Initializer for the weight tensor.  Defaults to
                ``normal(0, 0.02)``.
            axis_name: Mesh axis used for tensor parallelism.
        """
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.axis_name = axis_name
        self.cluster_axis = ttml.mesh().axis_index(axis_name)
        self.tp_size = ttml.mesh().axis_size(axis_name)

        if num_embeddings % self.tp_size != 0:
            raise ValueError(
                f"num_embeddings ({num_embeddings}) must be divisible by "
                f"TP size ({self.tp_size}) for vocab-parallel sharding."
            )
        self.local_num_embeddings = num_embeddings // self.tp_size

        if weight_init is None:
            weight_init = ttml.init.normal(0.0, 0.02)

        weight_shape = (1, 1, num_embeddings, embedding_dim)
        weight_mapper = ttml.mesh().axis_mapper(axis_name, tdim=2)
        self.weight = Parameter(weight_init(weight_shape, mapper=weight_mapper))

    def forward(self, input_ids) -> ttml.autograd.Tensor:
        local_V = self.local_num_embeddings

        if isinstance(input_ids, np.ndarray):
            ids_np = input_ids.astype(np.int64)
        else:
            composer = None
            topology = input_ids.get_value().tensor_topology()
            for p in tuple(topology.placements()):
                if isinstance(p, ttnn.PlacementShard):
                    dev = ttml.autograd.AutoContext.get_instance().get_device()
                    composer = ttml.core.distributed.concat_mesh_to_tensor_composer(dev, p.dim)
                    break
            ids_np = input_ids.to_numpy(ttnn.DataType.UINT32, composer=composer).astype(np.int64)

        ids_np = ids_np.reshape(ids_np.shape[0], -1)  # (B, T)
        B, T = ids_np.shape

        # Stack per-rank local ids and ownership masks along dim 0, then shard
        # dim 0 across TP so rank k receives the slice [k*B : (k+1)*B].
        all_local_ids = np.zeros((self.tp_size * B, 1, 1, T), dtype=np.uint32)
        all_ownership_mask = np.zeros((self.tp_size * B, 1, T, 1), dtype=np.float32)

        for k in range(self.tp_size):
            shard_ids = ids_np - (k * local_V)
            owned = (shard_ids >= 0) & (shard_ids < local_V)
            local_ids = np.clip(shard_ids, 0, local_V - 1)
            all_local_ids[k * B : (k + 1) * B, 0, 0, :] = local_ids.astype(np.uint32)
            all_ownership_mask[k * B : (k + 1) * B, 0, :, 0] = owned.astype(np.float32)

        mapper = ttml.mesh().axis_mapper(self.axis_name, tdim=0)
        local_ids_t = ttml.autograd.Tensor.from_numpy(
            all_local_ids, ttnn.Layout.ROW_MAJOR, ttnn.DataType.UINT32, mapper
        )
        ownership_mask_t = ttml.autograd.Tensor.from_numpy(all_ownership_mask, ttnn.Layout.TILE, ttnn.bfloat16, mapper)

        h = ttml.ops.embedding.embedding(local_ids_t, self.weight.tensor)
        h = ttml.ops.binary.mul(h, ownership_mask_t)
        h = ttml.ops.distributed.all_reduce(h, True, self.cluster_axis)
        return h
