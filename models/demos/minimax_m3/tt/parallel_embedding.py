# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M3 parallel (sharded) token embedding.

The token-embedding table ``[vocab, emb_dim]`` is sharded on ``emb_dim`` across the TP axis and
replicated across the SP axis, so each device stores only ``[vocab, emb_dim / tp]`` — a pure memory
win (~1.85 GB/device vs the old fully-replicated table; the weight, not the activation, shrinks).

Forward:
  * input  ``tokens``  [.., s_local] uint32, SP-sharded by the caller (row r holds its seq shard),
    replicated across the TP cols;
  * ``ttnn.embedding`` on the local emb-dim slice -> ``[1, 1, s_local, emb_dim / tp]`` (TP-sharded on
    hidden, no CCL: each TP col looks up its own slice);
  * a MANAGED TP all-gather (``mesh_config.allgather``) reconstructs the full ``emb_dim`` replicated
    across the TP cols -> ``[1, 1, s_local, emb_dim]``, exactly the TP-replicated residual stream the
    decoder layers expect. (This is M3-specific: DeepSeek's residual is TP-sharded, so its
    ``TtParallelEmbedding`` needs no gather; M3's is TP-replicated, so we add one — cheap, once per
    chunk, not per layer.)

Weight caching mirrors every other M3 weight: a per-tensor tilized ``.tensorbin`` via
``ttnn.as_tensor(cache_file_name=)``. On a cache hit ``torch_weight`` is ignored (may be ``None``),
so cache-only runs never read the bf16 source. The cache key differs from the old replicated
``model.embed_tokens.weight`` because the stored per-device layout is now sharded.
"""

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

# Cache key for the SHARDED embed table. Deliberately distinct from "model.embed_tokens.weight"
# (the old replicated layout) so a stale replicated .tensorbin is never loaded as if it were sharded.
EMBED_CACHE_NAME = "model.embed_tokens.weight_parallel"


class TtParallelEmbedding(LightweightModule):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        vocab_size: int,
        emb_dim: int,
        mesh_config,
        ccl_manager,
        torch_weight: Optional[torch.Tensor] = None,
        cache_file_name: Optional[str] = None,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.tp_axis = mesh_config.tp_axis
        self.dtype = dtype

        tp = mesh_device.shape[self.tp_axis]
        assert emb_dim % tp == 0, f"emb_dim ({emb_dim}) must be divisible by tp ({tp})"

        if torch_weight is not None:
            # Accept the HF table with any number of leading singleton dims ([1, 1, vocab, emb]).
            torch_weight = torch_weight.reshape(vocab_size, emb_dim)

        # Shard emb_dim across the TP axis; replicate across the SP axis.
        shard_dims = [None, None]
        shard_dims[self.tp_axis] = -1
        mesh_mapper = ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=mesh_device.shape, dims=tuple(shard_dims))

        self.weight = ttnn.as_tensor(
            torch_weight,  # ignored on a cache hit; must be present to populate the cache
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
            cache_file_name=cache_file_name,
        )

    def forward(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """tokens: SP-sharded (replicated across TP) uint32 indices -> [1, 1, s_local, emb_dim] bf16,
        full hidden replicated across the TP cols (the M3 residual-stream contract)."""
        emb = ttnn.embedding(tokens, self.weight, layout=ttnn.TILE_LAYOUT, dtype=self.dtype)
        if len(emb.shape) == 3:
            emb = ttnn.unsqueeze_to_4D(emb)

        tp = self.mesh_device.shape[self.tp_axis]
        if tp > 1:
            emb = self.mesh_config.allgather(emb, self.ccl_manager, axis=self.tp_axis, dim=3)
        return emb
