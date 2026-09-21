# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Token embedding, sharded on the hidden dim across TP.

Two sharding modes exist in the sibling packages: **emb-on-TP** (the vocab replicated, hidden
split, one all-gather after the lookup) and **vocab-on-SP** (each row owns a vocab slice, a masked
lookup and an SP all-reduce). This uses emb-on-TP, and the reason is capacity, not elegance: the
table is 248320 x 5120, which is 2.5 GB in bf16 — 635 MB per chip at tp=4, comfortable on a
Blackhole's DRAM. Vocab sharding would cut that eightfold and cost a masked gather plus a
collective; it is worth doing when the table stops fitting, and not before.

Sequence parallelism needs nothing here: each chip is handed only its own token block, so the
lookup is local and the output is already SP-sharded.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..config import MeshConfig
from ..utils.general_utils import get_cache_file_name


class ParallelEmbedding(LightweightModule):
    def __init__(
        self,
        mesh_device,
        vocab_size: int,
        hidden_size: int,
        state_dict: dict[str, torch.Tensor],
        *,
        mesh_config: MeshConfig,
        ccl_manager,
        dtype=ttnn.bfloat16,
        tensor_cache_path: Optional[str] = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        assert hidden_size % (mesh_config.tp * ttnn.TILE_SIZE) == 0

        weight = state_dict["weight"] if state_dict else None
        self.weight = ttnn.as_tensor(
            weight,
            device=mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=dtype,
            mesh_mapper=mesh_config.shard_mapper(mesh_device, tensor_dim=-1),
            cache_file_name=get_cache_file_name(tensor_cache_path, "embed_tokens"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, tokens: ttnn.Tensor) -> ttnn.Tensor:
        """``tokens`` uint32 ``[1, 1, 1, s_local]`` (already SP-sharded) -> ``[1, 1, s_local, hidden]``.

        bf16, not bf8: this seeds the residual stream, and bf8's per-tile shared exponent crushes
        small channels once the large activations a deep stack produces show up.
        """
        local = ttnn.embedding(tokens, self.weight, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
        if len(local.shape) == 3:
            local = ttnn.unsqueeze_to_4D(local)
        if self.mesh_config.tp == 1:
            return local
        full = self.mesh_config.allgather(local, self.ccl_manager, axis=self.mesh_config.tp_axis, dim=3)
        local.deallocate(True)
        return full
