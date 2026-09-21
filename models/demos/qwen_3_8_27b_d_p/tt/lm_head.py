# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LM head — column-parallel on the vocab dim.

The vocab (248320) is padded up to a tile-aligned multiple of TP **before** sharding, not after:
padding after sharding puts the pad columns inside the last TP column only, so the per-device
shard boundaries stop matching the global vocab offsets and every device above 0 reads the wrong
logit range. ``tie_word_embeddings`` is false for this model, so the head is its own tensor.

The head is not on the graded path — per-layer KV PCC is what P1/P2 measure — but it is what the
e2e number needs, so it is built and tested rather than left out.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from ..config import MeshConfig
from ..utils.general_utils import get_cache_file_name
from .compute import matmul_compute_config


def padded_vocab(vocab_size: int, tp: int) -> int:
    per_device = ((vocab_size + tp - 1) // tp + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE * ttnn.TILE_SIZE
    return per_device * tp


class LMHead(LightweightModule):
    def __init__(
        self,
        mesh_device,
        vocab_size: int,
        hidden_size: int,
        state_dict: dict[str, torch.Tensor],
        *,
        mesh_config: MeshConfig,
        weight_dtype=ttnn.bfloat8_b,
        tensor_cache_path: Optional[str] = None,
    ) -> None:
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.vocab_size = vocab_size
        self.padded_vocab_size = padded_vocab(vocab_size, mesh_config.tp)
        self.compute_config = matmul_compute_config(mesh_device)

        weight = None
        if state_dict:
            weight = state_dict["weight"].transpose(0, 1)  # [hidden, vocab]
            pad = self.padded_vocab_size - weight.shape[1]
            if pad:
                weight = torch.nn.functional.pad(weight, (0, pad), "constant", 0)
            weight = weight.unsqueeze(0).unsqueeze(0).contiguous()

        self.weight = ttnn.as_tensor(
            weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
            cache_file_name=get_cache_file_name(tensor_cache_path, "lm_head"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """``x`` full-emb -> per-device logit shard ``[1, 1, S, padded_vocab / tp]``.

        The TP gather is deliberately NOT done here: the caller decides whether to concatenate the
        shards on host (what the PCC tests do) or keep them sharded for on-device sampling.
        """
        return ttnn.matmul(x, self.weight, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_config)
