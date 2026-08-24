# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Token embedding lookup -- the model's entry point.

Turns token ids into ebedding vectors that flow through all 64 layers.
"""

from pathlib import Path
from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtQwen36Embedding(LightweightModule):
    """
    Token embedding table.

    ttnn.embedding reads row `token_id` out of the table and writes it to the output.
    No FLOPs, pure data movement.

    Dimensions:
        V = 248320    vocab size (already padded to nicely divisible size in the checkpoint)
        D = 5120      hidden size

    Weight:
        embed_tokens  [248320, 5120]   ~2.5 GB in bf16 -- the single largest
                                       tensor in the model, alongside lm_head

    Shapes:
        token_ids  [B, T]        uint32, ROW_MAJOR layout
        output     [B, T, D]     bfloat16, TILE layout

    Note: embeddings are NOT tied to the lm_head in this model
    (`tie_word_embeddings: false`), so the two big tables are separate weights.

    Single-device only for now.
    """

    def __init__(self, device, torch_weight: torch.Tensor, cache_file_name: Optional[Path] = None):
        self.device = device

        # ttnn.embedding validates the table hard (embedding_device_operation.cpp:33-36):
        #   bfloat16 only
        #   ROW_MAJOR only -- a TILE table gets silently untilized on *every* call
        # memory_config is not optional either: as_tensor raises if device is set
        # without it (ttnn/ttnn/operations/core.py:656).
        self.weight = ttnn.as_tensor(
            torch_weight,  # [V, D]
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def forward(self, token_ids: ttnn.Tensor) -> ttnn.Tensor:
        """
        [B, T] uint32 -> [B, T, D] bfloat16.

        layout=TILE selects the fused program factory that tilizes inside the
        kernel, but only when T % 32 == 0 (D = 5120 already is). Otherwise it
        quietly falls back to a separate to_layout -- so pad T at the caller.
        """
        return ttnn.embedding(
            token_ids,
            self.weight,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
