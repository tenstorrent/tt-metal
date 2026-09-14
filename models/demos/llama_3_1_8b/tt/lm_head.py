# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Column-parallel LM head.

``vocab_size`` 128256 / tp 4 = 32064, which IS tile-aligned (1002 tiles), so no padding is needed —
but the padding is implemented anyway and exercised by the unit test at a reduced vocab, because a
head whose per-device width is not a multiple of 32 is the common case and silently truncating it
would be a wrong-answer bug rather than a crash.

``tie_word_embeddings`` is false for this checkpoint, so ``lm_head.weight`` is a real tensor and not
a view of the embedding table; the loader asserts it is present.

Prefill is headless by default (the populated KV cache is the output), so this runs only when a
caller asks for logits. It is unit-tested but is not on the acceptance path.
"""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule

from .compute import matmul_compute_config


def padded_vocab(vocab_size: int, tp: int) -> int:
    """Vocab rounded up so each TP column's slice is a whole number of tiles."""
    per_device = (vocab_size + tp - 1) // tp
    per_device = ((per_device + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    return per_device * tp


class LMHead(LightweightModule):
    def __init__(
        self,
        mesh_device,
        cfg,
        mesh_config,
        state_dict: Optional[dict] = None,
        weight_dtype=ttnn.bfloat8_b,
        cache_file_name: Optional[str] = None,
    ):
        super().__init__()
        self.mesh_config = mesh_config
        self.vocab_size = cfg.vocab_size
        self.padded_vocab = padded_vocab(cfg.vocab_size, mesh_config.tp)
        self.compute_kernel_config = matmul_compute_config(mesh_device)

        w = None
        if state_dict:
            assert "weight" in state_dict, "lm_head.weight missing (tie_word_embeddings is false here)"
            w = state_dict["weight"].transpose(0, 1)  # [hidden, vocab]
            if self.padded_vocab > self.vocab_size:
                w = torch.nn.functional.pad(w, (0, self.padded_vocab - self.vocab_size), "constant", 0.0)
            w = w.unsqueeze(0).unsqueeze(0)

        self.weight = ttnn.as_tensor(
            w,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=cache_file_name,
        )

    def forward(self, x):
        """``[1, 1, s_local, hidden]`` -> ``[1, 1, s_local, padded_vocab/tp]``, vocab-sharded on TP.

        The result stays sharded: gathering 128256 logits per token onto every column would cost
        more than any caller of a headless prefill wants to pay. The padding columns are zeros in the
        last TP column only, and a caller reading logits trims to ``vocab_size`` after composing.
        """
        return ttnn.matmul(x, self.weight, dtype=ttnn.bfloat16, compute_kernel_config=self.compute_kernel_config)
