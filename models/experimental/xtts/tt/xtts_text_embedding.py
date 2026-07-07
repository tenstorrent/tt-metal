# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the XTTS-v2 *text* input path — text ids -> GPT input.

Mirrors ``reference/xtts_text_embedding.py``:

    text_emb = text_embedding(text_ids) + text_pos_embedding(0 .. text_len)

Token and position lookups both use ``ttnn.embedding`` (row-major weight tables);
the result is converted to TILE_LAYOUT so it can feed the GPT decoder directly.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule


def _to_device_rm(torch_tensor, device):
    """torch -> ttnn bf16 ROW_MAJOR tensor on device (weight table for ttnn.embedding)."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


class TtXttsTextEmbedding(LightweightModule):
    def __init__(self, state_dict, device):
        super().__init__()
        self.device = device
        self.text_emb_weight = _to_device_rm(state_dict["gpt.text_embedding.weight"], device)
        self.text_pos_weight = _to_device_rm(state_dict["gpt.text_pos_embedding.emb.weight"], device)

    def forward(self, text_ids):
        """text_ids: torch int tensor ``[batch, text_len]`` -> ttnn ``[batch, text_len, hidden]``.

        Token ids arrive from the host (tokenizer output); position ids 0..seq-1
        are generated on device with ``ttnn.arange`` (no torch fallback).
        """
        seq = text_ids.shape[1]
        ids_tt = ttnn.from_torch(
            text_ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )
        pos_tt = ttnn.arange(0, seq, 1, dtype=ttnn.uint32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos_tt = ttnn.reshape(pos_tt, (1, seq))  # [seq] -> [1, seq]; broadcasts over batch on add

        tok = ttnn.to_layout(ttnn.embedding(ids_tt, self.text_emb_weight), ttnn.TILE_LAYOUT)
        pos = ttnn.to_layout(ttnn.embedding(pos_tt, self.text_pos_weight), ttnn.TILE_LAYOUT)
        return ttnn.add(tok, pos)
