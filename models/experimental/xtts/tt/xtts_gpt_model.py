# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the XTTS-v2 GPT decoder *with embeddings and heads*.

Mirrors ``reference/xtts_gpt_model.py``:

    text_emb = text_embedding(text_ids) + text_pos_embedding(0..text_len)
    mel_emb  = mel_embedding(mel_ids)   + mel_pos_embedding(0..mel_len)
    emb = concat([text_emb, mel_emb], dim=1)
    enc = final_norm(stack(emb))          # stack = 30 blocks + ln_f
    text_logits = text_head(enc[:, :text_len])
    mel_logits  = mel_head(enc[:, text_len:])

Token/position lookups use ``ttnn.embedding``; the two heads are ``ttnn.linear``.
GPT-2 ``nn.Linear`` head weights are stored ``[out, in]`` so they are transposed
to the ``[in, out]`` layout ``ttnn.linear`` expects (y = x @ W + b).

NOTE: the ``[text] + [mel]`` concat and the text/mel split slice along the
sequence dim. In TILE_LAYOUT these are cleanest when ``text_len`` and ``mel_len``
are multiples of the 32-row tile; the PCC test picks tile-aligned lengths.
"""

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE, LAYER_NORM_EPS, NUM_LAYERS
from models.experimental.xtts.tt.xtts_gpt_block import _to_device
from models.experimental.xtts.tt.xtts_gpt_stack import TtXttsGptStack


def _to_device_rm(torch_tensor, device):
    """torch -> ttnn bf16 ROW_MAJOR tensor on device (weight table for ttnn.embedding)."""
    return ttnn.from_torch(
        torch_tensor.to(torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        dtype=ttnn.bfloat16,
    )


class TtXttsGptModel(LightweightModule):
    def __init__(self, state_dict, device, num_layers=NUM_LAYERS):
        super().__init__()
        self.device = device

        # Token + learned-position embedding tables (row-major for ttnn.embedding).
        self.text_emb_weight = _to_device_rm(state_dict["gpt.text_embedding.weight"], device)
        self.mel_emb_weight = _to_device_rm(state_dict["gpt.mel_embedding.weight"], device)
        self.text_pos_weight = _to_device_rm(state_dict["gpt.text_pos_embedding.emb.weight"], device)
        self.mel_pos_weight = _to_device_rm(state_dict["gpt.mel_pos_embedding.emb.weight"], device)

        # The 30 decoder blocks + ln_f.
        self.stack = TtXttsGptStack(state_dict, device, num_layers=num_layers)

        # Second final LayerNorm.
        self.final_norm_weight = _to_device(state_dict["gpt.final_norm.weight"], device)
        self.final_norm_bias = _to_device(state_dict["gpt.final_norm.bias"], device)

        # Heads. nn.Linear weight is [out, in]; ttnn.linear wants [in, out] -> transpose.
        self.text_head_weight = _to_device(state_dict["gpt.text_head.weight"].t().contiguous(), device)
        self.text_head_bias = _to_device(state_dict["gpt.text_head.bias"], device)
        self.mel_head_weight = _to_device(state_dict["gpt.mel_head.weight"].t().contiguous(), device)
        self.mel_head_bias = _to_device(state_dict["gpt.mel_head.bias"], device)

    def _embed(self, ids, tok_weight, pos_weight):
        """ids: torch int tensor [batch, seq] -> ttnn [batch, seq, hidden] (token + position).

        Token ids arrive from the host; position ids 0..seq-1 are generated on
        device with ``ttnn.arange`` (no torch fallback).
        """
        seq = ids.shape[1]
        ids_tt = ttnn.from_torch(
            ids.to(torch.int32), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device, dtype=ttnn.uint32
        )
        pos_tt = ttnn.arange(0, seq, 1, dtype=ttnn.uint32, device=self.device, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos_tt = ttnn.reshape(pos_tt, (1, seq))  # [seq] -> [1, seq]; broadcasts over batch on add

        tok = ttnn.to_layout(ttnn.embedding(ids_tt, tok_weight), ttnn.TILE_LAYOUT)
        pos = ttnn.to_layout(ttnn.embedding(pos_tt, pos_weight), ttnn.TILE_LAYOUT)
        return ttnn.add(tok, pos)

    def forward(self, text_ids, mel_ids):
        """text_ids / mel_ids are torch int tensors ``[batch, seq]``.

        Returns ``(text_logits, mel_logits)`` on device.
        """
        text_len, mel_len = text_ids.shape[1], mel_ids.shape[1]

        text_emb = self._embed(text_ids, self.text_emb_weight, self.text_pos_weight)
        mel_emb = self._embed(mel_ids, self.mel_emb_weight, self.mel_pos_weight)

        emb = ttnn.concat([text_emb, mel_emb], dim=1)  # [b, text_len + mel_len, hidden]
        enc = self.stack(emb)
        enc = ttnn.layer_norm(enc, weight=self.final_norm_weight, bias=self.final_norm_bias, epsilon=LAYER_NORM_EPS)

        b = enc.shape[0]
        text_part = ttnn.slice(enc, [0, 0, 0], [b, text_len, HIDDEN_SIZE])
        mel_part = ttnn.slice(enc, [0, text_len, 0], [b, text_len + mel_len, HIDDEN_SIZE])

        text_logits = ttnn.linear(text_part, self.text_head_weight, bias=self.text_head_bias)
        mel_logits = ttnn.linear(mel_part, self.mel_head_weight, bias=self.mel_head_bias)
        return text_logits, mel_logits
