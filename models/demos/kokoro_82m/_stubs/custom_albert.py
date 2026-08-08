# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `custom_albert` (hexgrad/Kokoro-82M `bert`, a
StyleTTS2 `CustomAlbert` — an HF `AlbertModel` returning `last_hidden_state`).

Reference forward:

    embedding_output = embeddings(input_ids, token_type_ids, position_ids)
    sequence_output  = encoder(embedding_output, extended_attention_mask)
    return sequence_output      # last_hidden_state (pooler is unused here)

The captured `attention_mask` is all ones, so the additive attention mask is
zero and has no effect on the softmax; this port therefore ignores it. It reuses
the graduated native `albert_embeddings` and `albert_transformer` ports and
threads the hidden state through them. Everything runs natively in float32.
"""

from __future__ import annotations

from models.demos.kokoro_82m._stubs.albert_embeddings import build as _build_embeddings
from models.demos.kokoro_82m._stubs.albert_transformer import build as _build_transformer

HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Bind embeddings + encoder ports and return a native ttnn forward."""
    m = torch_module
    emb_fwd = _build_embeddings(device, m.embeddings)
    enc_fwd = _build_transformer(device, m.encoder)

    def forward(input_ids, attention_mask=None, token_type_ids=None, position_ids=None, *args, **kwargs):
        emb = emb_fwd(input_ids, token_type_ids=token_type_ids, position_ids=position_ids)  # [B, T, embed]
        out = enc_fwd(emb, None)  # [B, T, hidden]
        return out

    return forward


def custom_albert(*args, **kwargs):
    raise RuntimeError(
        "custom_albert requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )
