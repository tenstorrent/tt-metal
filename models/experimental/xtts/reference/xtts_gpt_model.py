# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) XTTS-v2 GPT decoder *with embeddings and heads*.

Extends :mod:`models.experimental.xtts.reference.xtts_gpt_stack` (the 30
``GPT2Block`` decoder layers + ``ln_f``) with the pieces that surround it in the
upstream coqui XTTS-v2 ``gpt.py``:

  * input embeddings — a text token embedding and a mel/audio token embedding,
    each summed with its own *learned* position embedding
    (``LearnedPositionEmbeddings``: a plain ``nn.Embedding`` indexed by position);
  * the concatenated ``[text] + [mel]`` stream is run through the GPT decoder
    (30 blocks + ``ln_f``), then a **second** LayerNorm ``final_norm``;
  * two output heads — ``text_head`` and ``mel_head`` — that project the text and
    mel spans of the decoder output to their respective vocab logits.

Mirrors coqui ``GPT.get_logits`` with no conditioning ``prompt`` (offset 0):

    emb = cat([text_emb, mel_emb], dim=1)
    enc = final_norm(stack(emb))          # stack already applies ln_f
    text_logits = text_head(enc[:, :text_len])
    mel_logits  = mel_head(enc[:, text_len:])

Note the two consecutive LayerNorms (``gpt.gpt.ln_f`` then ``gpt.final_norm``)
are intentional — both are present in the upstream checkpoint.

Weights come straight from the upstream checkpoint at
https://huggingface.co/coqui/XTTS-v2 (``model.pth``); see ``xtts_gpt_block`` for
how the tensor state dict is extracted without depending on ``coqui-tts``.
"""

import torch
from torch import nn

from models.experimental.xtts.reference.xtts_gpt_block import (
    HIDDEN_SIZE,
    LAYER_NORM_EPS,
    MAX_MEL_POS,
    MAX_TEXT_POS,
    NUM_LAYERS,
    build_gpt2_config,
)
from models.experimental.xtts.reference.xtts_gpt_stack import XttsReferenceGptStack, reference_gpt_stack

# Vocab sizes, read off the checkpoint embedding/head tensors
# (gpt.text_embedding=6681, gpt.mel_embedding=1026).
NUM_TEXT_TOKENS = 6681
NUM_AUDIO_TOKENS = 1026


class XttsReferenceGptModel(nn.Module):
    """XTTS GPT decoder end to end: embeddings -> 30 blocks + ln_f -> final_norm -> heads.

    ``forward(text_ids, mel_ids)`` takes integer token ids
    (``text_ids`` in ``[0, NUM_TEXT_TOKENS)``, ``mel_ids`` in
    ``[0, NUM_AUDIO_TOKENS)``) and returns ``(text_logits, mel_logits)`` of shape
    ``[batch, text_len, NUM_TEXT_TOKENS]`` and ``[batch, mel_len, NUM_AUDIO_TOKENS]``.
    """

    def __init__(self, config, num_layers=NUM_LAYERS):
        super().__init__()
        # Token embeddings.
        self.text_embedding = nn.Embedding(NUM_TEXT_TOKENS, HIDDEN_SIZE)
        self.mel_embedding = nn.Embedding(NUM_AUDIO_TOKENS, HIDDEN_SIZE)
        # Learned position embeddings (LearnedPositionEmbeddings.emb in coqui).
        self.text_pos_embedding = nn.Embedding(MAX_TEXT_POS, HIDDEN_SIZE)
        self.mel_pos_embedding = nn.Embedding(MAX_MEL_POS, HIDDEN_SIZE)
        # The 30 decoder blocks + ln_f (weights attached by reference_gpt_model).
        self.stack = XttsReferenceGptStack(config, num_layers=num_layers)
        # Second final LayerNorm applied on top of the stack output.
        self.final_norm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)
        # Output heads.
        self.text_head = nn.Linear(HIDDEN_SIZE, NUM_TEXT_TOKENS)
        self.mel_head = nn.Linear(HIDDEN_SIZE, NUM_AUDIO_TOKENS)

    def forward(self, text_ids, mel_ids):
        text_len, mel_len = text_ids.shape[1], mel_ids.shape[1]
        text_pos = torch.arange(text_len, device=text_ids.device)
        mel_pos = torch.arange(mel_len, device=mel_ids.device)

        text_emb = self.text_embedding(text_ids) + self.text_pos_embedding(text_pos)
        mel_emb = self.mel_embedding(mel_ids) + self.mel_pos_embedding(mel_pos)

        emb = torch.cat([text_emb, mel_emb], dim=1)  # [b, text_len + mel_len, hidden]
        enc = self.stack(emb)  # 30 blocks + ln_f
        enc = self.final_norm(enc)

        text_logits = self.text_head(enc[:, :text_len])
        mel_logits = self.mel_head(enc[:, text_len:])
        return text_logits, mel_logits


def reference_gpt_model(state_dict, num_layers=NUM_LAYERS):
    """Build the full XTTS GPT decoder (embeddings + stack + heads) with real weights.

    Args:
        state_dict: full checkpoint state dict from
            :func:`models.experimental.xtts.reference.xtts_gpt_block.load_xtts_state_dict`.
        num_layers: number of repeating GPT-2 blocks (30 for XTTS-v2).
    """
    config = build_gpt2_config()
    module = XttsReferenceGptModel(config, num_layers=num_layers)

    # The 30 blocks + ln_f, already loaded and in eval mode.
    module.stack = reference_gpt_stack(state_dict, num_layers=num_layers)

    # Embeddings (note the learned position tables live under ``.emb.weight``).
    module.text_embedding.load_state_dict({"weight": state_dict["gpt.text_embedding.weight"]})
    module.mel_embedding.load_state_dict({"weight": state_dict["gpt.mel_embedding.weight"]})
    module.text_pos_embedding.load_state_dict({"weight": state_dict["gpt.text_pos_embedding.emb.weight"]})
    module.mel_pos_embedding.load_state_dict({"weight": state_dict["gpt.mel_pos_embedding.emb.weight"]})

    # Second final norm.
    module.final_norm.load_state_dict(
        {"weight": state_dict["gpt.final_norm.weight"], "bias": state_dict["gpt.final_norm.bias"]}
    )

    # Heads.
    module.text_head.load_state_dict(
        {"weight": state_dict["gpt.text_head.weight"], "bias": state_dict["gpt.text_head.bias"]}
    )
    module.mel_head.load_state_dict(
        {"weight": state_dict["gpt.mel_head.weight"], "bias": state_dict["gpt.mel_head.bias"]}
    )

    module.eval()
    return module
