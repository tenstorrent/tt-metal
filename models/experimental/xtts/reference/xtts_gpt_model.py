# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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

from models.experimental.xtts.config import (  # noqa: F401
    NUM_AUDIO_TOKENS,
    NUM_TEXT_TOKENS,
)


class XttsReferenceGptModel(nn.Module):
    def __init__(self, config, num_layers=NUM_LAYERS):
        """Build embeddings, GPT stack, and text/mel prediction heads."""
        super().__init__()
        self.text_embedding = nn.Embedding(NUM_TEXT_TOKENS, HIDDEN_SIZE)
        self.mel_embedding = nn.Embedding(NUM_AUDIO_TOKENS, HIDDEN_SIZE)
        self.text_pos_embedding = nn.Embedding(MAX_TEXT_POS, HIDDEN_SIZE)
        self.mel_pos_embedding = nn.Embedding(MAX_MEL_POS, HIDDEN_SIZE)
        self.stack = XttsReferenceGptStack(config, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(HIDDEN_SIZE, eps=LAYER_NORM_EPS)
        self.text_head = nn.Linear(HIDDEN_SIZE, NUM_TEXT_TOKENS)
        self.mel_head = nn.Linear(HIDDEN_SIZE, NUM_AUDIO_TOKENS)

    def forward(self, text_ids, mel_ids, cond_latents=None, return_latent=False):
        """Embed inputs, run GPT stack, and return logits or mel latents."""
        text_len, mel_len = text_ids.shape[1], mel_ids.shape[1]
        text_pos = torch.arange(text_len, device=text_ids.device)
        mel_pos = torch.arange(mel_len, device=mel_ids.device)

        text_emb = self.text_embedding(text_ids) + self.text_pos_embedding(text_pos)
        mel_emb = self.mel_embedding(mel_ids) + self.mel_pos_embedding(mel_pos)

        parts, offset = [text_emb, mel_emb], 0
        if cond_latents is not None:
            parts = [cond_latents] + parts
            offset = cond_latents.shape[1]

        emb = torch.cat(parts, dim=1)
        enc = self.stack(emb)
        enc = enc[:, offset:]
        enc = self.final_norm(enc)

        if return_latent:
            return enc[:, text_len:]

        text_logits = self.text_head(enc[:, :text_len])
        mel_logits = self.mel_head(enc[:, text_len:])
        return text_logits, mel_logits


def reference_gpt_model(state_dict, num_layers=NUM_LAYERS):
    """Load full reference GPT model weights from a checkpoint."""
    config = build_gpt2_config()
    module = XttsReferenceGptModel(config, num_layers=num_layers)

    module.stack = reference_gpt_stack(state_dict, num_layers=num_layers)

    module.text_embedding.load_state_dict({"weight": state_dict["gpt.text_embedding.weight"]})
    module.mel_embedding.load_state_dict({"weight": state_dict["gpt.mel_embedding.weight"]})
    module.text_pos_embedding.load_state_dict({"weight": state_dict["gpt.text_pos_embedding.emb.weight"]})
    module.mel_pos_embedding.load_state_dict({"weight": state_dict["gpt.mel_pos_embedding.emb.weight"]})

    module.final_norm.load_state_dict(
        {"weight": state_dict["gpt.final_norm.weight"], "bias": state_dict["gpt.final_norm.bias"]}
    )

    module.text_head.load_state_dict(
        {"weight": state_dict["gpt.text_head.weight"], "bias": state_dict["gpt.text_head.bias"]}
    )
    module.mel_head.load_state_dict(
        {"weight": state_dict["gpt.mel_head.weight"], "bias": state_dict["gpt.mel_head.bias"]}
    )

    module.eval()
    return module
