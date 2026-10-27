# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from models.experimental.xtts.config import (  # noqa: F401
    MAX_AUDIO_TOKENS,
    START_AUDIO_TOKEN,
    START_TEXT_TOKEN,
    STOP_AUDIO_TOKEN,
    STOP_TEXT_TOKEN,
)


def wrap_text_ids(text_ids):
    """Wrap text token ids with start and stop tokens."""
    b = text_ids.shape[0]
    start = torch.full((b, 1), START_TEXT_TOKEN, dtype=text_ids.dtype, device=text_ids.device)
    stop = torch.full((b, 1), STOP_TEXT_TOKEN, dtype=text_ids.dtype, device=text_ids.device)
    return torch.cat([start, text_ids, stop], dim=1)


@torch.no_grad()
def greedy_generate(model, text_ids, cond_latents, max_new_tokens=MAX_AUDIO_TOKENS, wrap_text=True):
    """Greedy-decode audio codes then return codes and GPT latents."""
    if wrap_text:
        text_ids = wrap_text_ids(text_ids)
    device = cond_latents.device
    text_ids = text_ids.to(device)
    mel_ids = torch.full((text_ids.shape[0], 1), START_AUDIO_TOKEN, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        _, mel_logits = model(text_ids, mel_ids, cond_latents=cond_latents)
        next_code = mel_logits[:, -1].argmax(dim=-1, keepdim=True)
        mel_ids = torch.cat([mel_ids, next_code], dim=1)
        if (next_code == STOP_AUDIO_TOKEN).all():
            break

    latents = model(text_ids, mel_ids, cond_latents=cond_latents, return_latent=True)
    codes = mel_ids[:, 1:]
    latents = latents[:, 1:]

    if codes.shape[1] > 0 and (codes[:, -1] == STOP_AUDIO_TOKEN).all():
        codes = codes[:, :-1]
        latents = latents[:, :-1]
    return codes, latents
