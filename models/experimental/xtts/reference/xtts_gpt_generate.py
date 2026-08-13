# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) XTTS-v2 GPT autoregressive greedy generation.

Ground-truth decode loop for the TTNN generator. Mirrors coqui ``Xtts`` inference:
build the GPT prompt ``[cond_latents | text | start_audio]``, greedily sample audio
codes until ``stop_audio_token``, return codes plus the mel-span GPT latents for
HiFi-GAN.

Greedy (argmax) is deterministic, so the TTNN port can be checked for an exact code
match. Token constants from coqui/XTTS-v2 ``config.json`` + ``vocab.json``:
  * audio: ``start_audio_token=1024``, ``stop_audio_token=1025``
  * text wrapped ``[START(261)] + ([lang] + tokens) + [STOP(0)]``

Latents are harvested in-loop: code ``c_i`` sits at mel position ``i + 1``
(``start_audio`` occupies 0).
"""

import torch

from models.experimental.xtts.config import (  # noqa: F401 — re-exported for callers
    MAX_AUDIO_TOKENS,
    START_AUDIO_TOKEN,
    START_TEXT_TOKEN,
    STOP_AUDIO_TOKEN,
    STOP_TEXT_TOKEN,
)


def wrap_text_ids(text_ids):
    """Wrap a ``[b, n]`` text-id tensor as ``[START] + text + [STOP]`` (coqui prefix)."""
    b = text_ids.shape[0]
    start = torch.full((b, 1), START_TEXT_TOKEN, dtype=text_ids.dtype, device=text_ids.device)
    stop = torch.full((b, 1), STOP_TEXT_TOKEN, dtype=text_ids.dtype, device=text_ids.device)
    return torch.cat([start, text_ids, stop], dim=1)


@torch.no_grad()
def greedy_generate(model, text_ids, cond_latents, max_new_tokens=MAX_AUDIO_TOKENS, wrap_text=True):
    """Greedy-decode audio codes from the XTTS GPT.

    Args:
        model: ``XttsReferenceGptModel``.
        text_ids: ``[1, n]`` BPE text ids (already prefixed with the ``[lang]`` tag).
        cond_latents: ``[1, n_cond, hidden]`` audio conditioning latents (GPT prompt).
        max_new_tokens: cap on generated codes.
        wrap_text: wrap ``text_ids`` with ``[START]``/``[STOP]`` (coqui prefix).

    Returns:
        codes: ``[1, T]`` audio-code ids (leading ``start`` and any trailing
            ``stop`` excluded).
        latents: ``[1, T, hidden]`` mel-span GPT latents aligned to ``codes`` — the
            HiFiGAN decoder input.
    """
    if wrap_text:
        text_ids = wrap_text_ids(text_ids)
    device = cond_latents.device
    text_ids = text_ids.to(device)
    mel_ids = torch.full((text_ids.shape[0], 1), START_AUDIO_TOKEN, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        _, mel_logits = model(text_ids, mel_ids, cond_latents=cond_latents)
        next_code = mel_logits[:, -1].argmax(dim=-1, keepdim=True)  # [b, 1]
        mel_ids = torch.cat([mel_ids, next_code], dim=1)
        if (next_code == STOP_AUDIO_TOKEN).all():
            break

    # Mel-span latents for [start, c_0, ..., c_{T-1}(, stop)]; drop the start
    # position to align 1:1 with the emitted codes.
    latents = model(text_ids, mel_ids, cond_latents=cond_latents, return_latent=True)
    codes = mel_ids[:, 1:]
    latents = latents[:, 1:]

    # The decoder consumes only real codes — strip a trailing stop token.
    if codes.shape[1] > 0 and (codes[:, -1] == STOP_AUDIO_TOKEN).all():
        codes = codes[:, :-1]
        latents = latents[:, :-1]
    return codes, latents
