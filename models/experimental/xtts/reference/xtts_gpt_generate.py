# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Reference (pure-PyTorch) XTTS-v2 GPT autoregressive greedy generation.

The ground-truth decode loop that the TTNN generator (Phase 2) is validated
against. Mirrors coqui ``Xtts`` inference: build the GPT prompt
``[cond_latents | text | start_audio]``, greedily sample audio codes one at a
time until ``stop_audio_token``, and return the codes plus the mel-span GPT
latents that feed the HiFiGAN decoder.

**Greedy (argmax) is deliberate** — not XTTS's default temperature/top-k/top-p
sampling. Greedy is deterministic, so the TTNN port can be checked for an *exact*
code-sequence match (the strongest correctness anchor). Real XTTS sampling gives
more natural audio; that is a later quality pass, not a correctness anchor.

Token / prefix constants come from coqui/XTTS-v2 ``config.json`` + ``vocab.json``:
  * audio: ``start_audio_token=1024``, ``stop_audio_token=1025`` (mel vocab 1026,
    ``gpt_max_audio_tokens=605``).
  * text is wrapped ``[START(261)] + ([lang] + tokens) + [STOP(0)]``.

NOTE (faithfulness): ``config.json`` carries ``gpt_start/stop_text_token=None``;
the coqui ``GPT`` constructor defaults are ``261``/``0``, which are exactly the
``[START]``/``[STOP]`` ids in ``vocab.json`` — hence the wrapping above. This is
not bit-verifiable here (``coqui-tts`` is not installed), but it only affects
real-audio faithfulness: the loop defined here is the ground truth the TTNN port
must reproduce regardless.

NOTE (latent alignment): latents are *harvested in-loop* — code ``c_i`` sits at
mel position ``i + 1`` (the ``start_audio`` token occupies mel position 0), and
its latent is the post-``final_norm`` hidden state at that position. This is what
the Phase-2 TTNN decode loop naturally produces too, so TT matches reference
exactly. coqui's alternative (a separate ``return_latent`` pass over just the
codes, positions starting at 0) is a faithfulness variant revisited at audio time.
"""

import torch


START_TEXT_TOKEN = 261  # [START] in vocab.json
STOP_TEXT_TOKEN = 0  # [STOP]
START_AUDIO_TOKEN = 1024  # gpt_start_audio_token
STOP_AUDIO_TOKEN = 1025  # gpt_stop_audio_token
MAX_AUDIO_TOKENS = 605  # gpt_max_audio_tokens


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
