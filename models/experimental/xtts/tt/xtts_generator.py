# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TTNN XTTS-v2 GPT autoregressive greedy generator (KV-cache decode).

Drives :class:`~models.experimental.xtts.tt.xtts_gpt_model.TtXttsGptModel`:

    kv = model.prefill([cond | text])            # fill the cache (tile-aligned prompt)
    c0 = argmax(model.decode(start_audio, 0))    # start token is the first decode step
    while: c_{i+1} = argmax(model.decode(c_i, i+1)); harvest latent for c_i

Greedy (on-device ``ttnn.argmax``) is deterministic, so the code sequence can be
checked for an *exact* match against the reference (``reference/xtts_gpt_generate.py``).
The single sampled id is read to host each step — this is loop control flow (stop
detection + next embedding index), not tensor compute on the host.

The KV cache is grown by concatenation inside each block's ``forward_decode`` (a
genuine incremental cache — only the new token is projected/attended each step);
swapping to preallocated ``update_cache`` is a later perf optimization.
"""

import torch
import ttnn

from models.experimental.xtts.reference.xtts_gpt_generate import (
    MAX_AUDIO_TOKENS,
    START_AUDIO_TOKEN,
    STOP_AUDIO_TOKEN,
)
from models.experimental.xtts.reference.xtts_gpt_model import NUM_AUDIO_TOKENS
from models.experimental.xtts.tt.xtts_sampler import TtSampler


class TtXttsGenerator:
    """Autoregressive greedy decode over a :class:`TtXttsGptModel`."""

    def __init__(self, model):
        self.model = model

    def _argmax(self, logits):  # logits [b, 1, NUM_AUDIO_TOKENS] -> Python int
        idx = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1)
        return int(ttnn.to_torch(idx).flatten()[0].item())

    def generate(
        self,
        text_ids,
        cond_latents,
        max_new_tokens=MAX_AUDIO_TOKENS,
        temperature=0.0,
        top_k=0,
        repetition_penalty=1.0,
        top_p=1.0,
    ):
        """Free-running decode.

        Args:
            text_ids: torch int ``[1, text_len]`` (already ``[START]``/``[STOP]``-wrapped
                and, for tile-clean prefill, padded to a multiple of 32).
            cond_latents: ttnn ``[1, n_cond, hidden]`` conditioning prompt (TILE).
            max_new_tokens: cap on generated codes.
            temperature/top_k/repetition_penalty/top_p: on-device sampling (``TtSampler``).
                ``temperature <= 0`` selects greedy argmax (deterministic, testable);
                XTTS's natural setting is temp 0.75 / top_k 50 / top_p 0.85 / rep 5.0.

        Returns:
            codes: torch long ``[1, T]`` audio codes (stop token excluded).
            latents: ttnn ``[1, T, hidden]`` mel-span latents aligned to ``codes``.
        """
        sampler = None
        if temperature and temperature > 0.0:
            sampler = TtSampler(self.model.device, NUM_AUDIO_TOKENS, temperature, top_k, repetition_penalty, top_p)
        pick = sampler.pick if sampler else self._argmax

        kv = self.model.prefill(text_ids, cond_latents)
        logits, _, kv = self.model.decode(START_AUDIO_TOKEN, 0, kv)  # start -> c0
        c = pick(logits)

        codes, latents, step = [], [], 1
        # Stop must be checked on the *first* predicted code too (matches the reference,
        # which strips a leading STOP and returns empty): a first-token STOP means
        # "generate nothing" — never emit 1025 as a real code / feed it to the vocoder.
        if c != STOP_AUDIO_TOKEN:
            while True:
                logits, latent, kv = self.model.decode(c, step, kv)
                codes.append(c)
                latents.append(latent)
                nxt = pick(logits)
                if nxt == STOP_AUDIO_TOKEN or len(codes) >= max_new_tokens:
                    break
                c = nxt
                step += 1
        latents_cat = ttnn.concat(latents, dim=1) if latents else None
        return torch.tensor([codes], dtype=torch.long), latents_cat

    def latents_for_codes(self, text_ids, cond_latents, codes):
        """Teacher-forced decode over a fixed code sequence — used to compare latents
        against the reference position-by-position (independent of free-run drift).

        Args:
            codes: list of ints ``[c_0, ..., c_{T-1}]`` (the reference codes).

        Returns:
            preds: list of ``T + 1`` argmax predictions (``preds[i]`` is the model's
                next-token guess after feeding through ``c_{i-1}``; ``preds[:T]``
                should equal ``codes`` when numerics agree).
            latents: ttnn ``[1, T, hidden]`` latents aligned to ``codes``.
        """
        kv = self.model.prefill(text_ids, cond_latents)
        logits, _, kv = self.model.decode(START_AUDIO_TOKEN, 0, kv)
        preds = [self._argmax(logits)]  # predicted c_0

        latents = []
        for i, code in enumerate(codes):
            logits, latent, kv = self.model.decode(int(code), i + 1, kv)
            latents.append(latent)
            preds.append(self._argmax(logits))
        return preds, ttnn.concat(latents, dim=1)
