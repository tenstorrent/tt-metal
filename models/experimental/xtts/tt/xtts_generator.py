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
        min_new_tokens=0,
    ):
        """Free-running decode.

        Args:
            text_ids: torch int ``[1, text_len]`` (already ``[START]``/``[STOP]``-wrapped
                and, for tile-clean prefill, padded to a multiple of 32).
            cond_latents: ttnn ``[1, n_cond, hidden]`` conditioning prompt (TILE).
            max_new_tokens: cap on generated codes.
            min_new_tokens: floor on generated codes — STOP is suppressed below it so a take
                can't self-terminate mid-sentence (0 disables, matching HF's default).
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

        # Suppress the STOP token until at least ``min_new_tokens`` codes are emitted (HF's
        # ``min_new_tokens``) so a take can't self-terminate mid-sentence — the fix for
        # "only part of the text was spoken". Add -inf to the STOP logit while below the floor.
        stop_mask = None
        if min_new_tokens > 0:
            m = torch.zeros(1, 1, NUM_AUDIO_TOKENS)
            m[0, 0, STOP_AUDIO_TOKEN] = -1e30
            stop_mask = ttnn.from_torch(m, device=self.model.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def _pick(logits, n_done):
            return pick(ttnn.add(logits, stop_mask) if (stop_mask is not None and n_done < min_new_tokens) else logits)

        kv = self.model.prefill(text_ids, cond_latents)
        logits, _, kv = self.model.decode(START_AUDIO_TOKEN, 0, kv)  # start -> c0
        c = _pick(logits, 0)

        codes, latents, step = [], [], 1
        # Stop must be checked on the *first* predicted code too (matches the reference,
        # which strips a leading STOP and returns empty): a first-token STOP means
        # "generate nothing" — never emit 1025 as a real code / feed it to the vocoder.
        if c != STOP_AUDIO_TOKEN:
            while True:
                logits, latent, kv = self.model.decode(c, step, kv)
                codes.append(c)
                latents.append(latent)
                nxt = _pick(logits, len(codes))
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

    def latents_for_codes_traced(self, text_ids, cond_latents, codes, max_seq):
        """Teacher-forced decode over ``codes`` using the STATIC-KV path captured as a SINGLE
        ttnn trace and replayed per token — the trace-compatible analogue of ``latents_for_codes``.

        Every decode step is the same static-shape, host-sync-free op sequence (fixed-size KV
        cache written in place at a device-driven position), so it is captured once and replayed
        for each token by writing the new token id / mel position / cache position into persistent
        input buffers. Codes are given (teacher-forced), so no sampling is needed inside the loop;
        sampling could not live in the trace anyway (``ttnn.rand`` would replay identical noise).
        Returns latents ``[1, T, hidden]`` aligned to ``codes`` (same contract/alignment as
        ``latents_for_codes``)."""
        m = self.model
        dev = m.device
        m.init_static_decode(max_seq)
        kv, prompt_len = m.prefill_static(text_ids, cond_latents)

        def host_ids(v):
            return ttnn.from_torch(
                torch.tensor([[v]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        def host_cpos(v):
            return ttnn.from_torch(
                torch.full((1, 1, 1, max_seq), float(v)), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT
            )

        # Persistent per-step input buffers (overwritten in place before each replay).
        tok_buf, mp_buf, cpos_buf = m._pos_ids(START_AUDIO_TOKEN), m._pos_ids(0), m.cache_pos(prompt_len)
        # Warm up (compile kernels); this writes at cache pos prompt_len — harmless, the first
        # replay step rewrites that position. Then capture ONE decode step.
        m.decode_static(tok_buf, mp_buf, cpos_buf, kv)
        ttnn.synchronize_device(dev)
        trace_id = ttnn.begin_trace_capture(dev, cq_id=0)
        _logits_dev, latent_dev = m.decode_static(tok_buf, mp_buf, cpos_buf, kv)
        ttnn.end_trace_capture(dev, trace_id, cq_id=0)
        ttnn.synchronize_device(dev)

        # Replay [START, c_0, ..., c_{T-1}]; harvest the latent after each code (START discarded),
        # matching latents_for_codes (code c_j at mel position j+1, cache position prompt_len+j+1).
        # Harvest to HOST each step (a device allocation while a trace is active is unsafe and
        # corrupts the trace buffers), then rebuild the latents tensor on device after releasing.
        seq = [START_AUDIO_TOKEN] + [int(c) for c in codes]
        host_latents = []
        for i, tok in enumerate(seq):
            ttnn.copy_host_to_device_tensor(host_ids(tok), tok_buf)
            ttnn.copy_host_to_device_tensor(host_ids(i), mp_buf)
            ttnn.copy_host_to_device_tensor(host_cpos(prompt_len + i), cpos_buf)
            ttnn.execute_trace(dev, trace_id, blocking=True)
            if i > 0:
                host_latents.append(ttnn.to_torch(latent_dev).float().clone())  # [1, 1, hidden] on host
        ttnn.release_trace(dev, trace_id)
        latents = torch.cat(host_latents, dim=1)  # [1, T, hidden]
        return ttnn.from_torch(latents.to(torch.bfloat16), device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
