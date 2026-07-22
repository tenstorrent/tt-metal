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
from models.experimental.xtts.reference.xtts_gpt_block import HIDDEN_SIZE
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

        # Fixed-size KV cache: size it for the prompt + the whole decode budget (mel token i sits
        # at cache position prompt_len + i; +1 for the start_audio step), rounded to a tile.
        prompt_len = cond_latents.shape[1] + text_ids.shape[1]
        max_seq = -(-(prompt_len + max_new_tokens + 1) // 32) * 32
        kv = self.model.prefill(text_ids, cond_latents, max_seq)
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
        prompt_len = cond_latents.shape[1] + text_ids.shape[1]
        max_seq = -(-(prompt_len + len(codes) + 2) // 32) * 32
        kv = self.model.prefill(text_ids, cond_latents, max_seq)
        logits, _, kv = self.model.decode(START_AUDIO_TOKEN, 0, kv)
        preds = [self._argmax(logits)]  # predicted c_0

        latents = []
        for i, code in enumerate(codes):
            logits, latent, kv = self.model.decode(int(code), i + 1, kv)
            latents.append(latent)
            preds.append(self._argmax(logits))
        return preds, ttnn.concat(latents, dim=1)

    def generate_ondevice_traced(self, prompt_len, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
        """FULLY on-device, end-to-end traceable decode: one captured step — decode_on_device +
        on-device Gumbel-max sampling (``TtSampler.pick_dev`` over PRE-DRAWN host noise, no
        ``ttnn.rand``) + in-place token feedback (``ttnn.copy``) + on-device latent/code accumulation
        (onehot writes) — replayed ``max_new_tokens`` times with ONLY counter/noise-row writes (NO
        per-step host readback / no host loop control). Requires ``self.model._static_kv`` already
        seeded (by the setup trace/prefill). Reads codes+latents ONCE at the end and trims at the
        first STOP. Returns ``(codes, latents)``.

        This is the clean pre->device->post shape: noise is drawn on host up front (preprocessing),
        the decode runs entirely on device, and STOP self-termination becomes a post-loop trim. The
        sampling now matches the host path in distribution (true uniform Gumbel draw); the only
        residual difference from host self-termination is the fixed step budget + trailing-drone trim."""
        m = self.model
        dev = m.device
        N = int(max_new_tokens)
        sampler = TtSampler(dev, NUM_AUDIO_TOKENS, temperature, top_k, repetition_penalty, top_p)
        T32 = ttnn.TILE_LAYOUT

        def f32(t):
            return ttnn.from_torch(t, device=dev, dtype=ttnn.float32, layout=T32)

        tok_buf = m._pos_ids(START_AUDIO_TOKEN)  # [1,1] uint32 (embedding input; fed back in place)
        mp_buf = m._pos_ids(0)  # [1,1] uint32 (mel position)
        cpos_buf = m.cache_pos(prompt_len)  # [1,1,1,max_seq] fp32 (absolute cache position)
        # Pre-draw ALL Gumbel noise on HOST with a proper RNG (torch), once, before the loop. The
        # noise is independent of the logits, so drawing it up front is preprocessing (not an in-loop
        # host fallback); each step's row is streamed into a persistent [1, V] buffer like the
        # position counters. This is a true uniform draw -> exact multinomial (vs a poor in-trace hash).
        sampled = bool(temperature and temperature > 0.0)
        if sampled:
            u = torch.rand(N, NUM_AUDIO_TOKENS).clamp_(1e-4, 1.0 - 1e-3)
            gumbel_all = -torch.log(-torch.log(u))  # [N, V] Gumbel(0,1)
        noise_buf = f32(torch.zeros(1, NUM_AUDIO_TOKENS)) if sampled else None  # [1, V] fp32, refreshed per step
        arange_row = f32(torch.arange(N, dtype=torch.float32).reshape(1, N, 1))  # latent-slot selector base
        slot_row = f32(torch.zeros(1, N, 1))
        arange_col = f32(torch.arange(N, dtype=torch.float32).reshape(1, N))  # code-slot selector base
        slot_col = f32(torch.zeros(1, N))
        latents_buf = ttnn.from_torch(torch.zeros(1, N, HIDDEN_SIZE), device=dev, dtype=ttnn.bfloat16, layout=T32)
        codes_buf = f32(torch.zeros(1, N))  # fp32 so code ids > 256 stay exact

        def step_ops():
            logits, latent = m.decode_on_device(tok_buf, mp_buf, cpos_buf, m._static_kv)  # kv updated in place
            tok = sampler.pick_dev(logits, noise_buf)  # [1,1] uint32 sampled on device (pre-drawn noise)
            ttnn.copy(tok, tok_buf)  # on-device token feedback -> next step's embedding
            oh_r = ttnn.typecast(ttnn.eq(arange_row, slot_row), ttnn.bfloat16)  # [1,N,1] one-hot at step
            ttnn.multiply(latents_buf, ttnn.add(ttnn.multiply(oh_r, -1.0), 1.0), output_tensor=latents_buf)
            ttnn.add(latents_buf, ttnn.multiply(latent, oh_r), output_tensor=latents_buf)
            oh_c = ttnn.typecast(ttnn.eq(arange_col, slot_col), ttnn.float32)  # [1,N] one-hot at step
            ttnn.multiply(codes_buf, ttnn.add(ttnn.multiply(oh_c, -1.0), 1.0), output_tensor=codes_buf)
            ttnn.add(codes_buf, ttnn.multiply(ttnn.typecast(tok, ttnn.float32), oh_c), output_tensor=codes_buf)

        # Warmup (compile). It executes one step at position prompt_len — harmless: real step 0
        # rewrites that cache slot; we reset the sampler's seen mask / token / accumulators below.
        step_ops()
        ttnn.synchronize_device(dev)
        sampler.reset()
        ttnn.copy(m._pos_ids(START_AUDIO_TOKEN), tok_buf)
        ttnn.multiply(latents_buf, 0.0, output_tensor=latents_buf)
        ttnn.multiply(codes_buf, 0.0, output_tensor=codes_buf)
        # Capture the one static-shape sample-step, then replay it N times with only counter writes.
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        step_ops()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)

        def h32(shape, v):
            return ttnn.from_torch(torch.full(shape, float(v)), dtype=ttnn.float32, layout=T32)

        def hu32(v):
            return ttnn.from_torch(
                torch.tensor([[v]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        for i in range(N):
            ttnn.copy_host_to_device_tensor(hu32(i), mp_buf)
            ttnn.copy_host_to_device_tensor(h32((1, 1, 1, m.max_seq), prompt_len + i), cpos_buf)
            if sampled:  # stream this step's pre-drawn Gumbel-noise row into the persistent buffer
                ttnn.copy_host_to_device_tensor(
                    ttnn.from_torch(gumbel_all[i : i + 1].contiguous(), dtype=ttnn.float32, layout=T32), noise_buf
                )
            ttnn.copy_host_to_device_tensor(h32((1, N, 1), i), slot_row)
            ttnn.copy_host_to_device_tensor(h32((1, N), i), slot_col)
            ttnn.execute_trace(dev, tid, blocking=True)
        ttnn.release_trace(dev, tid)

        # Read once. code c_i is sampled at step i; latent slot i is the fed-token latent (START at
        # slot 0), so c_j's latent is slot j+1. Trim at the first STOP (single post-loop host op).
        codes = ttnn.to_torch(codes_buf).float().round().to(torch.long).flatten().tolist()
        lat = ttnn.to_torch(latents_buf).float()
        stop = next((i for i, c in enumerate(codes) if c == STOP_AUDIO_TOKEN), N)
        seq = codes[:stop]
        # Fixed-N decode can't self-terminate mid-loop, so after the model is "done" it drones on a
        # single repeated code (the "accuracyyyy" drag). Trim that trailing same-code run to a short
        # hold — this is the on-device analogue of stopping at STOP.
        if seq:
            run = 1
            while run < len(seq) and seq[-1 - run] == seq[-1]:
                run += 1
            if run > 8:
                seq = seq[: len(seq) - run + 2]  # keep ~2 for a natural final hold
        cut = min(max(len(seq), 1), N - 1)
        codes_out = torch.tensor([codes[:cut]], dtype=torch.long)
        lat_out = ttnn.from_torch(
            lat[:, 1 : cut + 1, :].to(torch.bfloat16), device=dev, dtype=ttnn.bfloat16, layout=T32
        )
        return codes_out, lat_out
