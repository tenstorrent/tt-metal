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

    # ------------------------------------------------------------------ #
    # Traced static-KV decode (canonical tt-metal pattern; see models/tt_transformers/tt/
    # generator.py). The one static-shape decode step is captured ONCE; trace_id + the
    # persistent input/output device buffers are cached on ``self`` and the step is replayed
    # per token by writing the new (token id, mel position, cache position) into the persistent
    # inputs with ``copy_host_to_device_tensor`` then ``execute_trace``. Selection stays on HOST
    # between replays: allocating a device tensor while a trace is captured is unsafe/corrupting,
    # and ``ttnn.rand`` inside a trace would replay identical noise — so no in-trace sampling.
    # ------------------------------------------------------------------ #
    def _capture_decode_trace(self, text_ids, cond_latents, max_seq):
        """Prefill fixed-size caches (eager concat prefill) and capture one static-KV decode step."""
        self.model.init_static_decode(max_seq)
        kv, prompt_len = self.model.prefill_static(text_ids, cond_latents)
        return self._capture_decode_on(kv, prompt_len, max_seq)

    def _capture_decode_on(self, kv, prompt_len, max_seq):
        """Capture one static-KV decode step over the ALREADY-SEEDED caches ``kv`` (from either
        ``prefill_static`` or a fill_cache'd persistent cache). Caches trace state on ``self``;
        returns ``prompt_len`` (mel token i occupies cache position prompt_len + i)."""
        m = self.model
        self._trace_max_seq = max_seq
        self._trace_kv = kv
        self._trace_prompt_len = prompt_len
        # Persistent per-step input buffers — overwritten in place before each replay; their
        # device addresses must stay stable, so they are NEVER reallocated.
        self._tok_buf = m._pos_ids(START_AUDIO_TOKEN)
        self._mp_buf = m._pos_ids(0)
        self._cpos_buf = m.cache_pos(self._trace_prompt_len)
        # Warm up (compile kernels; this writes at cache pos prompt_len — harmless, the first
        # replay step rewrites it), then capture ONE decode step.
        m.decode_static(self._tok_buf, self._mp_buf, self._cpos_buf, self._trace_kv)
        ttnn.synchronize_device(m.device)
        self._trace_id = ttnn.begin_trace_capture(m.device, cq_id=0)
        self._logits_dev, self._latent_dev = m.decode_static(
            self._tok_buf, self._mp_buf, self._cpos_buf, self._trace_kv
        )
        ttnn.end_trace_capture(m.device, self._trace_id, cq_id=0)
        ttnn.synchronize_device(m.device)
        return self._trace_prompt_len

    def _decode_step_traced(self, token, mel_pos):
        """Replay the captured decode step for one token: refresh the persistent input buffers in
        place (token id, mel position, absolute cache position = prompt_len + mel_pos), execute the
        trace, return the (device) ``(logits, latent)`` output buffers. Allocates no device tensors
        (host->device copies only), so the captured trace stays valid."""
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([[token]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            self._tok_buf,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.tensor([[mel_pos]], dtype=torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            ),
            self._mp_buf,
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(
                torch.full((1, 1, 1, self._trace_max_seq), float(self._trace_prompt_len + mel_pos)),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
            ),
            self._cpos_buf,
        )
        ttnn.execute_trace(self.model.device, self._trace_id, blocking=True)
        return self._logits_dev, self._latent_dev

    def _release_decode_trace(self):
        ttnn.release_trace(self.model.device, self._trace_id)
        self._trace_id = None

    def latents_for_codes_traced(self, text_ids, cond_latents, codes, max_seq):
        """Teacher-forced traced decode (trace-compatible analogue of ``latents_for_codes``): replay
        the captured step over ``[START, c_0, ..., c_{T-1}]``, harvesting the latent after each code
        (START discarded), aligned exactly as ``latents_for_codes``. Returns latents ``[1, T, hidden]``.
        Latents are harvested to HOST each step (device alloc during a live trace is unsafe) and the
        device tensor is rebuilt after the trace is released."""
        self._capture_decode_trace(text_ids, cond_latents, max_seq)
        host_latents = []
        for i, tok in enumerate([START_AUDIO_TOKEN] + [int(c) for c in codes]):
            _, latent_dev = self._decode_step_traced(tok, i)
            if i > 0:
                host_latents.append(ttnn.to_torch(latent_dev).float().clone())  # [1, 1, hidden] on host
        self._release_decode_trace()
        latents = torch.cat(host_latents, dim=1)  # [1, T, hidden]
        return ttnn.from_torch(
            latents.to(torch.bfloat16), device=self.model.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def _host_pick(self, logits_dev, seen, n_done, min_new_tokens, temperature, top_k, top_p, rep):
        """Host-side token selection over the trace's logits output buffer, mirroring the on-device
        ``TtSampler`` order: repetition-penalty -> temperature -> top-k -> top-p (nucleus) -> sample.
        ``temperature <= 0`` selects greedy argmax. STOP is suppressed while fewer than
        ``min_new_tokens`` codes exist. Runs on host so no device tensor is allocated while the
        trace is live (that would corrupt it), and there is no ``ttnn.rand`` inside the trace."""
        lg = ttnn.to_torch(logits_dev).float().reshape(-1)
        if n_done < min_new_tokens:
            lg[STOP_AUDIO_TOKEN] = -1e30  # min-length floor
        if rep != 1.0 and seen:
            idx = torch.tensor(sorted(seen), dtype=torch.long)
            s = lg[idx]
            lg[idx] = torch.where(s > 0, s / rep, s * rep)  # HF repetition penalty on seen tokens
        if not (temperature and temperature > 0.0):
            return int(torch.argmax(lg).item())  # greedy
        lg = lg / temperature
        if top_k:
            kth = torch.topk(lg, min(int(top_k), lg.numel())).values[-1]
            lg = torch.where(lg >= kth, lg, torch.full_like(lg, -1e30))
        if 0.0 < top_p < 1.0:  # nucleus over the (top-k-masked) distribution
            sp, si = torch.sort(lg, descending=True)
            sm = torch.softmax(sp, dim=-1)
            keep = (torch.cumsum(sm, dim=-1) - sm) < top_p  # exclusive prefix < p (crossing token kept)
            lg[si[~keep]] = -1e30
        return int(torch.multinomial(torch.softmax(lg, dim=-1), 1).item())

    def generate_traced(
        self,
        text_ids,
        cond_latents,
        max_seq,
        max_new_tokens=MAX_AUDIO_TOKENS,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
    ):
        """REAL autoregressive text->audio generation via the captured static-KV decode trace — the
        trace-enabled analogue of ``generate`` FOLLOWING THE SAME sampling path (rep-penalty /
        temperature / top-k / top-p, ``temperature <= 0`` = greedy), done on HOST between
        ``execute_trace`` steps (on-device sampling can't be traced). Self-terminates at STOP.
        Returns ``(codes [1, T], latents [1, T, hidden])`` like ``generate``."""
        self._capture_decode_trace(text_ids, cond_latents, max_seq)
        return self._decode_loop(max_new_tokens, temperature, top_k, top_p, repetition_penalty, min_new_tokens)

    def generate_on_static_kv(
        self,
        prompt_len,
        max_new_tokens=MAX_AUDIO_TOKENS,
        temperature=0.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
    ):
        """Like ``generate_traced`` but the caches (``self.model._static_kv``) are ALREADY seeded —
        e.g. by a captured SETUP trace's ``prefill_dev``. Captures the decode step on those caches
        and runs the same host-sampling loop. Used by the fully-traced pipeline."""
        self._capture_decode_on(self.model._static_kv, prompt_len, self.model.max_seq)
        return self._decode_loop(max_new_tokens, temperature, top_k, top_p, repetition_penalty, min_new_tokens)

    def generate_ondevice_traced(self, prompt_len, max_new_tokens, temperature, top_k, top_p, repetition_penalty):
        """FULLY on-device, end-to-end traceable decode: one captured step — decode_static +
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
            logits, latent = m.decode_static(tok_buf, mp_buf, cpos_buf, m._static_kv)  # kv updated in place
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

    def _decode_loop(self, max_new_tokens, temperature, top_k, top_p, repetition_penalty, min_new_tokens):
        """Host-sampling autoregressive loop over the (already-captured) decode-step trace. Releases
        the trace at the end. Returns ``(codes [1, T], latents [1, T, hidden] or None)``."""
        seen = set()

        def pick(logits_dev, n_done):
            tok = self._host_pick(
                logits_dev, seen, n_done, min_new_tokens, temperature, top_k, top_p, repetition_penalty
            )
            if repetition_penalty != 1.0:
                seen.add(tok)
            return tok

        logits_dev, _ = self._decode_step_traced(START_AUDIO_TOKEN, 0)  # start -> c0
        c = pick(logits_dev, 0)
        codes, host_latents, step = [], [], 1
        if c != STOP_AUDIO_TOKEN:  # first-token STOP => generate nothing (matches generate())
            while True:
                logits_dev, latent_dev = self._decode_step_traced(c, step)
                codes.append(c)
                host_latents.append(ttnn.to_torch(latent_dev).float().clone())
                nxt = pick(logits_dev, len(codes))
                if nxt == STOP_AUDIO_TOKEN or len(codes) >= max_new_tokens:
                    break
                c = nxt
                step += 1
        self._release_decode_trace()
        if not host_latents:
            return torch.tensor([codes], dtype=torch.long), None
        latents = torch.cat(host_latents, dim=1)
        lat_dev = ttnn.from_torch(
            latents.to(torch.bfloat16), device=self.model.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        return torch.tensor([codes], dtype=torch.long), lat_dev
