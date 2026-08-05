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

import time

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
        # ttnn.argmax runs directly on the TILE logits (verified index-identical to the ROW_MAJOR
        # path) — drops a per-token full-tensor untilize (to_layout) from the decode hot loop.
        idx = ttnn.argmax(logits, dim=-1)
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

    def generate_ondevice_traced(
        self, prompt_len, max_new_tokens, temperature, top_k, top_p, repetition_penalty, min_new_tokens=0
    ):
        """FULLY on-device, end-to-end traceable decode: one captured step — decode_on_device +
        on-device Gumbel-max sampling (``TtSampler.pick_dev`` over PRE-DRAWN host noise, no
        ``ttnn.rand``) + in-place token feedback (``ttnn.copy``) + on-device latent/code accumulation
        (onehot writes) — replayed up to ``max_new_tokens`` times. Requires ``self.model._static_kv``
        already seeded (by the setup trace/prefill). Reads codes+latents ONCE at the end and trims
        at the first STOP. Returns ``(codes, latents, decode_replay_s)`` where ``decode_replay_s``
        is the replay loop only (warmup + capture excluded).

        Per-step counters (mel pos, cache pos, slot index) and the Gumbel / STOP-bias rows are
        advanced **inside** the captured program from a device step counter, so the replay loop
        does not ``copy_host_to_device_tensor`` between ``execute_trace`` calls — only a cheap
        ``tok_buf`` read for early STOP exit.

        ``min_new_tokens`` suppresses STOP below that many codes via a precomputed per-step bias
        table (same effect as the old between-replay STOP mask rewrite). ``max_new_tokens`` is
        only an upper bound / buffer size — unused steps are not paid for once STOP is seen."""
        m = self.model
        dev = m.device
        N = int(max_new_tokens)
        V = NUM_AUDIO_TOKENS
        sampler = TtSampler(dev, V, temperature, top_k, repetition_penalty, top_p)
        T32 = ttnn.TILE_LAYOUT

        def f32(t):
            return ttnn.from_torch(t, device=dev, dtype=ttnn.float32, layout=T32)

        tok_buf = m._pos_ids(START_AUDIO_TOKEN)  # [1,1] uint32 (embedding input; fed back in place)
        # Device step counter: advanced in-graph at the end of each captured step so replays need
        # no host counter writes. Mel-pos / cache-pos / slot one-hots / noise rows all derive from it.
        step_f = f32(torch.zeros(1, 1))  # [1,1] fp32
        cpos_buf = m.cache_pos(prompt_len)  # [1,1,1,max_seq] fp32; +=1 each step in-graph
        arange_col = f32(torch.arange(N, dtype=torch.float32).reshape(1, N))  # [1,N] slot selector
        latents_buf = ttnn.from_torch(torch.zeros(1, N, HIDDEN_SIZE), device=dev, dtype=ttnn.bfloat16, layout=T32)
        codes_buf = f32(torch.zeros(1, N))  # fp32 so code ids > 256 stay exact

        # Pre-draw ALL Gumbel noise on HOST once, then keep the full [N,V] table on device. Each
        # captured step selects row ``step`` with a matmul against the step one-hot — no per-step
        # host->device noise stream.
        noise_buf = None
        gumbel_dev = None
        if temperature and temperature > 0.0:
            u = torch.rand(N, V).clamp_(1e-4, 1.0 - 1e-3)
            gumbel_all = -torch.log(-torch.log(u))  # [N, V] Gumbel(0,1)
            gumbel_dev = f32(gumbel_all)  # [N, V]
            noise_buf = f32(torch.zeros(1, V))  # [1, V] persistent; filled in-graph each step

        # STOP-suppression: precompute per-step bias rows ( -inf at STOP for steps < floor ).
        # Selecting the row in-graph replaces the old single host rewrite at step ``floor``.
        floor = min(int(min_new_tokens), N)
        stop_bias_buf = None
        bias_dev = None
        if floor > 0:
            bias_all = torch.zeros(N, V)
            bias_all[:floor, STOP_AUDIO_TOKEN] = -1e30
            bias_dev = f32(bias_all)  # [N, V]
            stop_bias_buf = f32(torch.zeros(1, V))  # [1, V] persistent; filled in-graph each step

        def _select_row(table_nv, oh_1n, dest_1v):
            """``oh_1n @ table_nv -> dest_1v`` in place (same buffer address for traced pick_dev)."""
            ttnn.copy(ttnn.matmul(oh_1n, table_nv), dest_1v)

        def step_ops():
            # One-hot over the N decode slots from the device step counter (broadcast eq).
            oh_c = ttnn.typecast(ttnn.eq(arange_col, step_f), ttnn.float32)  # [1, N]
            oh_r = ttnn.reshape(ttnn.typecast(oh_c, ttnn.bfloat16), [1, N, 1])  # [1, N, 1]
            if gumbel_dev is not None:
                _select_row(gumbel_dev, oh_c, noise_buf)
            if bias_dev is not None:
                _select_row(bias_dev, oh_c, stop_bias_buf)

            # Mel position = step (uint32 ROW_MAJOR for embedding). Cache pos already holds
            # prompt_len+step from the in-graph +=1 chain (initialized to prompt_len).
            mp = ttnn.to_layout(ttnn.typecast(step_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
            logits, latent = m.decode_on_device(tok_buf, mp, cpos_buf, m._static_kv)  # kv updated in place
            tok = sampler.pick_dev(logits, noise_buf, stop_bias_buf)  # [1,1] uint32, sampled on device
            ttnn.copy(tok, tok_buf)  # on-device token feedback -> next step's embedding
            ttnn.multiply(latents_buf, ttnn.add(ttnn.multiply(oh_r, -1.0), 1.0), output_tensor=latents_buf)
            ttnn.add(latents_buf, ttnn.multiply(latent, oh_r), output_tensor=latents_buf)
            ttnn.multiply(codes_buf, ttnn.add(ttnn.multiply(oh_c, -1.0), 1.0), output_tensor=codes_buf)
            ttnn.add(codes_buf, ttnn.multiply(ttnn.typecast(tok, ttnn.float32), oh_c), output_tensor=codes_buf)
            # Advance counters for the next replay (captured; no host writes between executes).
            ttnn.add(step_f, 1.0, output_tensor=step_f)
            ttnn.add(cpos_buf, 1.0, output_tensor=cpos_buf)

        def _reset_step_state():
            """Restore step/cache/tok/accumulators after warmup or capture (those runs advance state)."""
            ttnn.multiply(step_f, 0.0, output_tensor=step_f)
            ttnn.copy(m.cache_pos(prompt_len), cpos_buf)
            ttnn.copy(m._pos_ids(START_AUDIO_TOKEN), tok_buf)
            ttnn.multiply(latents_buf, 0.0, output_tensor=latents_buf)
            ttnn.multiply(codes_buf, 0.0, output_tensor=codes_buf)
            sampler.reset()

        # Warmup (compile). It executes one step at position prompt_len — harmless: real step 0
        # rewrites that cache slot; we reset below before capture.
        step_ops()
        ttnn.synchronize_device(dev)
        _reset_step_state()
        # Capture the one static-shape sample-step (runs step 0, advances counters to 1).
        tid = ttnn.begin_trace_capture(dev, cq_id=0)
        step_ops()
        ttnn.end_trace_capture(dev, tid, cq_id=0)
        ttnn.synchronize_device(dev)
        # Capture left step=1 / dirty accumulators — reset so the first replay is real step 0.
        # (Matches prior host-written i=0..N-1 semantics; tok/seen reset included.)
        _reset_step_state()

        # Replay-only: execute_trace with on-device counter/noise updates inside the program.
        # Early-exit: read the just-sampled token from tok_buf; stop on STOP past the floor.
        t_replay = time.perf_counter()
        steps_run = 0
        for i in range(N):
            ttnn.execute_trace(dev, tid, blocking=True)
            steps_run = i + 1
            if i < floor:
                continue  # STOP still suppressed in the bias table; skip the host check
            tok_id = int(ttnn.to_torch(tok_buf).reshape(-1)[0].item())
            if tok_id == STOP_AUDIO_TOKEN:
                break
        decode_replay_s = time.perf_counter() - t_replay
        ttnn.release_trace(dev, tid)

        # Read once. code c_i is sampled at step i; latent slot i is the fed-token latent (START at
        # slot 0), so c_j's latent is slot j+1. Trim at the first STOP (or steps_run if no STOP).
        codes = ttnn.to_torch(codes_buf).float().round().to(torch.long).flatten().tolist()[:steps_run]
        lat = ttnn.to_torch(latents_buf).float()
        stop = next((i for i, c in enumerate(codes) if c == STOP_AUDIO_TOKEN), len(codes))
        seq = codes[:stop]
        # If the model hits the max-token cap without STOP it can drone on a repeated code. Trim
        # that trailing same-code run to a short hold (early STOP exit usually avoids this).
        if seq:
            run = 1
            while run < len(seq) and seq[-1 - run] == seq[-1]:
                run += 1
            if run > 8:
                seq = seq[: len(seq) - run + 2]  # keep ~2 for a natural final hold
        cut = min(max(len(seq), 1), max(steps_run - 1, 1))
        codes_out = torch.tensor([codes[:cut]], dtype=torch.long)
        lat_out = ttnn.from_torch(
            lat[:, 1 : cut + 1, :].to(torch.bfloat16), device=dev, dtype=ttnn.bfloat16, layout=T32
        )
        return codes_out, lat_out, decode_replay_s
