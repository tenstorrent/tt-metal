# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

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
    def __init__(self, model):
        """Wrap a GPT model for autoregressive code generation."""
        self.model = model

    def _argmax(self, logits):
        # TILE argmax avoids a per-token untilize in the decode hot loop.
        """Return the argmax token id from logits."""
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
        """Autoregressively generate audio codes and latents."""
        sampler = None
        if temperature and temperature > 0.0:
            sampler = TtSampler(self.model.device, NUM_AUDIO_TOKENS, temperature, top_k, repetition_penalty, top_p)
        pick = sampler.pick if sampler else self._argmax

        stop_mask = None
        if min_new_tokens > 0:
            m = torch.zeros(1, 1, NUM_AUDIO_TOKENS)
            m[0, 0, STOP_AUDIO_TOKEN] = -1e30
            stop_mask = ttnn.from_torch(m, device=self.model.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def _pick(logits, n_done):
            """Pick next token, masking STOP until min_new_tokens."""
            return pick(ttnn.add(logits, stop_mask) if (stop_mask is not None and n_done < min_new_tokens) else logits)

        prompt_len = cond_latents.shape[1] + text_ids.shape[1]
        max_seq = -(-(prompt_len + max_new_tokens + 1) // 32) * 32
        kv = self.model.prefill(text_ids, cond_latents, max_seq)
        logits, _, kv = self.model.decode(START_AUDIO_TOKEN, 0, kv)
        c = _pick(logits, 0)

        codes, latents, step = [], [], 1
        # First-token STOP means empty output (match reference; never emit 1025 as a code).
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
        """Teacher-force codes to collect latents and predictions."""
        prompt_len = cond_latents.shape[1] + text_ids.shape[1]
        max_seq = -(-(prompt_len + len(codes) + 2) // 32) * 32
        kv = self.model.prefill(text_ids, cond_latents, max_seq)
        logits, _, kv = self.model.decode(START_AUDIO_TOKEN, 0, kv)
        preds = [self._argmax(logits)]

        latents = []
        for i, code in enumerate(codes):
            logits, latent, kv = self.model.decode(int(code), i + 1, kv)
            latents.append(latent)
            preds.append(self._argmax(logits))
        return preds, ttnn.concat(latents, dim=1)

    def generate_ondevice_traced(
        self, prompt_len, max_new_tokens, temperature, top_k, top_p, repetition_penalty, min_new_tokens=0
    ):
        """Run traced on-device decode and return codes and latents."""
        dec = TtTracedDecoder(
            self.model,
            prompt_len,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            min_new_tokens=min_new_tokens,
        )
        try:
            codes, lat_host, decode_replay_s = dec.run()
            stopped = dec.stopped
        finally:
            dec.release()
        lat_out = ttnn.from_torch(lat_host, device=self.model.device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        return codes, lat_out, decode_replay_s, stopped


class TtTracedDecoder:
    def __init__(
        self,
        model,
        prompt_len,
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        min_new_tokens=0,
        capture=True,
    ):
        """Capture and replay a traced single-step decode loop."""
        self.model = model
        dev = model.device
        self.device = dev
        self.prompt_len = prompt_len
        self.N = N = int(max_new_tokens)
        V = NUM_AUDIO_TOKENS
        self.temperature = temperature
        self.sampler = TtSampler(dev, V, temperature, top_k, repetition_penalty, top_p)
        T32 = ttnn.TILE_LAYOUT

        def f32(t):
            """Upload a host tensor as float32 tiles on device."""
            return ttnn.from_torch(t, device=dev, dtype=ttnn.float32, layout=T32)

        self._f32 = f32
        self.tok_buf = model._pos_ids(START_AUDIO_TOKEN)
        self.step_f = f32(torch.zeros(1, 1))
        self.cpos_buf = model.cache_pos(prompt_len)
        self.arange_col = f32(torch.arange(N, dtype=torch.float32).reshape(1, N))
        self.latents_buf = ttnn.from_torch(torch.zeros(1, N, HIDDEN_SIZE), device=dev, dtype=ttnn.bfloat16, layout=T32)
        self.codes_buf = f32(torch.zeros(1, N))  # fp32 so code ids > 256 stay exact

        # Pre-draw Gumbel [N,V] on host once; select row in-graph (no per-step host->device noise).
        self.noise_buf = None
        self.gumbel_dev = None
        if temperature and temperature > 0.0:
            self.gumbel_dev = f32(self._draw_gumbel())
            self.noise_buf = f32(torch.zeros(1, V))

        self.floor = floor = min(int(min_new_tokens), N)
        self.stop_bias_buf = None
        self.bias_dev = None
        if floor > 0:
            bias_all = torch.zeros(N, V)
            bias_all[:floor, STOP_AUDIO_TOKEN] = -1e30
            self.bias_dev = f32(bias_all)
            self.stop_bias_buf = f32(torch.zeros(1, V))

        # Set by run(): True if the take self-terminated at STOP, False if it ran out of budget
        # (which leaves an unfinished — usually droning or noisy — tail).
        self.stopped = False
        self.tid = None
        if capture:
            self.warmup()
            self.capture()

    def warmup(self):
        """Warm up decode ops and reset buffers."""
        self._step_ops()
        ttnn.synchronize_device(self.device)
        self.reset()

    def capture(self):
        """Capture the decode step into a device trace."""
        self.tid = ttnn.begin_trace_capture(self.device, cq_id=0)
        self._step_ops()
        ttnn.end_trace_capture(self.device, self.tid, cq_id=0)
        ttnn.synchronize_device(self.device)
        self.reset()

    def _draw_gumbel(self):
        """Draw Gumbel noise for all decode steps."""
        u = torch.rand(self.N, NUM_AUDIO_TOKENS).clamp_(1e-4, 1.0 - 1e-3)
        return -torch.log(-torch.log(u))

    def _step_ops(self):
        """Execute one decode step writing codes and latents."""
        m, N = self.model, self.N

        def _select_row(table_nv, oh_1n, dest_1v):
            """Select one row from a table via one-hot matmul."""
            ttnn.copy(ttnn.matmul(oh_1n, table_nv), dest_1v)

        oh_c = ttnn.typecast(ttnn.eq(self.arange_col, self.step_f), ttnn.float32)
        oh_r = ttnn.reshape(ttnn.typecast(oh_c, ttnn.bfloat16), [1, N, 1])
        if self.gumbel_dev is not None:
            _select_row(self.gumbel_dev, oh_c, self.noise_buf)
        if self.bias_dev is not None:
            _select_row(self.bias_dev, oh_c, self.stop_bias_buf)

        mp = ttnn.to_layout(ttnn.typecast(self.step_f, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT)
        logits, latent = m.decode_on_device(self.tok_buf, mp, self.cpos_buf, m._static_kv)
        tok = self.sampler.pick_dev(logits, self.noise_buf, self.stop_bias_buf)
        ttnn.copy(tok, self.tok_buf)
        lb, cb = self.latents_buf, self.codes_buf
        ttnn.multiply(lb, ttnn.add(ttnn.multiply(oh_r, -1.0), 1.0), output_tensor=lb)
        ttnn.add(lb, ttnn.multiply(latent, oh_r), output_tensor=lb)
        ttnn.multiply(cb, ttnn.add(ttnn.multiply(oh_c, -1.0), 1.0), output_tensor=cb)
        ttnn.add(cb, ttnn.multiply(ttnn.typecast(tok, ttnn.float32), oh_c), output_tensor=cb)
        ttnn.add(self.step_f, 1.0, output_tensor=self.step_f)
        ttnn.add(self.cpos_buf, 1.0, output_tensor=self.cpos_buf)

    def reset(self, redraw_noise=False):
        # In-place only: capture bound these addresses; rebinding detaches from the trace.
        """Reset step buffers in place for a new replay."""
        ttnn.multiply(self.step_f, 0.0, output_tensor=self.step_f)
        ttnn.copy(self.model.cache_pos(self.prompt_len), self.cpos_buf)
        ttnn.copy(self.model._pos_ids(START_AUDIO_TOKEN), self.tok_buf)
        ttnn.multiply(self.latents_buf, 0.0, output_tensor=self.latents_buf)
        ttnn.multiply(self.codes_buf, 0.0, output_tensor=self.codes_buf)
        if redraw_noise and self.gumbel_dev is not None:
            ttnn.copy(self._f32(self._draw_gumbel()), self.gumbel_dev)
        self.sampler.reset()

    def run(self):
        """Replay the decode trace until STOP or max tokens."""
        dev, N, floor = self.device, self.N, self.floor
        # Poll every step: less-frequent polls run extra decode past STOP.
        t_replay = time.perf_counter()
        steps_run = 0
        self.stopped = False
        for i in range(N):
            ttnn.execute_trace(dev, self.tid, blocking=True)
            steps_run = i + 1
            if i < floor:
                continue
            tok_id = int(ttnn.to_torch(self.tok_buf).reshape(-1)[0].item())
            if tok_id == STOP_AUDIO_TOKEN:
                self.stopped = True
                break
        decode_replay_s = time.perf_counter() - t_replay

        codes = ttnn.to_torch(self.codes_buf).float().round().to(torch.long).flatten().tolist()[:steps_run]
        lat = ttnn.to_torch(self.latents_buf).float()
        stop = next((i for i, c in enumerate(codes) if c == STOP_AUDIO_TOKEN), len(codes))
        seq = codes[:stop]
        # Cap without STOP can drone; trim trailing same-code run to a short hold.
        if seq:
            run = 1
            while run < len(seq) and seq[-1 - run] == seq[-1]:
                run += 1
            if run > 8:
                seq = seq[: len(seq) - run + 2]
        # No floor at 1: a step-0 STOP means empty output (match eager generate(); never emit 1025
        # as a code). cut is also bounded by steps_run-1 because latents_buf[i] holds the latent of
        # code i-1, so cut codes need cut+1 replayed steps.
        cut = min(len(seq), max(steps_run - 1, 0))
        codes_out = torch.tensor([codes[:cut]], dtype=torch.long)
        return codes_out, lat[:, 1 : cut + 1, :].to(torch.bfloat16), decode_replay_s

    def release(self):
        """Release the decode trace and free scratch tensors."""
        if self.tid is not None:
            ttnn.release_trace(self.device, self.tid)
            self.tid = None
        # Free decode scratch before the vocoder compiles — these sit in L1/DRAM otherwise.
        for name in (
            "latents_buf",
            "codes_buf",
            "gumbel_dev",
            "bias_dev",
            "noise_buf",
            "stop_bias_buf",
            "tok_buf",
            "step_f",
            "cpos_buf",
            "arange_col",
        ):
            t = getattr(self, name, None)
            if t is not None and t.is_allocated():
                ttnn.deallocate(t)
            setattr(self, name, None)
        if getattr(self, "sampler", None) is not None:
            self.sampler.release()
            self.sampler = None
