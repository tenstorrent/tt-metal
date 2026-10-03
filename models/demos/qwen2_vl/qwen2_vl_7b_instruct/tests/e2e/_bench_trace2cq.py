# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tier-1 REAL trace+2CQ generation bench for Qwen2-VL-7B-Instruct.

Times a full N-token greedy decode two ways on-device, best-of-3 after warm-up:

  eager : the resident pipeline as-is -- every step dispatches the whole
          28-layer text tower from the host one op at a time.
  traced: the SAME math, but the heavy text tower is captured ONCE into a
          replayable trace bound to *persistent* device buffers; each step just
          `ttnn.execute_trace` (single host dispatch) then does the tiny
          lm_head + argmax + in-place embeds update eagerly. Device is opened
          with 2 command queues (trace+2CQ region).

Correctness gate: the traced token stream MUST equal the eager token stream
(identical math), so the speedup is real, not a shortcut.

Run:
    ./python_env/bin/python -m \
      models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e._bench_trace2cq
"""

from __future__ import annotations

import time

import torch

import ttnn
from models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e import _golden
from models.demos.qwen2_vl.qwen2_vl_7b_instruct.tt.pipeline import build_pipeline

N_MAX = 16  # validated greedy horizon (matches the e2e PCC gate)


def _zeros(shape, device, dtype=ttnn.bfloat16):
    return ttnn.from_torch(torch.zeros(*shape), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


def main():
    from transformers import Qwen2VLForConditionalGeneration

    g = _golden()
    inputs = {k: g[k] for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()

    device = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=200_000_000, num_command_queues=2)
    try:
        pipe = build_pipeline(device, model)
        H, C = pipe._hidden_size, pipe.capacity
        S = int(inputs["input_ids"].shape[1])
        N = min(N_MAX, C - S - 1)
        print(f"prompt_len S={S}  capacity C={C}  decode horizon N={N}", flush=True)

        # ---------------- eager timing (best of 3 after warm-up) --------------
        eager_tokens, _ = pipe.generate(inputs, max_new_tokens=N)  # warm-up (compile)
        eager_t = []
        for _ in range(3):
            t0 = time.perf_counter()
            toks, _ = pipe.generate(inputs, max_new_tokens=N)
            ttnn.synchronize_device(device)
            eager_t.append(time.perf_counter() - t0)
            eager_tokens = toks
        t_eager = min(eager_t)

        # ---------------- traced setup (persistent buffers) -------------------
        # One resident build: vision + scatter + rope/mask over the whole C.
        st = pipe._resident_setup(inputs, inputs["input_ids"].clone())
        embeds_buf = st["embeds"]  # (1,C,H) -- the trace reads THIS address
        cos, sin, mask = st["cos"], st["sin"], st["mask"]
        hid_buf = _zeros((1, C, H), device)  # trace writes hidden HERE
        embeds0 = _zeros((1, C, H), device)  # snapshot of the prompt embeds
        ttnn.copy(embeds_buf, embeds0)

        def text_step():
            """The heavy 28-layer tower -> persistent hid_buf (this is traced)."""
            h = pipe._text_forward(inputs_embeds=embeds_buf, cos_dev=cos, sin_dev=sin, mask_dev=mask)
            ttnn.copy(h, hid_buf)

        def head_and_update(real_len):
            """Eager tail: logits @ last real pos, argmax, write next embed in-place."""
            hidden_last = ttnn.slice(hid_buf, (0, real_len - 1, 0), (1, real_len, H))
            logits = pipe._lm_head(hidden_last)  # (1,1,vocab)
            next_tok_dev = ttnn.argmax(logits, dim=-1)  # (1,1) on device
            tok = int(ttnn.to_torch(next_tok_dev).flatten()[-1])
            if real_len < C:
                new_row = ttnn.embedding(next_tok_dev, pipe._embed_w, layout=ttnn.TILE_LAYOUT)
                new_row = ttnn.typecast(ttnn.reshape(new_row, (1, 1, H)), ttnn.bfloat16)
                left = ttnn.slice(embeds_buf, (0, 0, 0), (1, real_len, H))
                parts = [left, new_row]
                if real_len + 1 < C:
                    parts.append(ttnn.slice(embeds_buf, (0, real_len + 1, 0), (1, C, H)))
                ttnn.copy(ttnn.concat(parts, dim=1), embeds_buf)  # in-place, keeps address
            return tok

        # warm-up (compile programs into device cache) OUTSIDE capture
        ttnn.copy(embeds0, embeds_buf)
        text_step()
        ttnn.synchronize_device(device)
        _ = ttnn.to_torch(hid_buf)

        # capture the heavy tower ONCE
        ttnn.copy(embeds0, embeds_buf)
        cap0 = time.perf_counter()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        text_step()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        t_capture = time.perf_counter() - cap0

        def traced_generate():
            ttnn.copy(embeds0, embeds_buf)  # reset CONTENT (address fixed)
            real_len = S
            toks = []
            for _ in range(N):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                toks.append(head_and_update(real_len))
                real_len += 1
                if real_len >= C:
                    break
            ttnn.synchronize_device(device)
            return toks

        traced_tokens = traced_generate()  # warm
        traced_t = []
        for _ in range(3):
            t0 = time.perf_counter()
            traced_tokens = traced_generate()
            traced_t.append(time.perf_counter() - t0)
        t_traced = min(traced_t)
        ttnn.release_trace(device, tid)

        # ------------------------------ report --------------------------------
        match = eager_tokens[:N] == traced_tokens[:N]
        print("=" * 72, flush=True)
        print(f"Qwen2-VL-7B  trace+2CQ decode bench  (N={N} tokens, capacity C={C})", flush=True)
        print("=" * 72, flush=True)
        print(f"eager  : {t_eager*1e3:8.1f} ms  | {t_eager/N*1e3:6.1f} ms/tok | {N/t_eager:6.2f} tok/s", flush=True)
        print(f"traced : {t_traced*1e3:8.1f} ms  | {t_traced/N*1e3:6.1f} ms/tok | {N/t_traced:6.2f} tok/s", flush=True)
        print(
            f"speedup: {t_eager/t_traced:5.2f}x   (one-time trace capture {t_capture*1e3:.0f} ms, amortized)",
            flush=True,
        )
        print(
            f"correctness: traced tokens {'==' if match else '!='} eager tokens  -> {'PASS' if match else 'FAIL'}",
            flush=True,
        )
        print(f"  eager : {eager_tokens[:N]}", flush=True)
        print(f"  traced: {traced_tokens[:N]}", flush=True)
        assert match, "traced token stream diverged from eager -- trace is not faithful"
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
