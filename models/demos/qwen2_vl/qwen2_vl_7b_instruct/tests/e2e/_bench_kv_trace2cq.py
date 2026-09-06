# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tier-2 REAL KV-cache trace+2CQ decode bench for Qwen2-VL-7B-Instruct.

Times a full N-token greedy decode three ways on-device, best-of-3 after warm-up:

  full-seq eager : `generate()` -- recomputes the whole 28-layer tower over the
                   entire capacity C every token (no cache).
  KV eager       : `generate_kv()` -- prefill once, then seq=1 decode steps that
                   attend a fixed-capacity K/V cache (O(1) per step compute).
  KV traced+2CQ  : the SAME seq=1 decode step captured ONCE into a trace bound to
                   persistent buffers (emb / cos / sin / mask / onehot + the K/V
                   caches). Each token stages tiny per-step inputs then a single
                   `ttnn.execute_trace`. The per-step K/V write is a one-hot
                   device-index select, so one trace is valid at every position.
                   Device opened with 2 command queues.

Correctness gate: the traced token stream MUST equal the KV-eager token stream.

Run:
    ./python_env/bin/python -m \
      models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e._bench_kv_trace2cq
"""

from __future__ import annotations

import os
import time

import torch

import ttnn
from models.demos.qwen2_vl.qwen2_vl_7b_instruct.tests.e2e import _golden
from models.demos.qwen2_vl.qwen2_vl_7b_instruct.tt.pipeline import build_pipeline

# Capacity + horizon are env-tunable so we can show the KV/trace payoff scaling
# with context length. Default = the validated demo point (C=64, N=16).
CAP = int(os.environ.get("QV_CAP", "64"))
N = int(os.environ.get("QV_NTOK", "16"))


def _stage(host, buf):
    """Upload a host tensor into a persistent device buffer (no device alloc)."""
    ttnn.copy_host_to_device_tensor(
        ttnn.from_torch(host.float().contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT), buf
    )


def _f32(shape, device):
    return ttnn.from_torch(torch.zeros(*shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)


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
        pipe.capacity = CAP
        H, C, hd = pipe._hidden_size, pipe.capacity, pipe._head_dim
        S = int(inputs["input_ids"].shape[1])
        assert S + N <= C, f"need C >= S+N ({S}+{N})"
        print(f"prompt_len S={S}  capacity C={C}  decode horizon N={N}", flush=True)

        # ---------------- full-seq eager (best of 3) --------------------------
        pipe.generate(inputs, max_new_tokens=N)  # warm
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            fullseq_tokens, _ = pipe.generate(inputs, max_new_tokens=N)
            ttnn.synchronize_device(device)
            t.append(time.perf_counter() - t0)
        t_fullseq = min(t)

        # ---------------- KV-cache eager (best of 3) --------------------------
        pipe.generate_kv(inputs, max_new_tokens=N)  # warm
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            kv_tokens, _ = pipe.generate_kv(inputs, max_new_tokens=N)
            ttnn.synchronize_device(device)
            t.append(time.perf_counter() - t0)
        t_kv = min(t)

        # ---------------- KV-cache traced + 2CQ -------------------------------
        # Resident buffers the captured decode step reads (all persistent addrs).
        st = pipe._kv_setup(inputs, inputs["input_ids"].clone())
        cos_host = ttnn.to_torch(st["cos_full"]).float()  # (1,C,hd)
        sin_host = ttnn.to_torch(st["sin_full"]).float()
        emb_buf = ttnn.from_torch(torch.zeros(1, 1, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        cos_buf, sin_buf = _f32((1, 1, hd), device), _f32((1, 1, hd), device)
        mask_buf = _f32((1, 1, 1, C), device)
        oh_buf = _f32((1, 1, C, 1), device)
        log_buf = _f32((1, 1, pipe._vocab_size), device)

        def dmask(p):
            m = torch.zeros(1, 1, 1, C)
            m[..., p + 1 :] = float("-inf")
            return m

        def onehot(p):
            o = torch.zeros(1, 1, C, 1)
            o[0, 0, p, 0] = 1.0
            return o

        def stage_decode(p):
            _stage(cos_host[:, p : p + 1, :], cos_buf)
            _stage(sin_host[:, p : p + 1, :], sin_buf)
            _stage(dmask(p), mask_buf)
            _stage(onehot(p), oh_buf)

        def decode_step_traced():
            logits = pipe._kv_decode_step(
                emb_buf, 0, write_onehot=oh_buf, cos_1=cos_buf, sin_1=sin_buf, mask_1=mask_buf
            )
            ttnn.copy(logits, log_buf)

        def seed_and_first():
            """Eager prefill (seeds caches) + first token; returns (tok_dev, p)."""
            logits0 = pipe._kv_prefill()
            tok_dev = ttnn.argmax(logits0, dim=-1)
            return tok_dev, S

        # warm-up (compile decode-step programs into cache) OUTSIDE capture
        tok_dev, p = seed_and_first()
        emb0 = ttnn.typecast(
            ttnn.reshape(ttnn.embedding(tok_dev, pipe._embed_w, layout=ttnn.TILE_LAYOUT), (1, 1, H)), ttnn.bfloat16
        )
        ttnn.copy(emb0, emb_buf)
        stage_decode(p)
        decode_step_traced()
        ttnn.synchronize_device(device)
        _ = ttnn.to_torch(log_buf)

        # capture the seq=1 decode step ONCE
        cap0 = time.perf_counter()
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        decode_step_traced()
        ttnn.end_trace_capture(device, tid, cq_id=0)
        ttnn.synchronize_device(device)
        t_capture = time.perf_counter() - cap0

        def traced_generate():
            tok_dev, p = seed_and_first()  # eager prefill re-seed (part of a generation)
            tok = int(ttnn.to_torch(tok_dev).flatten()[-1])
            tokens = [tok]
            while len(tokens) < N and p < C:
                emb = ttnn.typecast(
                    ttnn.reshape(ttnn.embedding(tok_dev, pipe._embed_w, layout=ttnn.TILE_LAYOUT), (1, 1, H)),
                    ttnn.bfloat16,
                )
                ttnn.copy(emb, emb_buf)
                stage_decode(p)
                ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
                p += 1
                tok_dev = ttnn.argmax(log_buf, dim=-1)
                tok = int(ttnn.to_torch(tok_dev).flatten()[-1])
                tokens.append(tok)
            ttnn.synchronize_device(device)
            return tokens

        traced_tokens = traced_generate()  # warm
        t = []
        for _ in range(3):
            t0 = time.perf_counter()
            traced_tokens = traced_generate()
            t.append(time.perf_counter() - t0)
        t_traced = min(t)
        ttnn.release_trace(device, tid)

        match = kv_tokens[:N] == traced_tokens[:N]
        print("=" * 78, flush=True)
        print(f"Qwen2-VL-7B  KV-cache decode bench  (N={N} tokens, capacity C={C})", flush=True)
        print("=" * 78, flush=True)
        print(
            f"full-seq eager : {t_fullseq*1e3:8.1f} ms | {t_fullseq/N*1e3:6.1f} ms/tok | {N/t_fullseq:6.2f} tok/s",
            flush=True,
        )
        print(
            f"KV-cache eager : {t_kv*1e3:8.1f} ms | {t_kv/N*1e3:6.1f} ms/tok | {N/t_kv:6.2f} tok/s   ({t_fullseq/t_kv:.2f}x vs full-seq)",
            flush=True,
        )
        print(
            f"KV traced+2CQ  : {t_traced*1e3:8.1f} ms | {t_traced/N*1e3:6.1f} ms/tok | {N/t_traced:6.2f} tok/s   ({t_fullseq/t_traced:.2f}x vs full-seq, {t_kv/t_traced:.2f}x vs KV eager)",
            flush=True,
        )
        print(f"one-time trace capture: {t_capture*1e3:.0f} ms (amortized)", flush=True)
        print(
            f"correctness: traced tokens {'==' if match else '!='} KV-eager tokens -> {'PASS' if match else 'FAIL'}",
            flush=True,
        )
        print(f"  KV eager: {kv_tokens[:N]}", flush=True)
        print(f"  traced  : {traced_tokens[:N]}", flush=True)
        assert match, "traced token stream diverged from KV-eager"
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
