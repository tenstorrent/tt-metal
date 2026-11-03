# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Prefill and decode workload for optimizer runs on Ternary-Bonsai-2-27B (dense conversion), one P150, batch 1.

The prompt is the gate's 4096-token reference prompt (bonsai_support.ISL, two full prefill chunks).

BONSAI_OPT_STAGE picks the window (default decode):
- prefill: the chunk prefill trace is captured, one untimed traced prefill runs (it compiles the last-token norm and
  LM head), then PREFILL_SAMPLES timed traced prefills. Each prints TRACE_STAGE_MS[prefill]=<ms>: wall time from
  the call to the next-token logits on the host, the demo's TTFT.
- decode: one traced prefill, the decode trace primed at the post-prefill state (prime_decode_trace, as the demo),
  then DECODE_CALLS traced greedy decode calls: each replays the decode trace, takes the argmax over the real vocab
  on the device and reads one token id back. Call 0 compiles the argmax and is not timed; TRACE_STAGE_MS[decode] is
  the mean wall time of the other 128 calls.

Under the device profiler (TT_METAL_DEVICE_PROFILER=1, the tool's profile command) nothing is traced, and the model is
built with BONSAI_OPT_PROFILE_LAYERS layers (unset or 0: all 64). The device profiler records every op from the
start, and a full-depth 4096-token eager prefill alone writes a device log of about 4.5 GB per chunk, so:
- prefill: the chunk forwards run eagerly (bonsai_support.prefill_eager), once untimed, then once between
  PREFILL_START and PREFILL_END. The prefill spec profiles 4 layers, one repeat of the model's pattern (3 Gated
  DeltaNet layers, 1 full-attention layer); the tool scales in-layer rows to full depth.
- decode: no prefill runs. One untimed eager decode step at position ISL, then two between DECODE_START and
  DECODE_END, teacher forced from the reference continuation, against the untouched KV cache and GDN state. The
  ops are the same with and without a prior prefill (checked 2026-11-03 on a 4-layer build with ttnn graph
  capture: 1539 calls per step, same sequence); only the values differ, and the profile reads only times.
"""

from __future__ import annotations

import os
import statistics
import time

import pytest
from tracy import signpost

import ttnn
from models.demos.blackhole.qwen36.tests.optimizer_bonsai.bonsai_support import (
    CHUNK,
    DECODE_CALLS,
    DEVICE_PARAMS,
    PREFILL_SAMPLES,
    append_history,
    build_model,
    decode_step,
    device_argmax,
    load_reference,
    make_generator,
    prefill_eager,
    prefill_traced,
    prime_decode,
)

STAGE = os.environ.get("BONSAI_OPT_STAGE", "decode")
PROFILE_DECODE_STEPS = 2


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("mesh_device", [(1, 1)], indirect=True)
@pytest.mark.parametrize("device_params", [DEVICE_PARAMS], indirect=True)
def test_bonsai_perf(mesh_device, device_params):
    assert STAGE in ("prefill", "decode"), f"BONSAI_OPT_STAGE={STAGE!r}: prefill or decode"
    device = mesh_device
    device.enable_program_cache()
    profiling = os.environ.get("TT_METAL_DEVICE_PROFILER") == "1"
    ref = load_reference()
    prompt, cont = ref["prompt_ids"], ref["continuation_ids"]
    P = len(prompt)
    print(f"BONSAI_PROMPT_TOKENS={P}", flush=True)
    print(f"GATE_CONFIG stage={STAGE} profiling={int(profiling)} trace={int(not profiling)}", flush=True)

    if profiling:
        layers = int(os.environ.get("BONSAI_OPT_PROFILE_LAYERS", "0") or 0)
        model, page_table = build_model(device, n_layers=layers or None)
        if STAGE == "prefill":
            model.prepare_prefill_trace_chunked(device, page_table, chunk_size=CHUNK, warmup_masked_buckets=False)
            prefill_eager(model, prompt, page_table)  # compiles the last-token norm and LM head
            ttnn.synchronize_device(device)
            signpost("PREFILL_START")
            prefill_eager(model, prompt, page_table)
            ttnn.synchronize_device(device)
            signpost("PREFILL_END")
            return
        generator = make_generator(model, device)
        decode_step(generator, cont[0], P, page_table, trace=False, read=False)  # compiles the decode programs
        ttnn.synchronize_device(device)
        signpost("DECODE_START")
        for i in range(1, 1 + PROFILE_DECODE_STEPS):
            decode_step(generator, cont[i], P + i, page_table, trace=False, read=False)
        ttnn.synchronize_device(device)
        signpost("DECODE_END")
        return

    model, page_table = build_model(device)
    vocab = model.args.vocab_size
    model.capture_prefill_trace_chunked(device, page_table, chunk_size=CHUNK, warmup_masked_buckets=False)
    logits = prefill_traced(model, prompt, page_table)  # untimed: compiles the last-token norm and LM head
    prefill_ms = []
    for _ in range(PREFILL_SAMPLES if STAGE == "prefill" else 0):
        t = time.perf_counter()
        logits = prefill_traced(model, prompt, page_table)
        first = int(logits[:vocab].argmax())
        prefill_ms.append((time.perf_counter() - t) * 1000.0)
    first = int(logits[:vocab].argmax())
    assert not logits.isnan().any(), "NaN in prefill logits"

    if STAGE == "prefill":
        for ms in prefill_ms:
            print(f"TRACE_STAGE_MS[prefill]={ms:.4f}", flush=True)
        ttft = statistics.median(prefill_ms)
        print(
            f"PERF stage=prefill isl={P} ttft_ms={ttft:.3f} samples_ms={','.join(f'{ms:.3f}' for ms in prefill_ms)} "
            f"prefill_tok_s={P / ttft * 1000.0:.1f} first_token={first}",
            flush=True,
        )
        append_history("OPTIMIZER_PERF_HISTORY", {"stage": "prefill", "ttft_ms": round(ttft, 3), "isl": P})
        return

    generator = make_generator(model, device)
    prime_decode(generator, model, first, P, page_table)
    token, pos, tokens, step_ms = first, P, [first], []
    for call in range(DECODE_CALLS):
        t = time.perf_counter()
        out = decode_step(generator, token, pos, page_table, trace=True, read=False)
        token = device_argmax(out, vocab)
        if call > 0:
            step_ms.append((time.perf_counter() - t) * 1000.0)
        tokens.append(token)
        pos += 1
    per_token_ms = statistics.fmean(step_ms)
    from transformers import AutoTokenizer

    text = AutoTokenizer.from_pretrained(model.args.CKPT_DIR).decode(tokens)
    print(f"GATE_TEXT {text[:300]!r}", flush=True)
    print(f"TRACE_STAGE_MS[decode]={per_token_ms:.4f}", flush=True)
    print(
        f"PERF stage=decode isl={P} decode_ms_per_token={per_token_ms:.4f} decode_median_ms="
        f"{statistics.median(step_ms):.4f} tokens_per_s_per_user={1000.0 / per_token_ms:.3f} timed={len(step_ms)}",
        flush=True,
    )
    append_history(
        "OPTIMIZER_PERF_HISTORY", {"stage": "decode", "decode_ms_per_token": round(per_token_ms, 4), "isl": P}
    )
