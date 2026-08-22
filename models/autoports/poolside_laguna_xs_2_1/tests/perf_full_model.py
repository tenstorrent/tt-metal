# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full-model performance on a selected P150 D=1/2/4 profile.

Batch-1 prompt128/gen128 is the default vLLM single-user profile. Three decode metrics, each a
genuinely different captured trace, are measured with one
trace resident at a time (capture -> measure -> release):

  * logits-only decode  — embed -> 40 layers -> final norm -> LM head (sampler-ready logits), NO
    Sampling1D, NO token feedback. Positions advance on device. This is the true logits-only /
    PERF-style / teacher-forcing-logits comparison and the fair decoder-floor number.
  * token-out decode (no readback) — logits-only + Sampling1D top-k(k=1) split sampling + on-device
    token feedback + on-device position advance; N non-blocking replays + one sync. This is the
    on-device serving throughput with NO host token work.
  * token-out decode (+readback) — same, but per-token blocking replay + single-token readback (the
    readback a host caller/vLLM needs).

sampler+feedback device cost = token-out(no readback) - logits-only. Also runs a batch>1 low-level
prefill+decode smoke.

  cd /tmp && TT_METAL_HOME=<tree> PYTHONPATH=<repo> python -m \
    models.autoports.poolside_laguna_xs_2_1.tests.perf_full_model --prompt 128 --gen 128
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.poolside_laguna_xs_2_1.tt.generator import LagunaGenerator
from models.autoports.poolside_laguna_xs_2_1.tests.laguna_test_utils import (
    add_profile_args,
    assert_memory_margin,
    close_mesh,
    open_mesh,
    parse_layers,
    print_memory_snapshot,
    profile_from_args,
    profile_summary,
)


def _capture(mesh, body):
    body()  # compile
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    body()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    return tid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=int, default=128)
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--layers", type=str, default=None)
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="KV/context allocation (default: selected profile's qualified maximum)",
    )
    ap.add_argument("--enforce-memory-margin", action="store_true")
    ap.add_argument("--batch2-smoke", action="store_true", help="run the out-of-contract batch-2 diagnostic")
    ap.add_argument("--out", type=str, default=None)
    add_profile_args(ap)
    args = ap.parse_args()
    layers = parse_layers(args.layers)
    profile = profile_from_args(args)
    max_seq_len = args.max_seq_len or profile.max_context
    print("PROFILE", json.dumps(profile_summary(profile), sort_keys=True))

    mesh = open_mesh(ttnn, profile)
    gen = None
    memory = []

    def snapshot(label):
        value = print_memory_snapshot(ttnn, mesh, label)
        memory.append(value)
        if args.enforce_memory_margin:
            assert_memory_margin(value)

    try:
        if max_seq_len < args.prompt + 4 * args.gen + 64:
            raise ValueError("max-seq-len is too small for prompt plus performance iterations")
        gen = LagunaGenerator.from_pretrained(mesh, max_seq_len=max_seq_len, num_layers=layers)
        snapshot("weights")
        P, G = args.prompt, args.gen
        m = gen.model
        torch.manual_seed(0)
        prompt = torch.randint(0, gen.vocab, (P,), dtype=torch.int64).tolist()
        gen._ensure_cache(1, max_seq_len)
        kv, pt = gen._kv_cache, gen._page_table
        snapshot("uniform_kv")

        def prefill_ttft():
            x = m.embed_prefill(gen._tokens_to_device(torch.tensor(prompt)))
            h = m.prefill_layers(x, kv, pt, user_id=0, start_pos=0)
            last = ttnn.slice(h, [0, P - 1, 0], [1, P, gen.hidden])
            tb = gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
            gen._greedy_sample(m.lm_head_shards_decode(ttnn.reshape(last, (1, 1, 1, gen.hidden))), 1, tb)
            return tb

        chosen_tb = prefill_ttft()  # warm
        snapshot("prefill_warmup")
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        prefill_ttft()
        ttnn.synchronize_device(mesh)
        ttft_ms = (time.perf_counter() - t0) * 1e3

        # persistent decode tensors (all allocations happen here, before any trace capture)
        tok = gen._rep(torch.zeros([1, 1, 1, 1], dtype=torch.int32), ttnn.uint32)
        cur = gen._rep(torch.zeros([1], dtype=torch.int32), ttnn.int32)
        ridx = gen._rep(torch.zeros([1, 1], dtype=torch.int32), ttnn.uint32)
        _hold = {}

        fixed_tok = int(prompt[-1])  # any valid token id; throughput is token-independent

        def stage(pos):
            ttnn.copy_host_to_device_tensor(gen._host_rank4_tok(fixed_tok), tok)
            ttnn.copy_host_to_device_tensor(gen._host_pos(pos), cur)
            ttnn.copy_host_to_device_tensor(gen._host_ridx(pos), ridx)

        def step_logits():
            h = m.embed_decode(ttnn.reshape(tok, (1, 1)))
            h = m.decode_layers(h, cur, ridx, pt, kv)
            _hold["l"] = m.lm_head_shards_decode(h)  # sampler-ready logits; keep ref so capture holds it
            ttnn.plus_one(cur, skip_negative_entries=True)
            ttnn.plus_one(ridx)

        def step_tokenout():
            h = m.embed_decode(ttnn.reshape(tok, (1, 1)))
            h = m.decode_layers(h, cur, ridx, pt, kv)
            gen._greedy_sample(m.lm_head_shards_decode(h), 1, tok)
            ttnn.plus_one(cur, skip_negative_entries=True)
            ttnn.plus_one(ridx)

        def measure(tid, blocking, readback):
            stage(P)
            ttnn.synchronize_device(mesh)
            t = time.perf_counter()
            for _ in range(G):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=blocking)
                if readback:
                    gen._read_token(tok, 1)
            if not blocking:
                ttnn.synchronize_device(mesh)
            return (time.perf_counter() - t) / G * 1e3

        # --- logits-only trace (capture -> warm -> measure -> release) ---
        stage(P)
        tid_l = _capture(mesh, step_logits)
        snapshot("logits_trace_capture")
        stage(P)
        for _ in range(8):
            ttnn.execute_trace(mesh, tid_l, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        logits_only_ms = measure(tid_l, blocking=False, readback=False)
        ttnn.release_trace(mesh, tid_l)

        # --- token-out trace (with Sampling1D + feedback) ---
        stage(P)
        tid_t = _capture(mesh, step_tokenout)
        snapshot("token_trace_capture")
        stage(P)
        for _ in range(8):
            ttnn.execute_trace(mesh, tid_t, cq_id=0, blocking=True)
        ttnn.synchronize_device(mesh)
        tokenout_ms = measure(tid_t, blocking=False, readback=False)
        tokenout_rb_ms = measure(tid_t, blocking=True, readback=True)
        ttnn.release_trace(mesh, tid_t)

        # Batch one is the qualified contract; batch two remains an opt-in diagnostic.
        B = 2
        batch_ok = "not-run (batch-1 qualification profile)"
        if args.batch2_smoke:
            try:
                kv2 = m.alloc_kv_cache(max_users=B, max_seq_len=P + 8, block_size=32)
                pt2 = m.make_page_table(B, kv2[0]["blocks_per_user"])
                for u in range(B):
                    gen.prefill_forward(
                        tokens=torch.tensor([prompt[:64]]), page_table=pt2, kv_cache=kv2, prompt_lens=[64], user_id=u
                    )
                dec = gen.decode_forward(
                    torch.tensor([[prompt[0]], [prompt[1]]]), torch.tensor([64, 64]), page_table=pt2, kv_cache=kv2
                )
                batch_ok = bool(torch.as_tensor(dec).numel() == B and torch.isfinite(torch.as_tensor(dec).float()).all())
            except Exception as e:
                batch_ok = f"FAILED: {repr(e)[:160]}"

        res = {
            **profile_summary(profile),
            "layers": layers or "all-40",
            "max_seq_len": max_seq_len,
            "workload": f"prompt{P}/gen{G}",
            "ttft_ms": round(ttft_ms, 1),
            "logits_only_decode_ms_tok": round(logits_only_ms, 3),
            "logits_only_decode_tsu": round(1e3 / logits_only_ms, 2),
            "token_out_decode_ms_tok": round(tokenout_ms, 3),
            "token_out_decode_tsu": round(1e3 / tokenout_ms, 2),
            "token_out_plus_readback_ms_tok": round(tokenout_rb_ms, 3),
            "token_out_plus_readback_tsu": round(1e3 / tokenout_rb_ms, 2),
            "sampler_plus_feedback_ms_tok": round(tokenout_ms - logits_only_ms, 3),
            "readback_ms_tok": round(tokenout_rb_ms - tokenout_ms, 3),
            "batch2_lowlevel_prefill_decode_ok": batch_ok,
            "memory": memory,
        }
        print("PERF", json.dumps(res))
        if args.out:
            Path(args.out).write_text(json.dumps(res, indent=2))
    finally:
        try:
            if gen is not None:
                gen.teardown()
        except Exception:
            pass
        close_mesh(ttnn, mesh)


if __name__ == "__main__":
    import sys as _sys

    main()
    _sys.stdout.flush()
    _sys.stderr.flush()
    if "--layers" not in _sys.argv:
        import os as _os

        _os._exit(0)
