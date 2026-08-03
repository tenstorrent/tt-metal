# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Standalone Laguna-XS-2.1 ngram speculative-decode driver (batch-1), no vLLM server.

Mirrors the gemma4 generator-level spec-decode validation (models/demos/gemma4/
tests/unit/test_spec_decode.py) adapted to Laguna + ngram drafts. Stands up the
full model on the 1x4 mesh via the vLLM adapter (LagunaForCausalLM), then:

  ACCURACY  — prefill a prompt, run plain-greedy decode (single-token verify) as
              the reference, RE-PREFILL, run ngram spec-decode; assert the two
              token streams are IDENTICAL (the correctness contract: committed
              tokens always come from the target verify -> greedy spec == greedy).
              Reports the mean acceptance rate.
  LATENCY   — prefill ISL tokens, time OSL tokens of spec-decode -> tokens/s/user,
              and (optionally) the plain-greedy-via-verify baseline for the ratio.

Run (from /tmp, using the built ttnn tree):
  cd /tmp && TT_METAL_HOME=/home/ttuser/.local/lib/model-bringup/tt-metal \
    PYTHONPATH=/home/ttuser/dev/tt-metal \
    /home/ttuser/.tenstorrent-venv/bin/python -u -m \
    models.autoports.poolside_laguna_xs_2_1.doc.vllm_integration.scripts.spec_decode_driver \
    --mode accuracy --isl 512 --osl 48 --draft-len 4 \
    --log /home/ttuser/dev/tt-metal/models/autoports/poolside_laguna_xs_2_1/doc/vllm_integration/spec_activity.log
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from types import SimpleNamespace

import torch

import ttnn


class Tee:
    """Stream prints to stdout AND a tailable log file (line-buffered)."""

    def __init__(self, path):
        self.f = open(path, "a", buffering=1) if path else None

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.f:
            self.f.write(line + "\n")
            self.f.flush()


def _greedy_sp(batch, n_active=1):
    """TTSamplingParams-like greedy object (k=1) for `batch` rows (mirrors adapter_repro.sp)."""
    return SimpleNamespace(
        temperature=[0.0] * batch,
        top_k=[0] * batch,
        top_p=[1.0] * batch,
        seed=[None] * batch,
        num_logprobs=None,
        enable_log_probs=torch.zeros(batch, dtype=torch.bool),
    )


def build_prompt(tokenizer, isl, vocab, mode="synthetic"):
    """Build a ~isl-token prompt. mode='synthetic' repeats a small snippet (artificial best-case for
    ngram); mode='code' concatenates REAL source files from the model repo so the model's greedy
    continuation exhibits realistic code copying (the honest accept-rate workload)."""
    if mode == "code" and tokenizer is not None:
        import glob

        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")  # model dir
        files = sorted(glob.glob(os.path.join(root, "tt", "*.py")))
        text = ""
        for f in files:
            try:
                text += open(f).read() + "\n\n"
            except Exception:
                pass
            if len(text) > isl * 8:  # ~heuristic chars→tokens
                break
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        while len(ids) < isl:
            ids = ids + ids
        return ids[:isl]
    snippet = (
        "def compute_metrics(results):\n"
        "    total = 0\n"
        "    for r in results:\n"
        "        total += r.value\n"
        "    return total / len(results)\n\n"
        "# The compute_metrics function iterates over results and returns the mean value.\n"
        "# It is used throughout the pipeline to aggregate per-item scores into a summary.\n"
    )
    if tokenizer is not None:
        ids = tokenizer(snippet, add_special_tokens=False)["input_ids"]
        out = []
        while len(out) < isl:
            out.extend(ids)
        return out[:isl]
    g = torch.Generator().manual_seed(0)
    return torch.randint(0, vocab, (isl,), generator=g, dtype=torch.int64).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["accuracy", "latency", "both", "selfcheck", "decode-bench", "bcheck"], default="accuracy"
    )
    ap.add_argument(
        "--decode-batch",
        type=int,
        default=1,
        help="max_batch_size (bcheck uses B>1 to validate fused ops @ serving batch)",
    )
    ap.add_argument("--isl", type=str, default="512", help="input length (decode-bench accepts comma list)")
    ap.add_argument("--osl", type=int, default=48, help="latency tokens to generate")
    ap.add_argument("--isl-acc", type=int, default=512, help="accuracy-phase prompt length")
    ap.add_argument("--osl-acc", type=int, default=48, help="accuracy-phase tokens")
    ap.add_argument("--draft-len", type=int, default=4, help="ngram K")
    ap.add_argument("--ngram-max-n", type=int, default=3)
    ap.add_argument(
        "--verify-mode",
        choices=["prefill", "decode"],
        default="prefill",
        help="prefill = suffix-prefill verifier; decode = batched-decode verifier (fast, seq KV write)",
    )
    ap.add_argument("--traced", action="store_true", help="decode mode: replay a captured B=K+1 verify trace")
    ap.add_argument(
        "--prompt-mode",
        choices=["synthetic", "code"],
        default="synthetic",
        help="synthetic = repeated snippet (best-case ngram); code = real repo source (realistic accept)",
    )
    ap.add_argument("--max-model-len", type=int, default=None, help="default = next pow2 >= isl+osl+64")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--layers", type=str, default="", help="comma list to reduce layers (default full 40)")
    ap.add_argument("--baseline", action="store_true", help="also time plain-greedy-via-verify in latency mode")
    ap.add_argument("--log", type=str, default="")
    args = ap.parse_args()

    log = Tee(args.log)
    log(f"=== spec_decode_driver mode={args.mode} isl={args.isl} osl={args.osl} K={args.draft_len} ===")

    from transformers import AutoConfig

    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
    )  # model root for tt.spec_decode
    from models.autoports.poolside_laguna_xs_2_1.tt.generator_vllm import LagunaForCausalLM
    from models.autoports.poolside_laguna_xs_2_1.tt.spec_decode import SpeculativeDecoder, plain_greedy_via_verify

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("poolside/Laguna-XS-2.1", trust_remote_code=True)
        stop_ids = set()
        for a in ("eos_token_id",):
            v = getattr(tokenizer, a, None)
            if isinstance(v, int):
                stop_ids.add(v)
            elif isinstance(v, (list, tuple)):
                stop_ids.update(int(x) for x in v)
    except Exception as e:
        log(f"[warn] tokenizer unavailable ({e}); using random-token prompt, no stop tokens")
        tokenizer, stop_ids = None, set()

    B = args.decode_batch
    bs = args.block_size
    isl_list = [int(x) for x in str(args.isl).split(",")]
    isl_primary = isl_list[0]
    _need = max(max(isl_list) + args.osl, args.isl_acc + args.osl_acc) + 64
    mml = args.max_model_len or (1 << _need.bit_length())
    nblocks = math.ceil(mml / bs)
    num_gpu_blocks = B * nblocks + 16
    n_layers = [int(x) for x in args.layers.split(",")] if args.layers else None

    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1_500_000_000)
    try:
        hf = AutoConfig.from_pretrained("poolside/Laguna-XS-2.1", trust_remote_code=True)
        log(f"[phase] build model layers={'full' if n_layers is None else n_layers} max_model_len={mml}")
        model = LagunaForCausalLM.initialize_vllm_model(hf, mesh, max_batch_size=B, max_seq_len=mml, n_layers=n_layers)
        log(f"[phase] allocate_kv_cache blocks={num_gpu_blocks} shape=({num_gpu_blocks},2,{bs},128)")
        nl = len(n_layers) if n_layers is not None else model.model.n_layers if hasattr(model.model, "n_layers") else 40
        kv = model.allocate_kv_cache((num_gpu_blocks, 2, bs, 128), torch.bfloat16, nl)

        # two-phase warmup (exact order from adapter_repro / plugin model_runner).
        # NOTE: in traced spec-verify mode we deliberately SKIP the phase-2 decode-trace CAPTURE. The
        # spec path replays its OWN B=K+1 verify trace and never the normal B=1 decode trace; leaving
        # the B=1 decode trace resident makes TWO CCL-bearing traces coexist, which deadlocks the mesh
        # when the verify trace is captured/replayed (gemma4/tt/spec_decode.py documents this exact
        # hazard). Phase-1 decode warmup (enable_trace=False) still runs to compile decode programs +
        # record max_num_blocks_per_req; only the resident-trace capture is skipped.
        capture_decode_trace = not (args.verify_mode == "decode" and args.traced)
        log("[phase] warmup phase-1 (compile, no trace)")
        model.warmup_model_prefill(kv_cache=kv, enable_trace=False, can_sample_on_device=True)
        model.warmup_model_decode(
            kv_cache=kv, enable_trace=False, max_batch_size=B, num_blocks=nblocks, can_sample_on_device=True
        )
        model.already_warmed_up_prefill = False
        log(f"[phase] warmup phase-2 (prefill buffers; decode-trace capture={capture_decode_trace})")
        model.warmup_model_prefill(kv_cache=kv, enable_trace=True, can_sample_on_device=True)
        if capture_decode_trace:
            model.warmup_model_decode(
                kv_cache=kv, enable_trace=True, max_batch_size=B, num_blocks=nblocks, can_sample_on_device=True
            )
        else:
            # Traced spec-verify: capture the B=K1 verify trace HERE (safe warmup window), not lazily
            # mid-serving (that hangs). K1 = draft_len+1; verify always pads to this fixed batch.
            log(f"[phase] warmup verify-decode trace K1={args.draft_len + 1}")
            model.warmup_verify_decode(args.draft_len, kv, nblocks)
        ttnn.synchronize_device(mesh)
        log("[phase] warmup done")

        user0_blocks = list(range(0, nblocks))
        pt = torch.tensor([user0_blocks], dtype=torch.int32)
        spec = SpeculativeDecoder(
            model,
            kv_cache=kv,
            page_table=pt,
            stop_tokens=stop_ids,
            draft_len=args.draft_len,
            ngram_max_n=args.ngram_max_n,
            verify_mode=args.verify_mode,
            traced=args.traced,
        )
        log(f"[cfg] verify_mode={args.verify_mode} traced={args.traced}")

        def prefill_prompt(prompt):
            model.prefill_forward(
                torch.tensor(prompt, dtype=torch.int64).reshape(1, len(prompt)),
                page_table=pt,
                kv_cache=kv,
                prompt_lens=[len(prompt)],
                start_pos=[0],
                sampling_params=_greedy_sp(1),
            )

        def run_selfcheck(isl):
            # Localize the traced-verify divergence: run the SAME window through eager (known-correct)
            # and traced verify, per-row. drafts = the true greedy chain, so eager g should equal drafts.
            prompt = build_prompt(tokenizer, isl, model.vocab, mode=args.prompt_mode)
            log(f"[phase] SELFCHECK: prefill P={len(prompt)}")
            prefill_prompt(prompt)
            anchor = int(prompt[-1])
            ap = len(prompt) - 1
            hist = list(int(t) for t in prompt)
            drafts = []
            for _ in range(args.draft_len):
                gg = model.verify_greedy_decode([hist[-1]], [len(hist) - 1], page_table=pt, kv_cache=kv, traced=False)
                tk = int(gg.reshape(-1)[0])
                drafts.append(tk)
                hist.append(tk)
            window = [anchor] + drafts
            positions = [ap + j for j in range(len(window))]
            log(f"[selfcheck] greedy drafts={drafts} window={window} positions={positions}")
            prefill_prompt(prompt)
            ge = torch.argmax(model.verify_forward_decode(window, positions, page_table=pt, kv_cache=kv), dim=-1)
            ge = ge.reshape(-1).tolist()
            # DISCRIMINATOR: eager decode-verify through the on-device SAMPLER (no trace). If this ==
            # g_eager (host argmax), the sampler is fine and the traced bug is trace-only; if it diverges,
            # the sampler is the culprit (model-layer fixable).
            prefill_prompt(prompt)
            ges = model.verify_sampler_eager(window, positions, page_table=pt, kv_cache=kv).reshape(-1).tolist()
            log(
                f"[selfcheck] g_eager_sampler(no trace)={ges}  row0 {'==host-argmax' if ges[0]==ge[0] else 'DIVERGES → SAMPLER BUG'}"
            )
            prefill_prompt(prompt)
            gt = (
                model.verify_greedy_decode(window, positions, page_table=pt, kv_cache=kv, traced=True)
                .reshape(-1)
                .tolist()
            )
            # Staleness probe: SAME positions, DIFFERENT row-0 (anchor) token. If g_traced[0] is
            # unchanged, row 0 is returning a baked/stale value independent of its refreshed input.
            prefill_prompt(prompt)
            window2 = [int((anchor + 12345) % model.vocab)] + drafts
            gt2 = (
                model.verify_greedy_decode(window2, positions, page_table=pt, kv_cache=kv, traced=True)
                .reshape(-1)
                .tolist()
            )
            log(f"[selfcheck] g_eager  ={ge}")
            log(f"[selfcheck] g_traced ={gt}")
            log(
                f"[selfcheck] g_traced2(alt anchor {window2[0]})={gt2}  row0 {'STALE(unchanged)' if gt2[0]==gt[0] else 'changed'}"
            )
            log(f"[selfcheck] expect g[i]==drafts[i]: drafts={drafts}")
            log(
                f"[RESULT] SELFCHECK eager-vs-traced {'MATCH' if ge == gt else 'MISMATCH'}; "
                f"eager-vs-drafts {'MATCH' if ge[:len(drafts)] == drafts else 'MISMATCH'}"
            )

        def run_accuracy(isl, osl):
            prompt = build_prompt(tokenizer, isl, model.vocab, mode=args.prompt_mode)
            log(f"[phase] ACCURACY: prefill P={len(prompt)}, plain-greedy baseline ({osl} tok)")
            prefill_prompt(prompt)
            ref, ref_s = plain_greedy_via_verify(
                model, prompt, osl, kv_cache=kv, page_table=pt, stop_tokens=stop_ids, verify_mode=args.verify_mode
            )
            log(f"[ok] baseline {len(ref)} toks in {ref_s:.1f}s ; first10={ref[:10]}")
            log("[phase] RE-PREFILL then ngram spec-decode")
            prefill_prompt(prompt)
            got, accepts = spec.generate(prompt, osl)
            n = min(len(ref), len(got))
            mism = [i for i in range(n) if ref[i] != got[i]]
            acc_mean = sum(accepts) / max(1, len(accepts))
            log(f"[ok] spec {len(got)} toks ; iters={len(accepts)} mean_accept={acc_mean:.2f}/{args.draft_len}")
            log(f"[result] first_mismatch={mism[0] if mism else 'NONE'} total_mismatch={len(mism)}/{n}")
            if not mism and len(ref) == len(got):
                log("[RESULT] ACCURACY PASS — spec-decode is token-identical to plain greedy")
            else:
                log(f"[RESULT] ACCURACY DIVERGENCE at {mism[:5]}")
                for i in mism[:5]:
                    log(f"    idx {i}: ref={ref[i]} got={got[i]}")
            if tokenizer is not None:
                log(f"[text] greedy: {tokenizer.decode(ref)!r}")
                log(f"[text] spec  : {tokenizer.decode(got)!r}")

        def run_latency(isl, osl):
            prompt = build_prompt(tokenizer, isl, model.vocab, mode=args.prompt_mode)
            log(f"[phase] LATENCY {isl}/{osl}: prefill P={len(prompt)}")
            t_pf = time.perf_counter()
            prefill_prompt(prompt)
            ttnn.synchronize_device(mesh)
            ttft = time.perf_counter() - t_pf
            log(f"[ok] prefill/TTFT {ttft*1000:.0f}ms ; spec-decode K={args.draft_len} OSL={osl}")
            t0 = time.perf_counter()
            _p0 = [time.perf_counter()]

            def _prog(n, total, acc):
                now = time.perf_counter()
                dt = now - _p0[0]
                _p0[0] = now
                log(
                    f"[gen] {n}/{total} toks  mean_accept={acc:.2f}/{args.draft_len}  "
                    f"{(64/dt if dt > 0 else 0):.1f} tok/s (running)"
                )

            got, accepts = spec.generate(prompt, osl, on_progress=_prog, progress_every=64)
            spec_s = time.perf_counter() - t0
            acc_mean = sum(accepts) / max(1, len(accepts))
            spec_tps = len(got) / spec_s if spec_s > 0 else 0.0
            log(
                f"[RESULT-SPEC {isl}/{osl}] gen={len(got)} in {spec_s:.2f}s -> {spec_tps:.2f} tok/s/u ; "
                f"iters={len(accepts)} mean_accept={acc_mean:.2f}/{args.draft_len} "
                f"ms/tok={spec_s/max(1,len(got))*1000:.1f} TTFT={ttft*1000:.0f}ms"
            )
            if args.baseline:
                prefill_prompt(prompt)
                base, base_s = plain_greedy_via_verify(
                    model, prompt, osl, kv_cache=kv, page_table=pt, stop_tokens=stop_ids
                )
                base_tps = len(base) / base_s if base_s > 0 else 0.0
                log(
                    f"[RESULT-BASE {isl}/{osl}] via-verify {base_tps:.2f} tok/s/u ; spec speedup={spec_tps/max(1e-9,base_tps):.2f}x"
                )

        def run_decode_bench(isl, steps):
            # Measure the REAL serving decode path (decode_forward, B=1 greedy, traced) at context=isl.
            # Used to A/B TT_LAGUNA_DECODE_SDPA_PC (k_chunk=128) vs the default k_chunk=32 decode SDPA.
            prompt = build_prompt(tokenizer, isl, model.vocab, mode=args.prompt_mode)
            P = len(prompt)
            log(
                f"[phase] DECODE-BENCH {isl}: prefill P={P}, {steps} decode steps (B=1 greedy, "
                f"sdpa_pc={os.environ.get('TT_LAGUNA_DECODE_SDPA_PC','0')})"
            )
            t_pf = time.perf_counter()
            first = model.prefill_forward(
                torch.tensor(prompt, dtype=torch.int64).reshape(1, P),
                page_table=pt,
                kv_cache=kv,
                prompt_lens=[P],
                start_pos=[0],
                sampling_params=_greedy_sp(1),
            )
            ttnn.synchronize_device(mesh)
            log(f"[ok] prefill/TTFT {(time.perf_counter()-t_pf)*1000:.0f}ms")
            tok0 = int((first[0] if isinstance(first, tuple) else first).reshape(-1)[0])
            cur = torch.tensor([[tok0]], dtype=torch.int64)
            posv = P
            for w in range(3):  # warm the decode trace at this batch/pt
                out = model.decode_forward(
                    cur,
                    torch.tensor([posv], dtype=torch.int32),
                    page_table=pt,
                    kv_cache=kv,
                    sampling_params=_greedy_sp(1),
                    reset_batch=(w == 0),
                )
                cur[0, 0] = int(torch.as_tensor(out).reshape(-1)[0])
                posv += 1
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            for _ in range(steps):
                out = model.decode_forward(
                    cur,
                    torch.tensor([posv], dtype=torch.int32),
                    page_table=pt,
                    kv_cache=kv,
                    sampling_params=_greedy_sp(1),
                    reset_batch=False,
                )
                cur[0, 0] = int(torch.as_tensor(out).reshape(-1)[0])
                posv += 1
            ttnn.synchronize_device(mesh)
            dt = time.perf_counter() - t0
            gentoks = []
            cur[0, 0] = tok0
            posc = P
            for _ in range(24):  # correctness: log the greedy token sequence (diff across sdpa_pc for parity)
                out = model.decode_forward(
                    cur,
                    torch.tensor([posc], dtype=torch.int32),
                    page_table=pt,
                    kv_cache=kv,
                    sampling_params=_greedy_sp(1),
                    reset_batch=(len(gentoks) == 0),
                )
                t = int(torch.as_tensor(out).reshape(-1)[0])
                gentoks.append(t)
                cur[0, 0] = t
                posc += 1
            log(
                f"[RESULT-DECODE {isl}] {steps} steps in {dt:.2f}s -> {steps/dt:.2f} tok/s/u  "
                f"ms/tok={dt/steps*1000:.1f}  sdpa_pc={os.environ.get('TT_LAGUNA_DECODE_SDPA_PC','0')}"
            )
            log(f"[DECODE-TOKENS {isl} sdpa_pc={os.environ.get('TT_LAGUNA_DECODE_SDPA_PC','0')}] {gentoks}")

        def run_bcheck(isl):
            # Validate fused_rope/reduce at SERVING batch B: prefill row 0, run a B-row decode (rows 1..B-1
            # padded, pos=-1), log the row-0 greedy token sequence. Diff across configs (fused OFF vs ON):
            # bit-identical => the fused ops are correct at B (they shard across B cores) → safe to default ON.
            prompt = build_prompt(tokenizer, isl, model.vocab, mode=args.prompt_mode)
            P = len(prompt)
            r, rr = os.environ.get("TT_LAGUNA_FUSED_ROPE", "0"), os.environ.get("TT_LAGUNA_FUSED_REDUCE", "0")
            log(
                f"[phase] BCHECK B={B} isl={isl}: prefill P={P}, B-row decode; fused_rope={r} fused_reduce={rr} "
                f"sdpa_pc={os.environ.get('TT_LAGUNA_DECODE_SDPA_PC','1')}"
            )
            first = model.prefill_forward(
                torch.tensor(prompt, dtype=torch.int64).reshape(1, P),
                page_table=pt,
                kv_cache=kv,
                prompt_lens=[P],
                start_pos=[0],
                sampling_params=_greedy_sp(1),
            )
            tok0 = int((first[0] if isinstance(first, tuple) else first).reshape(-1)[0])
            nb = pt.shape[1]
            pt_b = torch.zeros((B, nb), dtype=torch.int32)
            pt_b[0] = torch.as_tensor(pt[0], dtype=torch.int32)
            cur = torch.zeros((B, 1), dtype=torch.int64)
            cur[0, 0] = tok0
            pos = torch.full((B,), -1, dtype=torch.int32)
            gentoks = []
            p0 = P
            for i in range(24):
                pos[0] = p0
                out = model.decode_forward(
                    cur, pos, page_table=pt_b, kv_cache=kv, sampling_params=_greedy_sp(B), reset_batch=(i == 0)
                )
                t = int(torch.as_tensor(out).reshape(-1)[0])
                gentoks.append(t)
                cur[0, 0] = t
                p0 += 1
            log(f"[BCHECK-TOKENS B={B} isl={isl} rope={r} reduce={rr}] {gentoks}")

        if args.mode == "selfcheck":
            run_selfcheck(args.isl_acc)
        if args.mode == "bcheck":
            run_bcheck(isl_primary)
        if args.mode == "decode-bench":
            for _isl in isl_list:
                run_decode_bench(_isl, args.osl)
        if args.mode in ("accuracy", "both"):
            run_accuracy(args.isl_acc, args.osl_acc)
        if args.mode in ("latency", "both"):
            run_latency(isl_primary, args.osl)

        log("[phase] done")
    finally:
        try:
            model.gen.teardown()
        except Exception:
            pass
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
