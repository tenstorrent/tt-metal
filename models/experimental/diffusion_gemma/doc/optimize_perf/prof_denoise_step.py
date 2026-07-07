# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reduced-layer traced profiling of the DiffusionGemma denoise step / prefill / commit.

Builds a reduced-layer (one-per-kind by default) real-checkpoint DiffusionGemma on
the (1,4) QB2 mesh, prefills a short prompt, builds the real denoise logits adapter,
then measures a warmed traced denoise step (adapter forward + terminal decision path)
with signposts. Per-layer device time projects the full 30-layer per-block cost.

Usage (e2e device timing):
    DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
    python -u models/experimental/diffusion_gemma/doc/optimize_perf/prof_denoise_step.py \
        --num-layers 2 --canvas-length 256 --iters 3

Under tracy for the op table (signposts DENOISE_START..DENOISE_END):
    python -m tracy -r -p -v -m \
        models/experimental/diffusion_gemma/doc/optimize_perf/prof_denoise_step.py --num-layers 2 --iters 3
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger
from tracy import signpost

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.tt import denoise_loop as DL
from models.experimental.diffusion_gemma.tt.denoise_forward import (
    make_generation_logits_fn_builder_from_checkpoint_state,
)
from models.experimental.diffusion_gemma.tt.generate import (
    host_canvas_to_device,
    prefill_prompt_tokens,
    tokenize_prompt,
)

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")
TEMP = 0.6
BUDGET = 0.1


def _log_dram(mesh, label):
    try:
        views = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
        logger.info(f"[{label}] dram views captured")
    except Exception:
        pass


def run(num_layers, canvas_length, iters, prompt, max_seq_len, do_trace, commit_tokens=None):
    # TP=4 attention all-reduce needs the 1D fabric initialized (as the text demo does).
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    # --no-trace (eager op-table profiling) does not use the trace region; shrink it to free
    # DRAM for full-model weights + the on-device profiler op buffer (avoids OOM at 30 layers).
    trace_region_size = 1300000000 if do_trace else 20000000
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_region_size)
    try:
        model_inputs = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = model_inputs.tt_model
        logger.info(f"built reduced model num_layers={num_layers}")

        prompt_tokens = tokenize_prompt(model_inputs.tokenizer, prompt)
        # prefill TTFT
        signpost("PREFILL_START")
        t0 = time.perf_counter()
        prefill = prefill_prompt_tokens(tt_model, prompt_tokens)
        ttnn.synchronize_device(mesh)
        ttft_ms = (time.perf_counter() - t0) * 1e3
        signpost("PREFILL_END")
        logger.info(f"[prefill] prompt_len={prefill.prompt_len} cache_len={prefill.cache_len} ttft_ms={ttft_ms:.2f}")

        adapter_kwargs = {}
        cfg = getattr(tt_model, "hf_config", None)
        if cfg is not None:
            adapter_kwargs["config"] = cfg
        logits_builder = make_generation_logits_fn_builder_from_checkpoint_state(
            model_inputs.state_dict, **adapter_kwargs
        )
        adapter = logits_builder(tt_model, prompt_tokens=prompt_tokens, prompt_len=prefill.cache_len)

        vocab = int(getattr(model_inputs.tokenizer, "vocab_size", 262144))
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        host_canvas = torch.randint(0, vocab, (1, canvas_length), dtype=torch.long, generator=gen)
        canvas = host_canvas_to_device(mesh, host_canvas)
        noise_tokens = host_canvas_to_device(mesh, torch.randint(0, vocab, (1, canvas_length), dtype=torch.long))
        # preallocate trace-safe accept/renoise constants outside any trace
        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=BUDGET)

        def one_step():
            logits = adapter(canvas, step=0)
            res = DL.denoise_step(
                logits,
                temperature=TEMP,
                entropy_budget=BUDGET,
                gumbel_noise=None,
                noise_tokens=noise_tokens,
                constants=consts,
            )
            res.canvas.deallocate(True)
            res.accept_mask.deallocate(True)
            res.entropy.deallocate(True)
            res.sampled.deallocate(True)
            res.argmax.deallocate(True)
            adapter.reset()

        # eager warm + timing (always works, per-layer projectable)
        t0 = time.perf_counter()
        one_step()
        ttnn.synchronize_device(mesh)
        eager_ms = (time.perf_counter() - t0) * 1e3
        logger.info(f"[denoise] EAGER one-step (warm compile) ms={eager_ms:.2f}")

        # measured eager loop
        signpost("DENOISE_START")
        t0 = time.perf_counter()
        for _ in range(iters):
            one_step()
        ttnn.synchronize_device(mesh)
        eager_step_ms = (time.perf_counter() - t0) * 1e3 / iters
        signpost("DENOISE_END")
        logger.info(f"[denoise] EAGER warmed ms_per_step={eager_step_ms:.3f} (num_layers={num_layers})")
        print(f"RESULT num_layers={num_layers} ttft_ms={ttft_ms:.2f} eager_ms_per_step={eager_step_ms:.3f}", flush=True)

        # commit: sequential single-token decode-appends into the KV cache.
        # commit_tokens lets a reduced-window profile capture only N of the
        # canvas_length commit tokens (each is an independent single-token
        # decode-append, so per-token cost is canvas-independent) to fit the
        # on-device profiler buffer; scale the measured per-token cost x256.
        n_commit = min(commit_tokens, canvas_length) if commit_tokens else canvas_length
        try:
            from models.experimental.diffusion_gemma.tt.generate import commit_canvas_tokens

            host_commit = torch.randint(0, vocab, (1, n_commit), dtype=torch.long)
            # warm
            commit_canvas_tokens(tt_model, host_commit, start_pos=prefill.cache_len)
            ttnn.synchronize_device(mesh)
            signpost("COMMIT_START")
            t0 = time.perf_counter()
            commit_canvas_tokens(tt_model, host_commit, start_pos=prefill.cache_len)
            ttnn.synchronize_device(mesh)
            commit_ms = (time.perf_counter() - t0) * 1e3
            signpost("COMMIT_END")
            logger.info(
                f"[commit] EAGER {n_commit}-token commit ms={commit_ms:.2f} "
                f"per_token_ms={commit_ms/max(n_commit,1):.3f} (num_layers={num_layers})"
            )
            print(
                f"RESULT_COMMIT num_layers={num_layers} n_commit={n_commit} "
                f"commit_ms={commit_ms:.2f} per_token_ms={commit_ms/max(n_commit,1):.4f}",
                flush=True,
            )
        except Exception as e:
            logger.warning(f"commit measurement failed: {type(e).__name__}: {str(e)[:200]}")
            print(f"COMMIT_BLOCKED {type(e).__name__}: {str(e)[:150]}", flush=True)

        if do_trace:
            try:
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                for _ in range(iters):
                    one_step()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.synchronize_device(mesh)
                ttnn.execute_trace(mesh, tid, blocking=False)
                ttnn.synchronize_device(mesh)
                signpost("DENOISE_TRACE_START")
                t0 = time.perf_counter()
                ttnn.execute_trace(mesh, tid, blocking=False)
                ttnn.synchronize_device(mesh)
                traced_ms = (time.perf_counter() - t0) * 1e3 / iters
                signpost("DENOISE_TRACE_END")
                ttnn.release_trace(mesh, tid)
                logger.info(f"[denoise] TRACED ms_per_step={traced_ms:.3f}")
                print(f"RESULT_TRACED num_layers={num_layers} traced_ms_per_step={traced_ms:.3f}", flush=True)
            except Exception as e:
                logger.warning(f"trace capture failed: {type(e).__name__}: {str(e)[:300]}")
                print(f"TRACE_BLOCKED {type(e).__name__}: {str(e)[:200]}", flush=True)

        canvas.deallocate(True)
        noise_tokens.deallocate(True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-seq-len", type=int, default=512)
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument(
        "--commit-tokens",
        type=int,
        default=None,
        help="Profile only N commit tokens (default: canvas_length). Use a small N to keep "
        "the commit phase inside the on-device profiler buffer; per-token cost is scaled x256.",
    )
    args = ap.parse_args()
    run(
        args.num_layers,
        args.canvas_length,
        args.iters,
        args.prompt,
        args.max_seq_len,
        do_trace=not args.no_trace,
        commit_tokens=args.commit_tokens,
    )


if __name__ == "__main__":
    main()
