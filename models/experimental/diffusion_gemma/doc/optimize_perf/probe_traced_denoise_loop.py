# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Feasibility + measurement probe for a TRACED serving denoise loop (#47465, 100-path win B).

The eager serving denoise loop pays ~137 ms/step eager-dispatch overhead over the
traced floor (`prof_denoise_step`: 30L eager-device 598 ms vs traced 461 ms) plus the
per-step host readbacks removed by `device_loop_denoise_block`. Capturing the whole
device-only fixed-step loop as one Metal trace removes the dispatch overhead.

`verify_trace_safe_loop.py` already proved `run_fixed_denoise_steps` traces with device
canvas feedback + pre-uploaded noise, but with a SYNTHETIC logits fn. This probe answers
the remaining question: **does the REAL DenoiseLogitsAdapter** (backbone forward + frozen
prompt-KV read + self-conditioning state) **trace correctly inside the multi-step loop**,
and what is the traced-vs-eager per-step win?

It:
  1. builds a reduced-layer real-checkpoint model + prefills (like prof_denoise_step);
  2. builds the real generation logits adapter (argmax mode: gumbel=None);
  3. pre-uploads per-step renoise tokens to persistent device buffers (from_torch is a
     WRITE, forbidden in trace — mirrors verify_trace_safe_loop);
  4. runs run_fixed_denoise_steps EAGER (baseline committed argmax + eager ms/step);
  5. captures + replays the SAME loop as a Metal trace (traced ms/step);
  6. asserts the traced committed argmax matches the eager one, and prints the win.

*** DEVICE-OWNERSHIP: run only when QB2 is free. ***
"""
from __future__ import annotations

import argparse
import os
import time

import torch
from loguru import logger

import ttnn
from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
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


def _repl(mesh):
    return ttnn.ReplicateTensorToMesh(mesh) if mesh.get_num_devices() > 1 else None


def _committed_ids(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).squeeze().long()


def run(num_layers, canvas_length, steps, prompt, max_seq_len):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1300000000)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = mi.tt_model
        prompt_tokens = tokenize_prompt(mi.tokenizer, prompt)
        prefill = prefill_prompt_tokens(tt_model, prompt_tokens)
        ttnn.synchronize_device(mesh)
        logger.info(f"prefilled cache_len={prefill.cache_len}")

        adapter_kwargs = {}
        cfg_hf = getattr(tt_model, "hf_config", None)
        if cfg_hf is not None:
            adapter_kwargs["config"] = cfg_hf
        builder = make_generation_logits_fn_builder_from_checkpoint_state(mi.state_dict, **adapter_kwargs)
        adapter = builder(tt_model, prompt_tokens=prompt_tokens, prompt_len=prefill.cache_len)

        # Diagnostic: null the self-conditioning (the adapter's only cross-step mutable
        # state) to test whether it is the trace-replay divergence. With it off the loop
        # is stateless like verify_trace_safe_loop's synthetic fn → expect ~100% match.
        if os.environ.get("DG_PROBE_NO_SELFCOND", "0") == "1":
            adapter.self_conditioning = None
            adapter.self_conditioning_embedding_weight = None
            logger.info("[probe] self-conditioning DISABLED (stateless loop)")

        cfg = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
        vocab = int(getattr(mi.tokenizer, "vocab_size", 262144))

        # pre-upload per-step renoise tokens to persistent device buffers (trace-safe)
        gen = torch.Generator().manual_seed(1)
        noise_list = []
        for _ in range(steps):
            nt = torch.randint(0, vocab, (1, canvas_length), dtype=torch.long, generator=gen)
            noise_list.append(host_canvas_to_device(mesh, nt))

        def noise_tokens_fn(step):
            return ttnn.clone(noise_list[step])

        init_host = torch.randint(
            0, vocab, (1, canvas_length), dtype=torch.long, generator=torch.Generator().manual_seed(7)
        )

        def make_init():
            return host_canvas_to_device(mesh, init_host)

        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=cfg.entropy_budget)

        # ---- EAGER baseline ----
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        committed_eager = DL.run_fixed_denoise_steps(
            adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_tokens_fn, constants=consts
        )
        ttnn.synchronize_device(mesh)
        eager_ms = (time.perf_counter() - t0) * 1e3
        eager_ids = _committed_ids(committed_eager)
        committed_eager.deallocate(True)
        logger.info(
            f"[eager] block_ms={eager_ms:.1f} ms/step={eager_ms/steps:.1f} committed[:8]={eager_ids[:8].tolist()}"
        )

        # ---- TRACE capture + replay ----
        traced_ms = None
        traced_ids = None
        try:
            # warm program cache
            warm = DL.run_fixed_denoise_steps(
                adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_tokens_fn, constants=consts
            )
            ttnn.synchronize_device(mesh)
            warm.deallocate(True)

            init_dev = make_init()
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            committed_traced = DL.run_fixed_denoise_steps(
                adapter, init_dev, cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_tokens_fn, constants=consts
            )
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.synchronize_device(mesh)
            # replay once (warm), then time
            ttnn.execute_trace(mesh, tid, blocking=False)
            ttnn.synchronize_device(mesh)
            t0 = time.perf_counter()
            ttnn.execute_trace(mesh, tid, blocking=False)
            ttnn.synchronize_device(mesh)
            traced_ms = (time.perf_counter() - t0) * 1e3
            traced_ids = _committed_ids(committed_traced)
            ttnn.release_trace(mesh, tid)
            logger.info(
                f"[traced] block_ms={traced_ms:.1f} ms/step={traced_ms/steps:.1f} committed[:8]={traced_ids[:8].tolist()}"
            )
        except Exception as e:
            logger.error(f"TRACE_CAPTURE_FAILED {type(e).__name__}: {str(e)[:400]}")
            print(f"RESULT_TRACE_BLOCKED layers={num_layers} {type(e).__name__}: {str(e)[:200]}", flush=True)

        print(
            f"RESULT_EAGER layers={num_layers} steps={steps} block_ms={eager_ms:.1f} ms_per_step={eager_ms/steps:.2f}",
            flush=True,
        )
        if traced_ms is not None:
            match = (eager_ids == traced_ids).float().mean().item()
            print(
                f"RESULT_TRACED layers={num_layers} steps={steps} block_ms={traced_ms:.1f} "
                f"ms_per_step={traced_ms/steps:.2f} win={eager_ms/traced_ms:.2f}x committed_match={match*100:.1f}%",
                flush=True,
            )
            print("TRACED_LOOP_OK" if match > 0.999 else "TRACED_LOOP_MISMATCH", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.steps, args.prompt, args.max_seq_len)


if __name__ == "__main__":
    main()
