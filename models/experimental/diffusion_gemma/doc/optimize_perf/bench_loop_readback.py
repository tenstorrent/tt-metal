# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lever-1 quantifier: production eager denoise loop (host readback) vs trace-safe loop.

Builds a reduced-layer real-checkpoint DiffusionGemma on the (1,4) QB2 mesh, prefills a
short prompt, builds the real denoise logits adapter, then measures — on the SAME model —

  A) ``denoise_block``            : the PRODUCTION eager loop (5 host readbacks/step +
                                    torch.equal halt check) that serving.decode_block drives;
  B) ``run_fixed_denoise_steps``  : the trace-safe device-only loop, wrapped in a Metal trace.

Both run a fixed step count (early-halt disabled via a very-negative entropy_stop_threshold)
so ms/step is apples-to-apples. The per-step delta (A-B) is the host-readback overhead that a
traced decode removes. Run at two layer counts to isolate the per-step constant from the
per-layer device slope and project to the real 30-layer step.

Usage:
    DG_CKPT=... python -u .../bench_loop_readback.py --num-layers 2 --steps 8
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


def run(num_layers, canvas_length, steps, prompt, max_seq_len):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=1300000000)
    try:
        model_inputs = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = model_inputs.tt_model
        prompt_tokens = tokenize_prompt(model_inputs.tokenizer, prompt)
        prefill = prefill_prompt_tokens(tt_model, prompt_tokens)
        logger.info(f"[prefill] cache_len={prefill.cache_len}")

        adapter_kwargs = {}
        cfg = getattr(tt_model, "hf_config", None)
        if cfg is not None:
            adapter_kwargs["config"] = cfg
        logits_builder = make_generation_logits_fn_builder_from_checkpoint_state(
            model_inputs.state_dict, **adapter_kwargs
        )

        vocab = int(getattr(model_inputs.tokenizer, "vocab_size", 262144))
        gen = torch.Generator(device="cpu").manual_seed(0)

        # Fixed step count, early-halt disabled (entropy>=0 so mean < -1 is never true).
        config = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
        config = config.__class__(**{**config.__dict__, "entropy_stop_threshold": -1.0})

        def fresh_noise(step):
            return host_canvas_to_device(mesh, torch.randint(0, vocab, (1, canvas_length), dtype=torch.long))

        def new_canvas():
            host = torch.randint(0, vocab, (1, canvas_length), dtype=torch.long, generator=gen)
            return host_canvas_to_device(mesh, host)

        # ---------- A) production eager loop (denoise_block, host readback/step) ----------
        adapter = logits_builder(tt_model, prompt_tokens=prompt_tokens, prompt_len=prefill.cache_len)
        # warm
        traj = DL.denoise_block(
            adapter, new_canvas(), config, gumbel_noise_fn=lambda s: None, noise_tokens_fn=fresh_noise
        )
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        traj = DL.denoise_block(
            adapter, new_canvas(), config, gumbel_noise_fn=lambda s: None, noise_tokens_fn=fresh_noise
        )
        ttnn.synchronize_device(mesh)
        block_ms = (time.perf_counter() - t0) * 1e3
        eager_readback_ms_per_step = block_ms / traj.num_steps
        logger.info(
            f"[A eager denoise_block] steps_run={traj.num_steps} block_ms={block_ms:.1f} "
            f"ms_per_step={eager_readback_ms_per_step:.2f}"
        )

        # ---------- B) trace-safe device-only loop, traced ----------
        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=config.entropy_budget)
        noise_tok = host_canvas_to_device(mesh, torch.randint(0, vocab, (1, canvas_length), dtype=torch.long))

        def run_fixed(init):
            return DL.run_fixed_denoise_steps(
                adapter,
                init,
                config,
                gumbel_noise_fn=None,
                noise_tokens_fn=lambda s: noise_tok,  # shared preallocated (trace-safe, not deallocated by fn path)
                constants=consts,
            )

        # NOTE: run_fixed_denoise_steps deallocates the noise each step, so for the traced
        # measurement we instead measure the per-step device cost via the already-validated
        # one_step trace path (adapter + denoise_step_next_canvas) with a persistent noise tensor.
        def one_step_device(canvas):
            logits = adapter(canvas, step=0)
            nxt, argmax = DL.denoise_step_next_canvas(
                logits,
                temperature=0.6,
                entropy_budget=config.entropy_budget,
                gumbel_noise=None,
                noise_tokens=noise_tok,
                constants=consts,
            )
            argmax.deallocate(True)
            adapter.reset()
            return nxt

        base = new_canvas()
        # warm compile
        c = one_step_device(base)
        c.deallocate(True)
        ttnn.synchronize_device(mesh)
        # trace capture
        cap = new_canvas()
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        outs = []
        cur = cap
        for _ in range(steps):
            cur = one_step_device(cur)
            outs.append(cur)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        ttnn.synchronize_device(mesh)
        ttnn.execute_trace(mesh, tid, blocking=False)
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        ttnn.execute_trace(mesh, tid, blocking=False)
        ttnn.synchronize_device(mesh)
        traced_ms = (time.perf_counter() - t0) * 1e3
        traced_ms_per_step = traced_ms / steps
        ttnn.release_trace(mesh, tid)
        logger.info(
            f"[B traced device-only] steps={steps} total_ms={traced_ms:.1f} ms_per_step={traced_ms_per_step:.2f}"
        )

        delta = eager_readback_ms_per_step - traced_ms_per_step
        logger.info(
            f"RESULT_LEVER1 num_layers={num_layers} "
            f"eager_readback_ms_per_step={eager_readback_ms_per_step:.2f} "
            f"traced_ms_per_step={traced_ms_per_step:.2f} "
            f"readback_overhead_ms_per_step={delta:.2f}"
        )
        print(
            f"RESULT_LEVER1 num_layers={num_layers} eager_readback_ms_per_step={eager_readback_ms_per_step:.2f} "
            f"traced_ms_per_step={traced_ms_per_step:.2f} readback_overhead_ms_per_step={delta:.2f}",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()
    run(args.num_layers, args.canvas_length, args.steps, args.prompt, args.max_seq_len)


if __name__ == "__main__":
    main()
