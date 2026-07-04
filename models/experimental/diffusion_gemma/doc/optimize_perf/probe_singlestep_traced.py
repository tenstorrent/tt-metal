# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Single-step traced denoise loop — mechanism validation (#47465, path to 30).

The whole-loop denoise trace RACES on the self-conditioning cross-step feedback:
capturing all N steps as one trace and replaying it gives traced-vs-traced 60.5%
(``probe_traced_denoise_loop.py``) — two replays of the same trace disagree, i.e.
a genuine race, NOT non-determinism (eager-vs-eager is 100%) and NOT a buffer-
management bug (in-place, fresh-tensor, and prev_logits-chain self-cond all give
the identical 60.5%). Crucially a SINGLE self-cond step traces at 100% (STEPS=1),
so the race is purely CROSS-STEP inside one trace.

This probe validates the fix: capture ONE denoise step as a Metal trace and replay
it once per step, carrying the cross-step state (canvas + self-cond signal) in
persistent device buffers updated in-place across replays — the KV-cache pattern a
traced decode already uses. N traces are captured (one per step index) so each
bakes its own temperature ``T[i]`` and reads its own renoise tokens ``noise[i]``;
all N read/write the SAME persistent ``canvas_buf`` / ``signal_buf`` / ``committed_buf``.

It proves, on device:
  A. the trace-safe self-cond adapter refactor (uniform ``forward`` over a persistent
     zeroed-for-step-0 signal buffer) is BIT-EXACT to the eager reference (original
     ``condition(None)`` + prev_logits chain), eager-vs-eager.
  B. the single-step traced loop committed argmax matches the eager reference
     (self-cond ON), and is self-consistent across two block replays.

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


def _committed_ids(t):
    return ttnn.to_torch(ttnn.get_device_tensors(t)[0]).squeeze().long()


def _capture_step_traces(mesh, adapter, cfg, canvas_buf, committed_buf, noise_list, consts):
    """Capture one Metal trace per step index; each threads the persistent buffers."""
    traces = []
    for step in range(cfg.max_denoise_steps):
        temperature = DL.temperature_at_step(step, cfg.max_denoise_steps, cfg.temperature_start, cfg.temperature_end)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        logits = adapter(canvas_buf, step)  # reads canvas_buf + signal_buf; writes signal_buf in-place
        next_canvas, argmax = DL.denoise_step_next_canvas(
            logits,
            temperature=temperature,
            entropy_budget=cfg.entropy_budget,
            gumbel_noise=None,
            noise_tokens=noise_list[step],
            constants=consts,
        )
        DL._deallocate_logits_if_unowned(adapter, logits)
        ttnn.copy(next_canvas, canvas_buf)  # thread the accepted canvas in-place
        ttnn.copy(argmax, committed_buf)  # emit the clean-argmax commit candidate in-place
        next_canvas.deallocate(True)
        argmax.deallocate(True)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        traces.append(tid)
    return traces


def _replay_block(mesh, adapter, traces, canvas_buf, committed_buf, init_dev):
    """One block: reset persistent state, then replay every step trace in order."""
    ttnn.copy(init_dev, canvas_buf)
    adapter.reset_signal_buffer()  # zero signal_buf so step 0 == post_norm(embed)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for tid in traces:
        ttnn.execute_trace(mesh, tid, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1e3
    return _committed_ids(committed_buf), ms


def run(num_layers, canvas_length, steps, prompt, max_seq_len):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=2000000000)
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

        cfg = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
        vocab = int(getattr(mi.tokenizer, "vocab_size", 262144))
        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=cfg.entropy_budget)

        # per-step renoise tokens (persistent, deterministic) + fixed init canvas
        gen = torch.Generator().manual_seed(1)
        noise_list = [
            host_canvas_to_device(mesh, torch.randint(0, vocab, (1, canvas_length), dtype=torch.long, generator=gen))
            for _ in range(steps)
        ]
        init_host = torch.randint(
            0, vocab, (1, canvas_length), dtype=torch.long, generator=torch.Generator().manual_seed(7)
        )

        def make_init():
            return host_canvas_to_device(mesh, init_host)

        def noise_fn(step):
            return ttnn.clone(noise_list[step])

        # ---- (ref) ORIGINAL eager (prev_logits chain, condition(None) step 0) ----
        committed_ref = DL.run_fixed_denoise_steps(
            adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_fn, constants=consts
        )
        ttnn.synchronize_device(mesh)
        ref_ids = _committed_ids(committed_ref)
        committed_ref.deallocate(True)
        logger.info(f"[ref-eager] committed[:8]={ref_ids[:8].tolist()}")

        # ---- (A) trace-safe adapter EAGER (uniform forward + persistent signal buf) ----
        adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_length)
        adapter.reset_signal_buffer()
        committed_tse = DL.run_fixed_denoise_steps(
            adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_fn, constants=consts
        )
        ttnn.synchronize_device(mesh)
        tse_ids = _committed_ids(committed_tse)
        committed_tse.deallocate(True)
        refactor_match = (ref_ids == tse_ids).float().mean().item()
        print(
            f"RESULT_REFACTOR layers={num_layers} steps={steps} match={refactor_match*100:.1f}% "
            f"committed[:8]={tse_ids[:8].tolist()}",
            flush=True,
        )

        # ---- (B) single-step TRACED loop (N traces, persistent-buffer threading) ----
        traced_ids = None
        try:
            canvas_buf = make_init()  # persistent (holds the init canvas)
            committed_buf = make_init()  # persistent (overwritten with argmax each step)
            adapter.reset_signal_buffer()
            traces = _capture_step_traces(mesh, adapter, cfg, canvas_buf, committed_buf, noise_list, consts)
            ttnn.synchronize_device(mesh)

            init_dev = make_init()
            traced_ids, _warm_ms = _replay_block(mesh, adapter, traces, canvas_buf, committed_buf, init_dev)
            traced_ids2, block_ms = _replay_block(mesh, adapter, traces, canvas_buf, committed_buf, init_dev)
            init_dev.deallocate(True)
            for tid in traces:
                ttnn.release_trace(mesh, tid)

            tt_match = (traced_ids == traced_ids2).float().mean().item()
            tr_match = (ref_ids == traced_ids).float().mean().item()
            print(
                f"RESULT_TRACED_VS_TRACED layers={num_layers} steps={steps} match={tt_match*100:.1f}%",
                flush=True,
            )
            print(
                f"RESULT_SINGLESTEP_TRACED layers={num_layers} steps={steps} block_ms={block_ms:.1f} "
                f"ms_per_step={block_ms/steps:.2f} match_vs_eager={tr_match*100:.1f}% "
                f"committed[:8]={traced_ids[:8].tolist()}",
                flush=True,
            )
            print("SINGLESTEP_OK" if tr_match > 0.999 and tt_match > 0.999 else "SINGLESTEP_MISMATCH", flush=True)
        except Exception as e:
            logger.error(f"SINGLESTEP_FAILED {type(e).__name__}: {str(e)[:400]}")
            print(f"RESULT_SINGLESTEP_BLOCKED {type(e).__name__}: {str(e)[:200]}", flush=True)
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
