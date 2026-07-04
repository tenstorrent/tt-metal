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

This probe was HYPOTHESIZED (session 5) to fix the race: capture ONE denoise step as a
Metal trace and replay it once per step, carrying the cross-step state (canvas + self-cond
signal) in persistent device buffers updated in-place across replays — the KV-cache pattern
a traced decode already uses. N traces are captured (one per step index) so each bakes its
own temperature ``T[i]`` and reads its own renoise tokens ``noise[i]``; all N read/write the
SAME persistent ``canvas_buf`` / ``signal_buf`` / ``committed_buf``.

*** DEVICE-VERIFIED STATUS (2026-07-04 session 8) — the "race" was a PROBE BUG ***
RESOLVED. The historical ``traced_vs_traced`` ~66-92% was NOT a self-cond race, NOT CCL-in-trace,
NOT in-place aliasing — it was THIS PROBE reusing an ``init_dev`` buffer allocated AFTER trace
capture. A Metal trace bakes its intermediate-tensor addresses at capture time; a buffer
allocated into post-capture-freed memory overlaps that trace scratch and is CLOBBERED on every
replay, so the 2nd+ replay copied corrupted data into the canvas. ``probe_selfcond_race.py``
proved it: ``--reuse-init`` = 66% vs ``--reuse-init --prealloc-init`` = 100% (in-place AND
ping-pong bit-identical, refuting the aliasing theory); fresh-upload-per-replay = 100%. Fix here:
allocate ``init_dev`` BEFORE capture. Now ``RESULT_REFACTOR`` = 100%, ``match_vs_eager`` = 100%,
AND ``traced_vs_traced`` = 100%. The single-step traced denoise loop is decision-fidelity-
preserving. ``probe_traced_serving.py`` confirms the same across blocks (CROSSBLOCK_OK, off1
100%). See ``perf_progress.md`` session 8.

It proves, on device:
  A. the trace-safe self-cond adapter refactor (uniform ``forward`` over a persistent
     zeroed-for-step-0 signal buffer) is BIT-EXACT to the eager reference (original
     ``condition(None)`` + prev_logits chain), eager-vs-eager (``RESULT_REFACTOR``).
  B. the single-step trace captures + replays without fatal, and the FIRST replay's committed
     argmax matches the eager reference (``match_vs_eager``); ``traced_vs_traced`` exposes the
     residual self-cond race (< 100%).

*** DEVICE-OWNERSHIP: run only when QB2 is free. A trace-capture FATAL poisons the device
    (the next open_mesh_device hangs) — reset with ``tt-smi -r`` after any capture fatal. ***
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
    """One block: reset persistent state, then replay every step trace in order.

    ``DG_PROBE_SYNC_PER_STEP=1`` inserts a device sync AFTER each per-step trace (so
    trace N's in-place signal_buf WRITE fully lands before trace N+1's READ). Device-
    tested 2026-07-04 session 7: it does NOT fix the self-cond cross-step race
    (traced-vs-traced 90.2% sync-off vs 91.8% sync-on, within noise) — the race is not
    a cross-trace ordering hazard, so it is left OFF by default.
    """
    sync_per_step = os.environ.get("DG_PROBE_SYNC_PER_STEP", "0") == "1"
    ttnn.copy(init_dev, canvas_buf)
    adapter.reset_signal_buffer()  # zero signal_buf so step 0 == post_norm(embed)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for tid in traces:
        ttnn.execute_trace(mesh, tid, blocking=False)
        if sync_per_step:
            ttnn.synchronize_device(mesh)
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
            # Allocate the persistent trace-write-target buffers by CLONING the ACTUAL first-step
            # outputs, so their spec (dtype/layout/memory_config) matches the copy source EXACTLY,
            # and run the copy eagerly to WARM its program cache. This fixes two real trace-capture
            # bugs (device-verified 2026-07-04 session 7):
            #   (1) next_canvas is TILE uint32 but argmax is ROW_MAJOR uint32 (argmax_last_dim) — a
            #       from_torch TILE committed_buf hits "input layout ROW_MAJOR != output TILE"
            #       (copy_device_operation.cpp:114);
            #   (2) run_fixed_denoise_steps threads the canvas via return values, so the two
            #       ttnn.copy(next_canvas->canvas_buf)/(argmax->committed_buf) are never program-cache
            #       warmed — a COLD copy compiled INSIDE begin_trace_capture enqueues a host write ->
            #       "Writes not supported during trace capture" (fd_mesh_command_queue.cpp:665).
            # Clone-from-real-output fixes BOTH: exact spec match + the eager copy warms the program.
            adapter.reset_signal_buffer()
            _c0 = make_init()
            _t0 = DL.temperature_at_step(0, cfg.max_denoise_steps, cfg.temperature_start, cfg.temperature_end)
            _logits0 = adapter(_c0, 0)
            _nc0, _am0 = DL.denoise_step_next_canvas(
                _logits0,
                temperature=_t0,
                entropy_budget=cfg.entropy_budget,
                gumbel_noise=None,
                noise_tokens=noise_list[0],
                constants=consts,
            )
            DL._deallocate_logits_if_unowned(adapter, _logits0)
            canvas_buf = ttnn.clone(_nc0)  # persistent (holds the init canvas; spec == next_canvas)
            committed_buf = ttnn.clone(_am0)  # persistent (spec == argmax, ROW_MAJOR)
            ttnn.copy(_nc0, canvas_buf)  # warm the exact copy programs
            ttnn.copy(_am0, committed_buf)
            _nc0.deallocate(True)
            _am0.deallocate(True)
            _c0.deallocate(True)
            # init_dev (the canvas-init SOURCE copied into canvas_buf at each replay's reset)
            # MUST be allocated BEFORE trace capture. A trace bakes its intermediate-tensor
            # addresses at capture time; a buffer allocated into post-capture-freed memory
            # overlaps that trace scratch and is CLOBBERED by every replay — so REUSING it
            # across the two replays made the 2nd replay copy corrupted data into the canvas.
            # THIS — not a self-cond race, not CCL, not in-place aliasing — is the entire cause
            # of the historical "traced_vs_traced ~90%": probe_selfcond_race.py measured
            # --reuse-init = 66% vs --reuse-init --prealloc-init = 100% (in-place AND ping-pong
            # bit-identical). Reserving init_dev's region before capture keeps trace scratch off
            # it, so reuse is now bit-exact. See perf_progress.md session 8.
            init_dev = make_init()
            ttnn.synchronize_device(mesh)
            adapter.reset_signal_buffer()
            traces = _capture_step_traces(mesh, adapter, cfg, canvas_buf, committed_buf, noise_list, consts)
            ttnn.synchronize_device(mesh)

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
