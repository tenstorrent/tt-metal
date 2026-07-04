# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Traced SERVING denoise loop — cross-block trace reuse + end-to-end t/s (#47465, path to 30).

The single-step trace mechanism (self-cond fixed) is validated by
``probe_singlestep_traced.py``. This probe validates the SECOND blocker — cross-block
trace reuse — and measures the real serving t/s.

The serving denoise attends to a FROZEN ``prompt_len`` prefix (``_set_q_rope_offset``
only advances the canvas RoPE offset; commit writes KV BEYOND ``prompt_len`` so the
read prefix is invariant), so the ONLY per-block variation inside the denoise is the
canvas RoPE (``q_rope_offset = start_pos``). The constant-shape canvas RoPE buffer
(``DenoiseLogitsAdapter.prepare_canvas_rope_buffers`` / ``update_canvas_rope_buffers``)
carries that variation as buffer CONTENT refreshed per block OUTSIDE the trace, so ONE
captured set of single-step traces replays for EVERY block.

It proves, on device:
  REFACTOR:  trace-safe self-cond eager == original eager (adapter refactor is bit-exact).
  CROSSBLOCK_ROPE:  N traces captured at block 0's offset, replayed at block 0 AND block 1
    offsets (canvas RoPE buffer refreshed between), match the EAGER reference committed
    argmax at each offset — i.e. the canvas-RoPE buffer is bit-exact to the growing-slice
    RoPE and the trace is cross-block reusable.
  SERVING_PERF:  a real multi-block denoise(traced)+commit(batched) loop — measured
    per-block latency / tokens-per-block-per-s at the given depth.

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
from models.experimental.diffusion_gemma.tt.commit_batched import commit_canvas_tokens_batched
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
    """Capture one Metal trace per step index; each threads the persistent buffers.

    The adapter must have use_canvas_rope=True and the canvas RoPE buffers populated
    for SOME offset before capture (the ops run once during capture)."""
    traces = []
    for step in range(cfg.max_denoise_steps):
        temperature = DL.temperature_at_step(step, cfg.max_denoise_steps, cfg.temperature_start, cfg.temperature_end)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        logits = adapter(canvas_buf, step)
        next_canvas, argmax = DL.denoise_step_next_canvas(
            logits,
            temperature=temperature,
            entropy_budget=cfg.entropy_budget,
            gumbel_noise=None,
            noise_tokens=noise_list[step],
            constants=consts,
        )
        DL._deallocate_logits_if_unowned(adapter, logits)
        ttnn.copy(next_canvas, canvas_buf)
        ttnn.copy(argmax, committed_buf)
        next_canvas.deallocate(True)
        argmax.deallocate(True)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        traces.append(tid)
    return traces


def _replay_block(mesh, adapter, traces, canvas_buf, committed_buf, init_dev, start_pos):
    """One block: refresh canvas RoPE for start_pos, reset state, replay every step trace."""
    adapter.update_canvas_rope_buffers(start_pos)  # per-block RoPE content update (OUTSIDE trace)
    ttnn.copy(init_dev, canvas_buf)
    adapter.reset_signal_buffer()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for tid in traces:
        ttnn.execute_trace(mesh, tid, blocking=False)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1e3
    return _committed_ids(committed_buf), ms


def run(num_layers, canvas_length, steps, num_blocks, prompt, max_seq_len, do_commit, trace_region_size=6000000000):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    # Single-step traces are captured ONE-PER-STEP, so total trace memory ~ steps * per-step
    # trace size (~168 MB/trace at 30L). 12 steps at 30L needs ~2.02 GB, 24 needs ~4.03 GB, so
    # default 6 GB covers up to ~36 steps. DRAM headroom on QB2 (~32 GB/chip) is ample.
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_region_size)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = mi.tt_model
        prompt_tokens = tokenize_prompt(mi.tokenizer, prompt)
        prefill = prefill_prompt_tokens(tt_model, prompt_tokens)
        ttnn.synchronize_device(mesh)
        prompt_len = prefill.cache_len
        logger.info(f"prefilled cache_len={prompt_len}")

        adapter_kwargs = {}
        cfg_hf = getattr(tt_model, "hf_config", None)
        if cfg_hf is not None:
            adapter_kwargs["config"] = cfg_hf
        builder = make_generation_logits_fn_builder_from_checkpoint_state(mi.state_dict, **adapter_kwargs)
        adapter = builder(tt_model, prompt_tokens=prompt_tokens, prompt_len=prompt_len)

        cfg = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
        vocab = int(getattr(mi.tokenizer, "vocab_size", 262144))
        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=cfg.entropy_budget)

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

        off0 = prompt_len
        off1 = prompt_len + canvas_length

        # ---- (ref) ORIGINAL eager (prev_logits chain, growing-slice RoPE) at off0 ----
        adapter.q_rope_offset = off0
        committed_ref0 = DL.run_fixed_denoise_steps(
            adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_fn, constants=consts
        )
        ttnn.synchronize_device(mesh)
        ref0_ids = _committed_ids(committed_ref0)
        committed_ref0.deallocate(True)
        adapter.reset()

        # ---- (A) trace-safe adapter EAGER (uniform forward + persistent signal buf), off0/off1 ----
        adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_length)

        def eager_ts(off):
            adapter.q_rope_offset = off
            adapter.use_canvas_rope = False  # growing-slice RoPE (reference)
            adapter.reset_signal_buffer()
            committed = DL.run_fixed_denoise_steps(
                adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_fn, constants=consts
            )
            ttnn.synchronize_device(mesh)
            ids = _committed_ids(committed)
            committed.deallocate(True)
            return ids

        ts0_ids = eager_ts(off0)
        ts1_ids = eager_ts(off1)
        refactor_match = (ref0_ids == ts0_ids).float().mean().item()
        print(
            f"RESULT_REFACTOR layers={num_layers} steps={steps} match={refactor_match*100:.1f}% "
            f"committed[:8]={ts0_ids[:8].tolist()}",
            flush=True,
        )

        # ---- (B) canvas RoPE + single-step TRACED loop, cross-block replay ----
        traced_ok = False
        try:
            adapter.prepare_canvas_rope_buffers(canvas_len=canvas_length)
            adapter.update_canvas_rope_buffers(off0)  # populate for capture
            adapter.q_rope_offset = off0
            adapter.use_canvas_rope = True  # traced path uses the constant-shape canvas RoPE buffer

            # WARM the persistent trace-write-target buffers by cloning the REAL first-step
            # outputs (exact dtype/layout/memory spec match) and running the in-trace copies
            # ONCE eagerly. Otherwise the cold copy(next_canvas->canvas_buf)/(argmax->committed_buf)
            # compiled INSIDE begin_trace_capture enqueues a host write -> "Writes not supported
            # during trace capture" (fd_mesh_command_queue.cpp:665). Same fix as
            # probe_singlestep_traced.py session 7; run under use_canvas_rope so the RoPE path warms too.
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
            canvas_buf = ttnn.clone(_nc0)
            committed_buf = ttnn.clone(_am0)
            ttnn.copy(_nc0, canvas_buf)
            ttnn.copy(_am0, committed_buf)
            _nc0.deallocate(True)
            _am0.deallocate(True)
            _c0.deallocate(True)
            ttnn.synchronize_device(mesh)

            # init_dev (the canvas-init SOURCE copied into canvas_buf each block) MUST be
            # allocated BEFORE trace capture. A trace bakes its intermediate-tensor
            # addresses at capture time; a buffer allocated into post-capture-freed memory
            # overlaps that trace scratch and is CLOBBERED by every replay — so reusing it
            # across replays/blocks corrupts the canvas from the 2nd replay on. Reserving its
            # region before capture keeps trace scratch off it. (Root cause of the phantom
            # "self-cond race": probe_selfcond_race.py --reuse-init vs --reuse-init
            # --prealloc-init = 66% vs 100%. See perf_progress.md session 8.)
            init_dev = make_init()
            adapter.reset_signal_buffer()
            traces = _capture_step_traces(mesh, adapter, cfg, canvas_buf, committed_buf, noise_list, consts)
            ttnn.synchronize_device(mesh)

            traced0_ids, warm_ms = _replay_block(mesh, adapter, traces, canvas_buf, committed_buf, init_dev, off0)
            traced0b_ids, b0_ms = _replay_block(mesh, adapter, traces, canvas_buf, committed_buf, init_dev, off0)
            traced1_ids, b1_ms = _replay_block(mesh, adapter, traces, canvas_buf, committed_buf, init_dev, off1)

            m0 = (traced0_ids == ts0_ids).float().mean().item()
            m0_tt = (traced0_ids == traced0b_ids).float().mean().item()
            m1 = (traced1_ids == ts1_ids).float().mean().item()
            print(
                f"RESULT_CROSSBLOCK_ROPE layers={num_layers} steps={steps} "
                f"off0_match_vs_eager={m0*100:.1f}% off0_tvt={m0_tt*100:.1f}% off1_match_vs_eager={m1*100:.1f}% "
                f"ms_per_step={b0_ms/steps:.2f}",
                flush=True,
            )
            traced_ok = m0 > 0.999 and m0_tt > 0.999 and m1 > 0.999
            print("CROSSBLOCK_OK" if traced_ok else "CROSSBLOCK_MISMATCH", flush=True)

            # ---- (C) real serving t/s: traced denoise + batched commit, num_blocks ----
            if do_commit:
                block_lat = []
                next_pos = prompt_len
                for b in range(num_blocks):
                    ttnn.synchronize_device(mesh)
                    tb = time.perf_counter()
                    committed_ids, _ = _replay_block(
                        mesh, adapter, traces, canvas_buf, committed_buf, init_dev, next_pos
                    )
                    committed_host = committed_ids.reshape(1, canvas_length)
                    commit_canvas_tokens_batched(tt_model, committed_host, start_pos=next_pos)
                    ttnn.synchronize_device(mesh)
                    lat = time.perf_counter() - tb
                    block_lat.append(lat)
                    next_pos += canvas_length
                    logger.info(f"[serving] block {b} start_pos={next_pos-canvas_length} latency_s={lat:.3f}")
                # steady-state = drop block 0 (includes first-commit warmup) if >1
                steady = block_lat[1:] if len(block_lat) > 1 else block_lat
                mean_lat = sum(steady) / len(steady)
                tps = canvas_length / mean_lat
                print(
                    f"RESULT_SERVING_PERF layers={num_layers} steps={steps} blocks={num_blocks} "
                    f"per_block_latency_s={[round(x,3) for x in block_lat]} mean_steady_s={mean_lat:.3f} "
                    f"tokens_per_block_per_s={tps:.2f}",
                    flush=True,
                )

            init_dev.deallocate(True)
            for tid in traces:
                ttnn.release_trace(mesh, tid)
        except Exception as e:
            logger.exception("SERVING_TRACED_FAILED")
            print(f"RESULT_SERVING_BLOCKED {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=6)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--num-blocks", type=int, default=3)
    ap.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--no-commit", action="store_true", help="skip the commit/serving-perf phase (correctness only)")
    ap.add_argument("--trace-region-size", type=int, default=6000000000)
    args = ap.parse_args()
    run(
        args.num_layers,
        args.canvas_length,
        args.steps,
        args.num_blocks,
        args.prompt,
        args.max_seq_len,
        do_commit=not args.no_commit,
        trace_region_size=args.trace_region_size,
    )


if __name__ == "__main__":
    main()
