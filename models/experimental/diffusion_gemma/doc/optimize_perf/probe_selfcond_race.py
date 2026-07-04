# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Self-cond trace-race characterization + ping-pong fix (#47465, path to 30).

Session 7 found the single-step traced denoise loop RUNS but ``traced_vs_traced`` ~ 90%
(two replays of the SAME captured traces disagree) with self-cond ON, while ``match_vs_eager``
(first replay) = 100%. Session 7 attributed the residual to a "CCL-in-trace" issue and called
it likely-upstream. BUT the self-cond module is REPLICATED across the mesh (no TP shard, no
all-reduce — see ``tt/self_conditioning.py`` docstring + replicated ``as_tensor`` weights), so
the only thing self-cond ON adds over OFF is the cross-step ``signal_buf`` **in-place read+write
feedback**: each step reads ``signal_buf`` at the start of its trace and writes it in-place at the
end. That is a WAR/anti-dependency on a FIXED-address buffer — something a Metal trace can
mishandle, and which the eager path (fresh-allocs the signal every step) never hits.

This probe settles two questions in ONE device session (model load dominates cost):

  Q1 (serving relevance): is a single FRESH-RESET traced replay RELIABLY == eager? Serving does
     exactly one reset-then-replay per block, so if K independent fresh replays each match eager
     (100%), the traced serving loop is already decision-fidelity-preserving and ``traced_vs_traced``
     is a red herring. If fresh replays randomly match ~90%, it is a real serving fidelity gap.

  Q2 (fix): does PING-PONG double-buffering (read buffer != write buffer, no in-place alias)
     eliminate the cross-replay disagreement?

For each scheme (A=in-place, B=ping-pong) it: prepares the trace-safe adapter, captures N
single-step traces, then runs K INDEPENDENT fresh-reset replays and reports each replay's committed
argmax vs the eager reference AND the full pairwise agreement matrix.

*** DEVICE-OWNERSHIP: run only when QB2 is free. A trace-capture FATAL poisons the device
    (next open_mesh_device hangs) — reset with ``tt-smi -r`` after any capture fatal. ***
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
    """Capture one Metal trace per step index; each threads the persistent buffers.

    ``adapter(canvas_buf, step)`` internally reads/writes the self-cond signal buffer(s)
    according to the adapter's scheme (in-place or ping-pong) — the trace bakes those
    addresses, so the scheme is fixed at capture time.
    """
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


def _replay_block(mesh, adapter, traces, canvas_buf, committed_buf, init_dev, sync_per_step=False):
    """One block: reset persistent state (canvas + signal), replay every step trace in order."""
    ttnn.copy(init_dev, canvas_buf)
    adapter.reset_signal_buffer()
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    for tid in traces:
        ttnn.execute_trace(mesh, tid, blocking=False)
        if sync_per_step:
            ttnn.synchronize_device(mesh)
    ttnn.synchronize_device(mesh)
    ms = (time.perf_counter() - t0) * 1e3
    return _committed_ids(committed_buf), ms


def _warm_persistent_buffers(mesh, adapter, cfg, make_init, noise_list, consts):
    """Allocate canvas_buf/committed_buf by cloning REAL first-step outputs (spec match) and
    warm the copy programs — fixes the two session-7 trace-capture bugs (cold-copy host-write +
    argmax ROW_MAJOR vs committed TILE). Returns (canvas_buf, committed_buf)."""
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
    return canvas_buf, committed_buf


def _report(scheme, ref_ids, replay_ids_list):
    n = len(replay_ids_list)
    vs_eager = []
    for i, ids in enumerate(replay_ids_list):
        m = (ref_ids == ids).float().mean().item() * 100
        vs_eager.append(m)
        print(
            f"RESULT_{scheme}_REPLAY{i}_VS_EAGER match={m:.2f}% committed[:8]={ids[:8].tolist()}",
            flush=True,
        )
    min_pair = 100.0
    for i in range(n):
        for j in range(i + 1, n):
            m = (replay_ids_list[i] == replay_ids_list[j]).float().mean().item() * 100
            min_pair = min(min_pair, m)
            print(f"RESULT_{scheme}_PAIR_{i}_{j} match={m:.2f}%", flush=True)
    all_ident = min_pair >= 99.999
    all_match_eager = min(vs_eager) >= 99.999
    print(
        f"SUMMARY_{scheme} replays={n} min_vs_eager={min(vs_eager):.2f}% max_vs_eager={max(vs_eager):.2f}% "
        f"min_pairwise={min_pair:.2f}% all_replays_identical={all_ident} all_match_eager={all_match_eager}",
        flush=True,
    )
    return all_match_eager and all_ident


def _run_scheme(
    scheme,
    mesh,
    adapter,
    cfg,
    canvas_buf,
    committed_buf,
    noise_list,
    consts,
    make_init,
    ref_ids,
    replays,
    sync_per_step,
    reuse_init=False,
    prealloc_init=False,
):
    """Capture + K reset replays for the CURRENT adapter buffer scheme.

    ``reuse_init``: allocate ONE init_dev and reuse it across replays (the session-7
    ``probe_singlestep_traced.py`` pattern). Default False = fresh device upload per
    replay (the serving pattern: each block resets from its own canvas init).
    ``prealloc_init``: allocate the reused init_dev BEFORE trace capture (reserves its
    region so trace scratch cannot overlap it). If this makes reuse match the fresh
    pattern (100%), the divergence is proven to be trace-scratch clobbering a buffer
    that was allocated into post-capture-freed memory — a probe artifact, not a model
    or self-cond race."""
    replay_ids_list = []
    shared_init_pre = make_init() if (reuse_init and prealloc_init) else None
    adapter.reset_signal_buffer()
    traces = _capture_step_traces(mesh, adapter, cfg, canvas_buf, committed_buf, noise_list, consts)
    ttnn.synchronize_device(mesh)
    shared_init = shared_init_pre if shared_init_pre is not None else (make_init() if reuse_init else None)
    for _ in range(replays):
        init_dev = shared_init if reuse_init else make_init()
        ids, _ms = _replay_block(
            mesh, adapter, traces, canvas_buf, committed_buf, init_dev, sync_per_step=sync_per_step
        )
        if not reuse_init:
            init_dev.deallocate(True)
        replay_ids_list.append(ids)
    if shared_init is not None:
        shared_init.deallocate(True)
    for tid in traces:
        ttnn.release_trace(mesh, tid)
    return _report(scheme, ref_ids, replay_ids_list)


def run(
    num_layers,
    canvas_length,
    steps,
    prompt,
    max_seq_len,
    replays,
    schemes,
    sync_per_step,
    reuse_init=False,
    prealloc_init=False,
):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=2000000000)
    try:
        t_load = time.perf_counter()
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        logger.info(f"[load] model built in {time.perf_counter() - t_load:.1f}s")
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

        # ---- eager reference (original prev_logits chain, condition(None) step 0) ----
        committed_ref = DL.run_fixed_denoise_steps(
            adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_fn, constants=consts
        )
        ttnn.synchronize_device(mesh)
        ref_ids = _committed_ids(committed_ref)
        committed_ref.deallocate(True)
        logger.info(f"[ref-eager] committed[:8]={ref_ids[:8].tolist()}")

        # Persistent trace-write-target buffers (shared across schemes; spec-cloned + warmed once).
        # Warm under the in-place adapter so signal_buf exists.
        adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_length, ping_pong=False)
        canvas_buf, committed_buf = _warm_persistent_buffers(mesh, adapter, cfg, make_init, noise_list, consts)

        overall_pass = True
        if "A" in schemes:
            # Scheme A: in-place single signal buffer (current default; expected racy per session 7).
            adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_length, ping_pong=False)
            okA = _run_scheme(
                "INPLACE",
                mesh,
                adapter,
                cfg,
                canvas_buf,
                committed_buf,
                noise_list,
                consts,
                make_init,
                ref_ids,
                replays,
                sync_per_step,
                reuse_init=reuse_init,
                prealloc_init=prealloc_init,
            )
            overall_pass = overall_pass and okA
        if "B" in schemes:
            # Scheme B: ping-pong double buffer (read buf != write buf; the candidate fix).
            adapter.prepare_trace_safe_self_conditioning(canvas_len=canvas_length, ping_pong=True)
            okB = _run_scheme(
                "PINGPONG",
                mesh,
                adapter,
                cfg,
                canvas_buf,
                committed_buf,
                noise_list,
                consts,
                make_init,
                ref_ids,
                replays,
                sync_per_step,
                reuse_init=reuse_init,
                prealloc_init=prealloc_init,
            )
            overall_pass = overall_pass and ("B" in schemes and okB)

        print(f"PROBE_DONE overall_pass={overall_pass}", flush=True)
    except Exception as e:
        logger.error(f"PROBE_FAILED {type(e).__name__}: {str(e)[:500]}")
        print(f"RESULT_PROBE_BLOCKED {type(e).__name__}: {str(e)[:300]}", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--replays", type=int, default=5)
    ap.add_argument("--schemes", default="AB", help="which schemes to run: A (in-place), B (ping-pong), or AB")
    ap.add_argument("--sync-per-step", action="store_true")
    ap.add_argument("--reuse-init", action="store_true", help="reuse one init_dev across replays (session-7 pattern)")
    ap.add_argument("--prealloc-init", action="store_true", help="with --reuse-init, allocate init_dev BEFORE capture")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-seq-len", type=int, default=512)
    args = ap.parse_args()
    run(
        args.num_layers,
        args.canvas_length,
        args.steps,
        args.prompt,
        args.max_seq_len,
        args.replays,
        args.schemes,
        args.sync_per_step,
        reuse_init=args.reuse_init,
        prealloc_init=args.prealloc_init,
    )


if __name__ == "__main__":
    main()
