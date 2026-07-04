# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Multi-step trace batching — device verify + block(K) bench (#47465, path to 100 t/s, lever 10).

The landed single-step traced serving loop (``tt/traced_denoise.py``,
``probe_traced_serving.py``) replays **N single-step traces per block** — one
``execute_trace`` per denoise step — so the per-block fixed dispatch overhead scales with the
step count: ``block(K) ≈ 0.275·K + 1.09 s`` (58.3 t/s @12, 33.3 @24). 100 t/s ⇔
``block ≤ 2.56 s``, which single-step replays reach only at ``K ≤ ~5``.

:class:`~models.experimental.diffusion_gemma.tt.traced_denoise.MultiStepTracedDenoiseController`
captures a WINDOW of ``G`` steps into ONE Metal trace (default ``G = K`` ⇒ the whole block in
ONE capture + ONE replay), so a block does ``ceil(K/G)`` replays instead of ``K``. This script
proves, on device, that the multi-step loop is BIT-EXACT to the single-step loop and to the
eager reference, and measures how much it flattens the per-block dispatch term.

It reports:
  MULTISTEP_VERIFY:  eager ``run_fixed_denoise_steps`` vs single-step traced vs multi-step
    traced committed argmax, at block-0 offset (twice, for replay determinism) and block-1
    offset (canvas RoPE refreshed between) — all three must be byte-identical
    (``MULTISTEP_BITEXACT``).
  MULTISTEP_PERF:  per-block DENOISE-REPLAY latency (no commit; the portion multi-step
    optimizes) for the single-step loop vs the multi-step loop, swept over step counts, with a
    least-squares ``denoise(K) = a·K + b`` fit for each. Multi-step should collapse the fixed
    intercept ``b`` (K replay dispatches → ``ceil(K/G)``) while leaving the per-step compute
    slope ``a`` unchanged.
  MULTISTEP_PROJECTION:  the largest step budget ``K`` at which each loop holds ``block ≤ 2.56 s``
    (= 100 t/s), using ``block(K) = denoise(K) + commit_ms``. ``--commit-ms`` supplies the
    additive batched-commit constant (unchanged by multi-step); pass ``--measure-commit`` to time
    ONE real batched commit and use that (it mutates the KV cache, so it runs LAST).

Commit is deliberately excluded from the swept timing: multi-step trace batching changes only
the denoise replay, and running two serving loops (single then multi) with commit on one model
build would double-commit into the same cache. The commit cost is an additive per-block constant
folded back in for the 100 t/s projection.

Grouping (``--group G`` / ``DG_DENOISE_MULTISTEP_GROUP``) is the trace-region memory knob: a
whole-block capture records ``K`` steps of commands (heavy per-step logits/activation
intermediates are freed between steps inside the capture, so peak scratch is ~1 step's). If a
whole-block capture overflows ``--trace-region-size`` the run prints ``MULTISTEP_BLOCKED`` with
the region size; retry with a larger region or a smaller ``--group``.

*** DEVICE-OWNERSHIP: run only when QB2 is free. This script is NOT run at authoring time. ***

Examples:
  # verify bit-exactness at 8 steps (whole-block capture), 6 layers, quick:
  python bench_multistep_trace.py --mode verify --num-layers 6 --steps 8
  # full 30L bench sweep, whole-block multi-step vs single-step:
  python bench_multistep_trace.py --mode perf --num-layers 30 --steps-sweep 8,12,16,20,24
  # everything, with a measured commit constant folded into the 100 t/s projection:
  python bench_multistep_trace.py --mode all --num-layers 30 --steps-sweep 8,12,16,20,24 --measure-commit
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
from models.experimental.diffusion_gemma.tt.traced_denoise import (
    MultiStepTracedDenoiseController,
    TracedDenoiseController,
)

CKPT = os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it")

# 100 t/s target for a 256-token canvas: block(K) ≤ 256 / 100 s = 2.56 s.
TARGET_BLOCK_S = 256.0 / 100.0


def _committed_flat(t_or_host) -> torch.Tensor:
    """Flat [L] long tensor from either a device committed tensor or a host trajectory tensor."""
    if isinstance(t_or_host, torch.Tensor):
        return t_or_host.reshape(-1).long()
    return ttnn.to_torch(ttnn.get_device_tensors(t_or_host)[0]).reshape(-1).long()


def _linfit(xs, ys):
    """Least-squares slope/intercept for ys = a*xs + b (pure python; xs are step counts)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 0.0, sy / n
    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n
    return a, b


def _max_k_under_target(a_s: float, b_s: float, commit_s: float) -> int:
    """Largest integer K with block(K) = a_s*K + b_s + commit_s ≤ TARGET_BLOCK_S (0 if none)."""
    if a_s <= 0:
        return 999 if (b_s + commit_s) <= TARGET_BLOCK_S else 0
    k = (TARGET_BLOCK_S - b_s - commit_s) / a_s
    return max(0, int(k))


def _make_env(mesh, tokenizer, tt_model, state_dict, prompt, canvas_length, max_steps, seed_noise=1, seed_init=7):
    """Build the prefilled adapter + a deterministic noise/init stream shared by all loops."""
    prompt_tokens = tokenize_prompt(tokenizer, prompt)
    prefill = prefill_prompt_tokens(tt_model, prompt_tokens)
    ttnn.synchronize_device(mesh)
    prompt_len = prefill.cache_len
    logger.info(f"prefilled cache_len={prompt_len}")

    adapter_kwargs = {}
    cfg_hf = getattr(tt_model, "hf_config", None)
    if cfg_hf is not None:
        adapter_kwargs["config"] = cfg_hf
    builder = make_generation_logits_fn_builder_from_checkpoint_state(state_dict, **adapter_kwargs)
    adapter = builder(tt_model, prompt_tokens=prompt_tokens, prompt_len=prompt_len)

    vocab = int(getattr(tokenizer, "vocab_size", 262144))
    # One persistent noise tensor per step index (max_steps of them); every loop consumes
    # clone(noise_list[step]) in step order, so the seeded stream is identical across
    # eager / single-step / multi-step for a given K.
    gen = torch.Generator().manual_seed(seed_noise)
    noise_list = [
        host_canvas_to_device(mesh, torch.randint(0, vocab, (1, canvas_length), dtype=torch.long, generator=gen))
        for _ in range(max_steps)
    ]
    init_host = torch.randint(
        0, vocab, (1, canvas_length), dtype=torch.long, generator=torch.Generator().manual_seed(seed_init)
    )
    return adapter, prompt_len, noise_list, init_host


def _release_adapter_persistent(adapter):
    """Free the adapter's per-capture persistent buffers between controllers/K iterations."""
    if hasattr(adapter, "release_canvas_rope_buffers"):
        adapter.release_canvas_rope_buffers()
    if getattr(adapter, "signal_buf", None) is not None:
        adapter.signal_buf.deallocate(True)
        adapter.signal_buf = None
    if getattr(adapter, "signal_buf_b", None) is not None:
        adapter.signal_buf_b.deallocate(True)
        adapter.signal_buf_b = None
    adapter.trace_safe_self_conditioning = False
    if hasattr(adapter, "reset"):
        adapter.reset()


def _eager_committed(mesh, adapter, cfg, make_init, noise_fn, consts, off):
    """Trace-safe eager reference (uniform forward + persistent signal, growing-slice RoPE).

    Proven == the original eager path in ``probe_traced_serving.py`` (RESULT_REFACTOR 100%),
    used here as the ground-truth committed argmax the traced loops must match."""
    adapter.prepare_trace_safe_self_conditioning(canvas_len=cfg.canvas_length)
    adapter.q_rope_offset = off
    adapter.use_canvas_rope = False  # growing-slice RoPE (reference)
    adapter.reset_signal_buffer()
    committed = DL.run_fixed_denoise_steps(
        adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_fn, constants=consts
    )
    ttnn.synchronize_device(mesh)
    ids = _committed_flat(committed)
    committed.deallocate(True)
    _release_adapter_persistent(adapter)
    return ids


def _controller_committed(mesh, controller, adapter, cfg, make_init, noise_fn, off):
    """Run one traced denoise block through a controller; return the committed flat ids."""
    adapter.q_rope_offset = off
    traj = controller.denoise_block(adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_fn)
    return _committed_flat(traj.committed)


def verify(mesh, adapter, prompt_len, noise_list, init_host, canvas_length, steps, group):
    """Bit-exactness: eager vs single-step vs multi-step at block-0 (x2) and block-1 offsets."""
    cfg = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
    consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=cfg.entropy_budget)

    def make_init():
        return host_canvas_to_device(mesh, init_host)

    def noise_fn(step):
        return ttnn.clone(noise_list[step])

    off0 = prompt_len
    off1 = prompt_len + canvas_length

    eager0 = _eager_committed(mesh, adapter, cfg, make_init, noise_fn, consts, off0)
    eager1 = _eager_committed(mesh, adapter, cfg, make_init, noise_fn, consts, off1)

    single = TracedDenoiseController(mesh, cfg, consts=consts)
    s0 = _controller_committed(mesh, single, adapter, cfg, make_init, noise_fn, off0)  # capture block
    s0b = _controller_committed(mesh, single, adapter, cfg, make_init, noise_fn, off0)  # steady replay
    s1 = _controller_committed(mesh, single, adapter, cfg, make_init, noise_fn, off1)
    single.release()
    _release_adapter_persistent(adapter)

    multi = MultiStepTracedDenoiseController(mesh, cfg, consts=consts, group_size=group)
    m0 = _controller_committed(mesh, multi, adapter, cfg, make_init, noise_fn, off0)  # capture block
    m0b = _controller_committed(mesh, multi, adapter, cfg, make_init, noise_fn, off0)  # steady replay
    m1 = _controller_committed(mesh, multi, adapter, cfg, make_init, noise_fn, off1)
    n_traces = len(multi.traces)
    multi.release()
    _release_adapter_persistent(adapter)

    def match(a, b):
        return (a == b).float().mean().item() * 100.0

    print(
        f"RESULT_MULTISTEP_VERIFY steps={steps} group={multi.group_size} windows(replays/block)={n_traces} "
        f"single_vs_eager_off0={match(s0, eager0):.1f}% single_off0_tvt={match(s0, s0b):.1f}% "
        f"single_vs_eager_off1={match(s1, eager1):.1f}% "
        f"multi_vs_eager_off0={match(m0, eager0):.1f}% multi_off0_tvt={match(m0, m0b):.1f}% "
        f"multi_vs_eager_off1={match(m1, eager1):.1f}% "
        f"multi_vs_single_off0={match(m0, s0):.1f}% multi_vs_single_off1={match(m1, s1):.1f}%",
        flush=True,
    )
    bitexact = all(
        match(x, y) > 99.999
        for x, y in [(s0, eager0), (s0, s0b), (s1, eager1), (m0, eager0), (m0, m0b), (m1, eager1), (m0, s0), (m1, s1)]
    )
    print("MULTISTEP_BITEXACT" if bitexact else "MULTISTEP_MISMATCH", flush=True)
    return bitexact


def _bench_denoise_only(mesh, controller, adapter, cfg, make_init, noise_fn, prompt_len, canvas_length, num_blocks):
    """Mean per-block denoise-replay latency (s), advancing the offset per block, NO commit.

    Block 0 (capture/compile) is dropped as warmup; blocks 1..num_blocks-1 are the steady mean.
    Denoise reads the FROZEN prompt prefix (no commit ⇒ prefix invariant), so the measured
    per-block cost is representative of steady-state denoise latency."""
    lats = []
    for b in range(num_blocks):
        off = prompt_len + b * canvas_length
        adapter.q_rope_offset = off
        ttnn.synchronize_device(mesh)
        t0 = time.perf_counter()
        traj = controller.denoise_block(adapter, make_init(), cfg, gumbel_noise_fn=None, noise_tokens_fn=noise_fn)
        ttnn.synchronize_device(mesh)
        lats.append(time.perf_counter() - t0)
        _ = traj.committed
    steady = lats[1:] if len(lats) > 1 else lats
    return sum(steady) / len(steady), lats


def perf(mesh, adapter, prompt_len, noise_list, init_host, canvas_length, steps_sweep, group, num_blocks):
    """Denoise-only latency sweep: single-step vs multi-step per-block latency + linear fit."""
    init_host_t = init_host

    def make_init():
        return host_canvas_to_device(mesh, init_host_t)

    rows = []  # (K, single_s, multi_s, n_windows)
    for k in steps_sweep:
        cfg = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=k)
        consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=cfg.entropy_budget)

        def noise_fn(step):
            return ttnn.clone(noise_list[step])

        single = TracedDenoiseController(mesh, cfg, consts=consts)
        single_s, _ = _bench_denoise_only(
            mesh, single, adapter, cfg, make_init, noise_fn, prompt_len, canvas_length, num_blocks
        )
        single.release()
        _release_adapter_persistent(adapter)

        multi = MultiStepTracedDenoiseController(mesh, cfg, consts=consts, group_size=group)
        multi_s, _ = _bench_denoise_only(
            mesh, multi, adapter, cfg, make_init, noise_fn, prompt_len, canvas_length, num_blocks
        )
        n_windows = len(multi.traces)
        multi.release()
        _release_adapter_persistent(adapter)

        rows.append((k, single_s, multi_s, n_windows))
        print(
            f"RESULT_MULTISTEP_PERF steps={k} group={min(group or k, k)} multi_windows={n_windows} "
            f"single_denoise_s={single_s:.3f} multi_denoise_s={multi_s:.3f} "
            f"single_denoise_tps={canvas_length / single_s:.2f} multi_denoise_tps={canvas_length / multi_s:.2f} "
            f"speedup={single_s / multi_s:.2f}x",
            flush=True,
        )

    ks = [r[0] for r in rows]
    a_s, b_s = _linfit(ks, [r[1] for r in rows])
    a_m, b_m = _linfit(ks, [r[2] for r in rows])
    print(
        f"RESULT_MULTISTEP_FIT single_denoise(K)={a_s:.4f}*K+{b_s:.4f}s "
        f"multi_denoise(K)={a_m:.4f}*K+{b_m:.4f}s intercept_drop={b_s - b_m:.4f}s",
        flush=True,
    )
    return rows, (a_s, b_s), (a_m, b_m)


def measure_commit(mesh, adapter, tt_model, prompt_len, noise_list, init_host, canvas_length, steps):
    """Time ONE real batched commit (mutates the KV cache — call LAST). Returns commit seconds."""
    cfg = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
    consts = DL.make_denoise_constants(mesh, batch=1, canvas_len=canvas_length, budget=cfg.entropy_budget)

    def make_init():
        return host_canvas_to_device(mesh, init_host)

    def noise_fn(step):
        return ttnn.clone(noise_list[step])

    multi = MultiStepTracedDenoiseController(mesh, cfg, consts=consts, group_size=None)
    committed_flat = _controller_committed(mesh, multi, adapter, cfg, make_init, noise_fn, prompt_len)
    multi.release()
    _release_adapter_persistent(adapter)

    committed_host = committed_flat.reshape(1, canvas_length)
    ttnn.synchronize_device(mesh)
    t0 = time.perf_counter()
    commit_canvas_tokens_batched(tt_model, committed_host, start_pos=prompt_len)
    ttnn.synchronize_device(mesh)
    commit_s = time.perf_counter() - t0
    print(f"RESULT_MULTISTEP_COMMIT batched_commit_s={commit_s:.3f}", flush=True)
    return commit_s


def run(
    num_layers,
    canvas_length,
    steps,
    steps_sweep,
    group,
    num_blocks,
    prompt,
    max_seq_len,
    mode,
    commit_ms,
    do_measure_commit,
    trace_region_size,
):
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D, ttnn.FabricReliabilityMode.STRICT_INIT, None)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=trace_region_size)
    try:
        mi = build_tt_model_from_checkpoint_dir(
            mesh, CKPT, max_batch_size=1, max_seq_len=max_seq_len, num_layers=num_layers, create_kv_cache=True
        )
        tt_model = mi.tt_model
        max_steps = max([steps, *steps_sweep]) if steps_sweep else steps
        adapter, prompt_len, noise_list, init_host = _make_env(
            mesh, mi.tokenizer, tt_model, mi.state_dict, prompt, canvas_length, max_steps
        )

        if mode in ("verify", "all"):
            verify(mesh, adapter, prompt_len, noise_list, init_host, canvas_length, steps, group)

        commit_s = (commit_ms / 1e3) if commit_ms is not None else 0.0
        if mode in ("perf", "all"):
            _, (a_s, b_s), (a_m, b_m) = perf(
                mesh, adapter, prompt_len, noise_list, init_host, canvas_length, steps_sweep, group, num_blocks
            )
            if do_measure_commit:
                commit_s = measure_commit(
                    mesh, adapter, tt_model, prompt_len, noise_list, init_host, canvas_length, steps_sweep[-1]
                )
            k_single = _max_k_under_target(a_s, b_s, commit_s)
            k_multi = _max_k_under_target(a_m, b_m, commit_s)
            print(
                f"RESULT_MULTISTEP_PROJECTION commit_s={commit_s:.3f} target_block_s={TARGET_BLOCK_S:.2f} "
                f"max_steps_at_100tps_single={k_single} max_steps_at_100tps_multi={k_multi}",
                flush=True,
            )
    except Exception as e:
        logger.exception("MULTISTEP_BENCH_FAILED")
        print(
            f"RESULT_MULTISTEP_BLOCKED trace_region_size={trace_region_size} {type(e).__name__}: {str(e)[:200]}",
            flush=True,
        )
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["verify", "perf", "all"], default="all")
    ap.add_argument("--num-layers", type=int, default=30)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8, help="step count for the verify phase")
    ap.add_argument("--steps-sweep", default="8,12,16,20,24", help="comma-separated K sweep for the perf phase")
    ap.add_argument(
        "--group",
        type=int,
        default=0,
        help="steps captured per Metal trace window (0 = whole block = one replay/block)",
    )
    ap.add_argument("--num-blocks", type=int, default=4, help="denoise-only bench blocks per K (block 0 dropped)")
    ap.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument(
        "--commit-ms", type=float, default=None, help="additive batched-commit ms for the 100 t/s projection"
    )
    ap.add_argument("--measure-commit", action="store_true", help="time ONE real commit (mutates cache; runs last)")
    ap.add_argument("--trace-region-size", type=int, default=6000000000)
    args = ap.parse_args()
    steps_sweep = [int(x) for x in args.steps_sweep.split(",") if x.strip()] if args.steps_sweep else []
    group = args.group if args.group and args.group > 0 else None
    run(
        args.num_layers,
        args.canvas_length,
        args.steps,
        steps_sweep,
        group,
        args.num_blocks,
        args.prompt,
        args.max_seq_len,
        args.mode,
        args.commit_ms,
        args.measure_commit,
        args.trace_region_size,
    )


if __name__ == "__main__":
    main()
