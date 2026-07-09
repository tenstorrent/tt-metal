# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dg-08 lever 8 — traced data-dependent EARLY-HALT: correctness + overhead + break-even.

Loads the DiffusionGemma model ONCE and runs a set of fresh serving sessions (releasing each
config's traces before the next) to measure the traced early-halt loop against the fixed-budget
traced baseline. Three things, all on the SAME warmed traced harness (ENABLE_TRACY=OFF ⇒ Metal
capture/replay + ``time.perf_counter`` + ``ttnn.synchronize_device`` block timing, not tt-perf-report):

CORRECTNESS
  * Guard 1 (no-halt ≡ fixed-48): the scheme-A early-halt loop with an impossible threshold
    (never halts) commits the BYTE-IDENTICAL argmax of the fixed-budget traced path
    (``committed_sha`` match). The under-#48291 real-threshold (0.005) run also matches (halt
    never fires ⇒ full budget), a triple check.
  * Guard 2 (forced-halt ≡ eager at the same step): with an elevated threshold that DOES fire,
    scheme-A commits the byte-identical argmax the EAGER reference (``tt_denoise_block``,
    the same StableAndConfident rule on host) commits, at the same realized halt step
    (``committed_sha`` + ``denoise_steps_per_block`` + ``halted_per_block`` all match). A
    per-step device-scalar-vs-eager-record agreement table backs it.

OVERHEAD + BREAK-EVEN
  * fixed-48 steady block latency (the 17.92 t/s baseline) at two step counts ⇒ solve
    ``block = commit + steps · step_dev`` for ``step_dev`` and ``commit``.
  * scheme-A / scheme-B(K) no-halt steady block latency ⇒ per-step (A) / per-window (B)
    orchestration overhead = the host sync+read+branch cost the traced fixed path does not pay.
  * break-even halt-step: the step count below which A / B beat fixed-48.
  * the REALIZED halt-step distribution under the real 0.005 threshold (honest: #48291 keeps it
    at the full budget).

    DG_SPARSE_MOE=1 DG_DEDUP_ARGMAX=1 DG_SPARSE_MOE_TUNED=1 DG_TRACE_REGION_SIZE=10737418240 \
      DG_CKPT=... python -u -m models.experimental.diffusion_gemma.doc.optimize_perf.probe_early_halt \
        --mode correctness --num-layers 6 --max-denoise-steps 12
    (perf) --mode perf  (full 30L, --num-layers unset, --max-denoise-steps 48)

Markers: RESULT_EARLY_HALT <json>. *** DEVICE-OWNERSHIP: run only when QB2 is free. ***
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import torch
from loguru import logger

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _log_mesh_dram,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.tt.generate import denoise_and_commit_block, tokenize_prompt
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession

_DG_FLAGS = (
    "DG_SPARSE_MOE",
    "DG_SPARSE_MOE_TUNED",
    "DG_DEDUP_ARGMAX",
    "DG_DENOISE_TRACED",
    "DG_DENOISE_TRACED_MULTISTEP",
    "DG_DENOISE_MULTISTEP_GROUP",
    "DG_DENOISE_DEVICE_LOOP",
    "DG_DENOISE_EARLY_HALT",
    "DG_DENOISE_EARLY_HALT_WINDOW",
    "DG_COMMIT_BATCHED",
)

# tuned-MoE stack shared by every config (matches sweep_at48 / the 17.92 t/s baseline).
STACK = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_SPARSE_MOE_TUNED": "1"}
TRACED_FIXED = {**STACK, "DG_DENOISE_TRACED": "1"}
EAGER = dict(STACK)  # no traced/device-loop/early-halt ⇒ eager tt_denoise_block (real halt)


def _early_halt_env(window: int) -> dict:
    return {**STACK, "DG_DENOISE_EARLY_HALT": "1", "DG_DENOISE_EARLY_HALT_WINDOW": str(window)}


def _apply_env(env: dict) -> None:
    for k in _DG_FLAGS:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v


def _release_controllers(session) -> None:
    fn = getattr(session, "_logits_fn", None)
    for attr in (
        "_traced_denoise_controller",
        "_traced_denoise_multistep_controller",
        "_traced_early_halt_controller",
    ):
        controller = getattr(fn, attr, None)
        if controller is not None:
            try:
                controller.release()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"[early_halt] {attr}.release failed: {exc}")
            try:
                delattr(fn, attr)
            except Exception:  # noqa: BLE001
                pass


def _make_config(args, steps: int, threshold: float) -> DiffusionConfig:
    return DiffusionConfig(
        canvas_length=args.canvas_length,
        max_denoise_steps=steps,
        entropy_stop_threshold=threshold,
    )


def _committed_sha(tokens: torch.Tensor) -> str:
    return hashlib.sha256(tokens.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]


def run_config(bundle, mesh_device, prompt_tokens, spec, args) -> dict:
    """Fresh session, prefill + `blocks` decode blocks, steady = mean(block[1:])."""
    _apply_env(spec["env"])
    steps = spec["steps"]
    blocks = spec["blocks"]
    threshold = spec["threshold"]
    config = _make_config(args, steps, threshold)
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode="argmax",
        seed=args.seed,
        stop_token_ids=[],  # disable EOS halt (RUN-first degenerate output)
    )
    halt_traces = []
    try:
        t0 = time.perf_counter()
        session.prefill(prompt_tokens)
        emissions = [session.decode_block()]
        ttft_s = time.perf_counter() - t0
        halt_traces.append(_grab_halt_trace(session))
        for _ in range(1, blocks):
            emissions.append(session.decode_block())
            halt_traces.append(_grab_halt_trace(session))

        block_latencies = [e.latency_s for e in emissions]
        steady = block_latencies[1:] if len(block_latencies) > 1 else block_latencies
        mean_block = sum(steady) / len(steady)
        tps = args.canvas_length / mean_block if mean_block > 0 else 0.0
        committed = torch.cat([e.tokens for e in emissions], dim=1)
        result = {
            "label": spec["label"],
            "env": spec["env"],
            "steps": steps,
            "blocks": blocks,
            "threshold": threshold,
            "ttft_s": ttft_s,
            "per_block_latency_s": block_latencies,
            "steady_block_latency_s": mean_block,
            "tokens_per_block_per_s": tps,
            "denoise_steps_per_block": [e.num_denoise_steps for e in emissions],
            "halted_per_block": [e.halted for e in emissions],
            "committed_sha": _committed_sha(committed),
            "halt_trace_per_block": halt_traces,  # [(w_end_steps, mean_entropy, mismatch), ...] for early-halt
        }
    finally:
        _release_controllers(session)
        session.reset()
    logger.info(
        f"[early_halt] {spec['label']}: {tps:.2f} t/s block={mean_block:.3f}s "
        f"steps={result['denoise_steps_per_block']} halted={result['halted_per_block']}"
    )
    print("SWEEP_RESULT " + json.dumps(result))
    return result


def _grab_halt_trace(session):
    fn = getattr(session, "_logits_fn", None)
    controller = getattr(fn, "_traced_early_halt_controller", None)
    if controller is None:
        return None
    return [list(x) for x in getattr(controller, "last_halt_trace", [])]


def eager_block_records(bundle, mesh_device, prompt_tokens, args, steps: int, threshold: float) -> dict:
    """Run ONE eager (tt_denoise_block) block keeping per-step records — the halt oracle.

    Returns per-step entropy_mean + argmax_change_count for step-scalar agreement vs the
    device halt trace, plus the block's halt step/committed for the direct eager comparison.
    """
    _apply_env(EAGER)
    config = _make_config(args, steps, threshold)
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode="argmax",
        seed=args.seed,
        stop_token_ids=[],
    )
    try:
        session.prefill(prompt_tokens)
        start_pos = session.next_pos
        block_idx = session.block_idx
        gumbel = session._gumbel_noise_fn(block_idx) if session._gumbel_noise_fn else None
        noise = session._noise_tokens_fn(block_idx) if session._noise_tokens_fn else None
        init_canvas = session._init_canvas_fn(block_idx, start_pos)
        block = denoise_and_commit_block(
            session.tt_model,
            session._logits_fn,
            init_canvas,
            config,
            start_pos=start_pos,
            gumbel_noise_fn=gumbel,
            noise_tokens_fn=noise,
        )
        traj = block.trajectory
        per_step = getattr(traj, "per_step", None) or getattr(traj, "records", [])
        ents = [float(r.entropy_mean) for r in per_step]
        changes = []
        prev = None
        for r in per_step:
            am = r.argmax
            changes.append(None if prev is None else int((am != prev).sum().item()))
            prev = am
        return {
            "num_steps": int(traj.num_steps),
            "halted": bool(traj.halted),
            "committed_sha": _committed_sha(block.committed),
            "entropy_mean_per_step": [round(e, 6) for e in ents],
            "argmax_changes_per_step": changes,
        }
    finally:
        session.reset()


def build_configs(args) -> list:
    N = args.max_denoise_steps
    K_list = [int(k) for k in args.window_sweep.split(",") if k.strip()]
    THR = 0.005  # released entropy_stop_threshold
    NEVER = -1.0e9  # impossible threshold ⇒ scheme runs the full budget
    cfgs = []
    if args.mode in ("perf", "all"):
        # Two fixed-48 points to solve step_dev + commit; the baseline is the s=N point.
        cfgs.append({"label": f"fixed_traced_s{N}", "env": TRACED_FIXED, "steps": N, "blocks": 3, "threshold": THR})
        n2 = max(2, N // 4)
        cfgs.append({"label": f"fixed_traced_s{n2}", "env": TRACED_FIXED, "steps": n2, "blocks": 3, "threshold": THR})
        # scheme-A: no-halt (pure overhead) + real-threshold (realized distribution under #48291).
        cfgs.append({"label": "earlyhaltA_nohalt", "env": _early_halt_env(1), "steps": N, "blocks": 3, "threshold": NEVER})
        cfgs.append({"label": "earlyhaltA_real", "env": _early_halt_env(1), "steps": N, "blocks": 3, "threshold": THR})
        # scheme-B: per-window overhead at each K (no-halt so the full budget runs => amortized o_B/K).
        for k in K_list:
            if k <= 1:
                continue
            cfgs.append(
                {"label": f"earlyhaltB_k{k}_nohalt", "env": _early_halt_env(k), "steps": N, "blocks": 3, "threshold": NEVER}
            )
    if args.mode in ("correctness", "all"):
        # Guard 1: no-halt scheme-A must equal the fixed budget commit (byte-identical).
        cfgs.append({"label": "fixedC_traced", "env": TRACED_FIXED, "steps": N, "blocks": 2, "threshold": THR})
        cfgs.append({"label": "earlyhaltA_nohaltC", "env": _early_halt_env(1), "steps": N, "blocks": 2, "threshold": NEVER})
        # Guard 2: forced elevated threshold — eager vs scheme-A must agree (sha + steps + halted).
        cfgs.append({"label": "eager_forced", "env": EAGER, "steps": N, "blocks": 2, "threshold": args.forced_threshold})
        cfgs.append(
            {"label": "earlyhaltA_forced", "env": _early_halt_env(1), "steps": N, "blocks": 2, "threshold": args.forced_threshold}
        )
        for k in K_list:
            if k <= 1:
                continue
            cfgs.append(
                {"label": f"earlyhaltB_k{k}_forced", "env": _early_halt_env(k), "steps": N, "blocks": 2, "threshold": args.forced_threshold}
            )
    return cfgs


def _by_label(results):
    return {r["label"]: r for r in results}


def analyze(results, args, eager_diag) -> dict:
    R = _by_label(results)
    N = args.max_denoise_steps
    out = {"guards": {}, "overhead": {}, "break_even": {}, "realized_distribution": {}}

    # -------- Guard 1: no-halt scheme-A ≡ fixed budget (byte-identical commit) --------
    for fixed_key, a_key in (("fixedC_traced", "earlyhaltA_nohaltC"), (f"fixed_traced_s{N}", "earlyhaltA_nohalt")):
        if fixed_key in R and a_key in R:
            match = R[fixed_key]["committed_sha"] == R[a_key]["committed_sha"]
            out["guards"][f"guard1_{a_key}_eq_{fixed_key}"] = {
                "sha_match": bool(match),
                "fixed_sha": R[fixed_key]["committed_sha"],
                "a_sha": R[a_key]["committed_sha"],
            }
    # scheme-A real-threshold also equals fixed (halt never fires under #48291).
    if f"fixed_traced_s{N}" in R and "earlyhaltA_real" in R:
        out["guards"]["guard1_real_eq_fixed"] = {
            "sha_match": bool(R[f"fixed_traced_s{N}"]["committed_sha"] == R["earlyhaltA_real"]["committed_sha"]),
        }

    # -------- Guard 2: forced-halt scheme-A ≡ eager (sha + steps + halted) --------
    if "eager_forced" in R and "earlyhaltA_forced" in R:
        e, a = R["eager_forced"], R["earlyhaltA_forced"]
        out["guards"]["guard2_A_eq_eager"] = {
            "sha_match": bool(e["committed_sha"] == a["committed_sha"]),
            "steps_match": bool(e["denoise_steps_per_block"] == a["denoise_steps_per_block"]),
            "halted_match": bool(e["halted_per_block"] == a["halted_per_block"]),
            "eager_steps": e["denoise_steps_per_block"],
            "a_steps": a["denoise_steps_per_block"],
            "eager_halted": e["halted_per_block"],
            "a_halted": a["halted_per_block"],
        }
    # scheme-B forced: commit + steps agreement with eager (correct under convergence-stability).
    for k in [int(x) for x in args.window_sweep.split(",") if x.strip() and int(x) > 1]:
        bkey = f"earlyhaltB_k{k}_forced"
        if "eager_forced" in R and bkey in R:
            e, b = R["eager_forced"], R[bkey]
            out["guards"][f"guard2_Bk{k}_vs_eager"] = {
                "sha_match": bool(e["committed_sha"] == b["committed_sha"]),
                "eager_steps": e["denoise_steps_per_block"],
                "b_steps": b["denoise_steps_per_block"],
                "b_halted": b["halted_per_block"],
            }

    # per-step device-scalar vs eager-record agreement (scheme-A forced, block that halts or block 0).
    if eager_diag is not None and "earlyhaltA_forced" in R:
        a = R["earlyhaltA_forced"]
        # compare block-0's device halt trace vs the eager per-step records (same seed/prompt).
        dev_trace = a["halt_trace_per_block"][0] if a["halt_trace_per_block"] else None
        if dev_trace:
            ent_err = []
            mism_err = []
            for (w_end, mean_ent, mismatch) in dev_trace:
                s = int(w_end) - 1  # scheme-A: w_end == step+1
                if 0 <= s < len(eager_diag["entropy_mean_per_step"]):
                    ent_err.append(abs(mean_ent - eager_diag["entropy_mean_per_step"][s]))
                    ch = eager_diag["argmax_changes_per_step"][s]
                    if ch is not None:
                        mism_err.append(abs(int(mismatch) - int(ch)))
            out["guards"]["guard2_per_step_scalar_agreement"] = {
                "max_abs_entropy_err": max(ent_err) if ent_err else None,
                "max_abs_mismatch_err": max(mism_err) if mism_err else None,
                "n_steps_compared": len(ent_err),
                "eager_num_steps": eager_diag["num_steps"],
                "eager_halted": eager_diag["halted"],
            }

    # -------- Overhead + break-even (perf mode) --------
    fkey, n2key = f"fixed_traced_s{N}", f"fixed_traced_s{max(2, N // 4)}"
    if fkey in R and n2key in R:
        bN, n_hi = R[fkey]["steady_block_latency_s"], N
        b2, n_lo = R[n2key]["steady_block_latency_s"], R[n2key]["steps"]
        step_dev = (bN - b2) / (n_hi - n_lo) if n_hi != n_lo else float("nan")
        commit = bN - n_hi * step_dev
        out["overhead"]["fixed_step_dev_s"] = step_dev
        out["overhead"]["fixed_commit_s"] = commit
        out["overhead"]["fixed_block_sN_s"] = bN
        out["overhead"]["fixed_tps"] = args.canvas_length / bN if bN > 0 else 0.0

        if "earlyhaltA_nohalt" in R:
            bA = R["earlyhaltA_nohalt"]["steady_block_latency_s"]
            o_A = (bA - bN) / N  # extra per-step host sync+read+branch cost
            out["overhead"]["schemeA_block_nohalt_s"] = bA
            out["overhead"]["schemeA_per_step_overhead_s"] = o_A
            # break-even: H·(step_dev + o_A) + commit < N·step_dev + commit
            if step_dev + o_A > 0:
                out["break_even"]["schemeA_break_even_steps"] = N * step_dev / (step_dev + o_A)

        for k in [int(x) for x in args.window_sweep.split(",") if x.strip() and int(x) > 1]:
            bkey = f"earlyhaltB_k{k}_nohalt"
            if bkey in R:
                bB = R[bkey]["steady_block_latency_s"]
                n_windows = (N + k - 1) // k
                o_Bwin = (bB - bN) / n_windows  # per-window host overhead
                out["overhead"][f"schemeB_k{k}_block_nohalt_s"] = bB
                out["overhead"][f"schemeB_k{k}_per_window_overhead_s"] = o_Bwin
                denom = step_dev + o_Bwin / k
                if denom > 0:
                    out["break_even"][f"schemeB_k{k}_break_even_steps"] = N * step_dev / denom

    # -------- Realized halt-step distribution (real threshold, honest) --------
    if "earlyhaltA_real" in R:
        r = R["earlyhaltA_real"]
        out["realized_distribution"]["schemeA_real_threshold"] = {
            "denoise_steps_per_block": r["denoise_steps_per_block"],
            "halted_per_block": r["halted_per_block"],
            "note": "under #48291 the entropy gate never clears 0.005 => full budget (no early halt)",
        }
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    p.add_argument("--mesh", default="P150x4")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    p.add_argument("--canvas-length", type=int, default=256)
    p.add_argument("--max-denoise-steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["correctness", "perf", "all"], default="all")
    p.add_argument("--window-sweep", default="4,8", help="comma-separated scheme-B window sizes")
    p.add_argument("--forced-threshold", type=float, default=100.0, help="elevated threshold that forces halt (Guard 2)")
    p.add_argument("--out", default=None)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    mesh_device = _open_mesh_device(args.mesh)
    out = {"config": {k: v for k, v in vars(args).items()}, "results": [], "analysis": {}}
    try:
        _log_mesh_dram(mesh_device, "baseline")
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        t_load = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(mesh_device, args.checkpoint, **model_kwargs)
        logger.info(f"[early_halt] model load took {time.perf_counter() - t_load:.1f}s")
        _log_mesh_dram(mesh_device, "post-build")
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        logger.info(f"[early_halt] prompt_len={int(prompt_tokens.shape[1])}")

        eager_diag = None
        if args.mode in ("correctness", "all"):
            eager_diag = eager_block_records(
                bundle, mesh_device, prompt_tokens, args, args.max_denoise_steps, args.forced_threshold
            )
            logger.info(f"[early_halt] eager forced-halt block: steps={eager_diag['num_steps']} halted={eager_diag['halted']}")
            print("EAGER_DIAG " + json.dumps(eager_diag))

        for spec in build_configs(args):
            try:
                out["results"].append(run_config(bundle, mesh_device, prompt_tokens, spec, args))
            except BaseException as exc:  # noqa: BLE001
                logger.error(f"DG_EARLY_HALT_CONFIG_FAILURE label={spec['label']} err={type(exc).__name__}: {exc}")
                raise
        out["analysis"] = analyze(out["results"], args, eager_diag)
    finally:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        _close_mesh_device(mesh_device)
    print("RESULT_EARLY_HALT " + json.dumps(out["analysis"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
