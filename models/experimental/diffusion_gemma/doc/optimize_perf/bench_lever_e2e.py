# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dg-08 L1-residency pass — generalized TRACED end-to-end lever comparison.

Loads the full 30L model ONCE, then runs a list of (label, extra-env) configs at each requested
budget, reporting traced steady-state tok/s + committed_sha (must match baseline for a placement /
per-row-identical lever -> quality-safe). Generalizes bench_moe_l1_e2e.py to any DG_* lever flag.

Run (device-free window):
  DG_TRACE_REGION_SIZE=10737418240 \
    DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
    python models/experimental/diffusion_gemma/doc/optimize_perf/bench_lever_e2e.py \
      --levers baseline,norm --budgets 48,12 --blocks 3

Levers (extra env on top of the traced-tuned base):
  baseline -> {}                               (current default path)
  norm     -> DG_NORM_FULLCANVAS=1             (HIGH-4)
  moe      -> DG_MOE_L1=both                   (HIGH-1+HIGH-2)
  norm_moe -> DG_NORM_FULLCANVAS=1 DG_MOE_L1=both
  selfcond_l1_off -> DG_SELFCOND_LOGITS_L1=off (diagnostic control)
  selfcond_l1 -> DG_SELFCOND_LOGITS_L1=chain   (selected default, explicit)

Marker: E2E_RESULT <json> per (budget, lever).
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
from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _log_mesh_dram, _open_mesh_device
from models.experimental.diffusion_gemma.demo.serving_smoke import _DeviceGenLike
from models.experimental.diffusion_gemma.tt.self_conditioning import (
    self_conditioning_embedding_prechunk_enabled,
    self_conditioning_logits_l1_mode,
)
from models.experimental.diffusion_gemma.tt.generate import decode_generation, tokenize_prompt
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.experimental.diffusion_gemma.doc.optimize_perf.sweep_serving import _release_controller

BASE_ENV = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_SPARSE_MOE_TUNED": "1", "DG_DENOISE_TRACED": "1"}
EXPECTED_TRACE_REGION_SIZE = 10 * 1024**3
CANONICAL_PROMPT = "Explain what a diffusion language model is in one sentence."
LEVER_ENV = {
    "baseline": {},
    "norm": {"DG_NORM_FULLCANVAS": "1"},
    "moe": {"DG_MOE_L1": "both"},
    "norm_moe": {"DG_NORM_FULLCANVAS": "1", "DG_MOE_L1": "both"},
    "selfcond_l1_off": {"DG_SELFCOND_LOGITS_L1": "off"},
    "selfcond_l1": {"DG_SELFCOND_LOGITS_L1": "chain"},
}
# flags we toggle across levers — cleared before applying each lever so configs don't leak
_TOGGLE = ("DG_NORM_FULLCANVAS", "DG_MOE_L1", "DG_SELFCOND_LOGITS_L1")


def _required_trace_region_size() -> int:
    raw = os.environ.get("DG_TRACE_REGION_SIZE")
    if raw is None:
        raise RuntimeError("DG_TRACE_REGION_SIZE must be set before launching this traced benchmark")
    try:
        size = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"DG_TRACE_REGION_SIZE must be an integer, got {raw!r}") from exc
    if size != EXPECTED_TRACE_REGION_SIZE:
        raise RuntimeError(
            "DG_TRACE_REGION_SIZE must be exactly "
            f"{EXPECTED_TRACE_REGION_SIZE} bytes for comparable dg-08 runs, got {size}"
        )
    return size


def _validate_canonical_workload(args, budgets) -> None:
    expected = {
        "mesh": (args.mesh, "P150x4"),
        "max_seq_len": (args.max_seq_len, 1024),
        "canvas_length": (args.canvas_length, 256),
        "seed": (args.seed, 0),
        "blocks": (args.blocks, 3),
        "prompt": (args.prompt, CANONICAL_PROMPT),
    }
    mismatches = [
        f"{name}={actual!r} (expected {wanted!r})" for name, (actual, wanted) in expected.items() if actual != wanted
    ]
    if mismatches:
        raise RuntimeError("non-canonical dg-08 benchmark workload: " + ", ".join(mismatches))
    if not budgets or any(budget not in (12, 48) for budget in budgets):
        raise RuntimeError(f"dg-08 benchmark budgets must be a non-empty subset of (12, 48), got {budgets}")


def run_one(bundle, prompt_tokens, lever, steps, blocks, canvas_length, seed):
    for k, v in BASE_ENV.items():
        os.environ[k] = v
    for k in _TOGGLE:
        os.environ.pop(k, None)
    for k, v in LEVER_ENV[lever].items():
        os.environ[k] = v
    config = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode="argmax",
        seed=seed,
        stop_token_ids=[],
    )
    try:
        t0 = time.perf_counter()
        session.prefill(prompt_tokens)
        prefill_s = time.perf_counter() - t0
        first = session.decode_block()
        ttft_s = time.perf_counter() - t0
        emissions = [first]
        for _ in range(1, blocks):
            emissions.append(session.decode_block())
        full_generation_s = time.perf_counter() - t0
        lat = [e.latency_s for e in emissions]
        steady = lat[1:] if len(lat) > 1 else lat
        mean_block = sum(steady) / len(steady)
        tps = canvas_length / mean_block if mean_block > 0 else 0.0
        committed = torch.cat([e.tokens for e in emissions], dim=1)
        sha = hashlib.sha256(committed.to(torch.int64).cpu().numpy().tobytes()).hexdigest()[:16]
        text = decode_generation(
            bundle.tokenizer,
            prompt_tokens,
            _DeviceGenLike(committed, session.cache_len, session.next_pos),
            skip_prompt=True,
            skip_special_tokens=True,
        )
        text_str = (text[0] if text else "")[:160]
        result = {
            "lever": lever,
            "env": LEVER_ENV[lever],
            "base_env": BASE_ENV,
            "resolved_selfcond_prechunk": self_conditioning_embedding_prechunk_enabled(),
            "DG_SELFCOND_PRECHUNK_EMBED": os.environ.get("DG_SELFCOND_PRECHUNK_EMBED", "<unset>"),
            "DG_SELFCOND_LOGITS_L1": os.environ.get("DG_SELFCOND_LOGITS_L1", "<unset>"),
            "resolved_selfcond_logits_l1": self_conditioning_logits_l1_mode(),
            "DG_TRACE_REGION_SIZE": os.environ["DG_TRACE_REGION_SIZE"],
            "trace_region_size_bytes": _required_trace_region_size(),
            "steps": steps,
            "blocks": blocks,
            "prefill_s": round(prefill_s, 4),
            "ttft_s": round(ttft_s, 3),
            "per_block_latency_s": [round(x, 4) for x in lat],
            "sum_block_latency_s": round(sum(lat), 4),
            "full_generation_s": round(full_generation_s, 4),
            "steady_block_latency_s": round(mean_block, 4),
            "tokens_per_block_per_s": round(tps, 3),
            "denoise_steps_per_block": [e.num_denoise_steps for e in emissions],
            "committed_sha": sha,
            "text_head": text_str,
        }
    finally:
        _release_controller(session)
        session.reset()
    logger.info(f"[e2e] {lever} @{steps}: {tps:.3f} t/s block={mean_block:.4f}s sha={result['committed_sha']}")
    print("E2E_RESULT " + json.dumps(result), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    ap.add_argument("--mesh", default="P150x4")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--prompt", default=CANONICAL_PROMPT)
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--levers", default="baseline,norm")
    ap.add_argument("--budgets", default="48,12")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    levers = args.levers.split(",")
    budgets = [int(x) for x in args.budgets.split(",")]
    _validate_canonical_workload(args, budgets)
    unknown_levers = sorted(set(levers) - set(LEVER_ENV))
    if unknown_levers:
        raise RuntimeError(f"unknown levers: {unknown_levers}")
    for k, v in BASE_ENV.items():
        os.environ[k] = v
    for k in _TOGGLE:
        os.environ.pop(k, None)
    _required_trace_region_size()
    mesh = _open_mesh_device(args.mesh)
    results = []
    try:
        _log_mesh_dram(mesh, "baseline")
        t_load = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(
            mesh, args.checkpoint, max_seq_len=args.max_seq_len, create_kv_cache=True
        )
        logger.info(f"[e2e] model load {time.perf_counter() - t_load:.1f}s")
        _log_mesh_dram(mesh, "post-build")
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        for steps in budgets:
            for lever in levers:
                try:
                    results.append(
                        run_one(bundle, prompt_tokens, lever, steps, args.blocks, args.canvas_length, args.seed)
                    )
                except BaseException as exc:  # noqa: BLE001
                    logger.error(f"E2E_CONFIG_FAILURE lever={lever} steps={steps} {type(exc).__name__}: {exc}")
                    raise
    finally:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        _close_mesh_device(mesh)
    print("E2E_DONE n=" + str(len(results)), flush=True)
    for r in results:
        print(
            f"  @{r['steps']} {r['lever']}: {r['tokens_per_block_per_s']:.3f} t/s block={r['steady_block_latency_s']:.4f}s sha={r['committed_sha']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
