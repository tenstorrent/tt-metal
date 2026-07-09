# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""dg-08 L1-residency pass — TRACED end-to-end confirmation of the MoE L1 levers.

The isolated MoE micro-bench (bench_moe_l1_residency.py) showed DG_MOE_L1=both saves ~3.2% of the
MoE forward with bit-identical output. But the micro-bench is EAGER and isolated (the 23 MB gather
DRAM write sits on the critical path); under a captured Metal trace that write may overlap adjacent
ops and the win could shrink or vanish (the Permute "97%->0%" trap). This driver settles it on the
ranking metric the deliverable requires: **traced** steady-state tok/s.

Loads the full 30L model ONCE, then runs, per budget, DG_MOE_L1=off vs both (fresh traced session
each, controller released between so only one trace occupies the region). Same seed/prompt/canvas, so
the accept schedule is pinned: ``committed_sha`` MUST match off<->both (quality-safe, placement-only).

Run (device-free window; ~20 min model load + a few min decode):
  DG_CKPT=/home/zni/dg_models/diffusiongemma-26B-A4B-it \
    python models/experimental/diffusion_gemma/doc/optimize_perf/bench_moe_l1_e2e.py

Marker: E2E_RESULT <json> per (budget, mode).
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
from models.experimental.diffusion_gemma.tt.generate import decode_generation, tokenize_prompt
from models.experimental.diffusion_gemma.tt.serving import BlockDiffusionServingSession
from models.experimental.diffusion_gemma.doc.optimize_perf.sweep_serving import _release_controller

BASE_ENV = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_SPARSE_MOE_TUNED": "1", "DG_DENOISE_TRACED": "1"}


def run_one(bundle, prompt_tokens, mode, steps, blocks, canvas_length, seed):
    for k, v in BASE_ENV.items():
        os.environ[k] = v
    os.environ["DG_MOE_L1"] = mode  # read live inside sparse_experts_forward at trace-capture time
    config = DiffusionConfig(canvas_length=canvas_length, max_denoise_steps=steps)
    session = BlockDiffusionServingSession(
        bundle.tt_model,
        bundle.state_dict,
        config=config,
        tokenizer=bundle.tokenizer,
        gumbel_mode="argmax",
        seed=seed,
        stop_token_ids=[],  # disable EOS halt (RUN-first)
    )
    try:
        t0 = time.perf_counter()
        session.prefill(prompt_tokens)
        first = session.decode_block()
        ttft_s = time.perf_counter() - t0
        emissions = [first]
        for _ in range(1, blocks):
            emissions.append(session.decode_block())
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
            "mode": mode,
            "steps": steps,
            "blocks": blocks,
            "ttft_s": round(ttft_s, 3),
            "per_block_latency_s": [round(x, 4) for x in lat],
            "steady_block_latency_s": round(mean_block, 4),
            "tokens_per_block_per_s": round(tps, 3),
            "denoise_steps_per_block": [e.num_denoise_steps for e in emissions],
            "committed_sha": sha,
            "text_head": text_str,
        }
    finally:
        _release_controller(session)
        session.reset()
    logger.info(f"[e2e] mode={mode} steps={steps}: {tps:.3f} t/s block={mean_block:.4f}s sha={result['committed_sha']}")
    print("E2E_RESULT " + json.dumps(result), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    ap.add_argument("--mesh", default="P150x4")
    ap.add_argument("--max-seq-len", type=int, default=1024)
    ap.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="48,12", help="comma-separated step budgets")
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    budgets = [int(x) for x in args.budgets.split(",")]
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
            for mode in ("off", "both"):
                try:
                    results.append(
                        run_one(bundle, prompt_tokens, mode, steps, args.blocks, args.canvas_length, args.seed)
                    )
                except BaseException as exc:  # noqa: BLE001
                    logger.error(f"E2E_CONFIG_FAILURE mode={mode} steps={steps} {type(exc).__name__}: {exc}")
                    raise
    finally:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        _close_mesh_device(mesh)
    print("E2E_DONE n=" + str(len(results)), flush=True)
    for r in results:
        print(
            f"  steps={r['steps']} mode={r['mode']}: {r['tokens_per_block_per_s']:.3f} t/s block={r['steady_block_latency_s']:.4f}s sha={r['committed_sha']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
