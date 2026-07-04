# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Model-faithful @48 throughput measurement (denoise step count == HF reference).

METHODOLOGY: the denoise step count is a MODEL parameter, not a free perf knob. It must
equal the reference. The HF reference is ADAPTIVE (StableAndConfidentStoppingCriteria, cap
``max_denoising_steps=48``). Under #48291 the argmax decisions are degenerate, so the
stable+confident early-halt NEVER fires => the model runs the FULL 48 steps. Therefore the
only model-faithful throughput is measured at the model's real adaptive step count = 48.
Fewer-step numbers (@24/@12/@6/@4) do NOT count as the model's t/s.

This driver loads the full 30L model ONCE and runs a small set of configs in RISK ORDER
(safe first, the memory-risky whole-block multi-step LAST), writing each config's result to
its OWN JSON file the instant it is computed. A trace-capture FATAL poisons the device and can
hard-abort the process, so per-config disk writes guarantee the headline result survives a
later config's crash.

Configs (in run order):
  1. traced_tuned_s12   — ANCHOR: reproduces the session-8/9 verified 58 t/s @12 in this harness
                          so the @48 point inherits that trust.
  2. eager_adaptive_s48 — the EAGER adaptive path (real StableAndConfident early-halt) at K=48.
                          Reports the ACTUAL num_denoise_steps + halted per block => confirms
                          early-halt is a no-op (runs ~48, halted=False) so the fixed-48 traced
                          number IS model-faithful. Also the eager @48 baseline.
  3. traced_tuned_s48   — HEADLINE: the single-step traced serving denoise loop at the full
                          model-faithful K=48. This is the LEGITIMATE @48 t/s.
  4. multistep_g12_s48  — task-2 lever (bit-exact): multi-step trace batching, window G=12
                          (ceil(48/12)=4 replays/block instead of 48). committed_sha MUST match
                          traced_tuned_s48 (bit-exact). Bounded window => safer memory than whole-block.
  5. multistep_wb_s48   — task-2 lever (max dispatch-flatten): whole-block window (1 replay/block).
                          RISKIEST memory; runs LAST.

Requires a large trace region: 48 single-step traces at 30L ~= 8 GB, so run with
``DG_TRACE_REGION_SIZE=10737418240`` (10 GB). Flags per config are applied live to os.environ.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from loguru import logger

from models.experimental.diffusion_gemma.checkpoint import build_tt_model_from_checkpoint_dir
from models.experimental.diffusion_gemma.demo.text_demo import (
    _close_mesh_device,
    _log_mesh_dram,
    _open_mesh_device,
)
from models.experimental.diffusion_gemma.tt.generate import tokenize_prompt

# Reuse the tested per-config runner from sweep_serving (fresh session, steady=mean(blocks[1:]),
# committed_sha, controller.release()).
from models.experimental.diffusion_gemma.doc.optimize_perf.sweep_serving import run_config


def at48_configs():
    STACK = {"DG_SPARSE_MOE": "1", "DG_DEDUP_ARGMAX": "1", "DG_SPARSE_MOE_TUNED": "1"}
    TRACED = {**STACK, "DG_DENOISE_TRACED": "1"}
    EAGER = dict(STACK)  # no traced / no device-loop => tt_denoise_block (real adaptive early-halt)
    MULTISTEP = {**STACK, "DG_DENOISE_TRACED_MULTISTEP": "1"}
    return [
        {"label": "traced_tuned_s12", "env": TRACED, "steps": 12, "blocks": 3},
        {"label": "eager_adaptive_s48", "env": EAGER, "steps": 48, "blocks": 2},
        {"label": "traced_tuned_s48", "env": TRACED, "steps": 48, "blocks": 3},
        {
            "label": "multistep_g12_s48",
            "env": {**MULTISTEP, "DG_DENOISE_MULTISTEP_GROUP": "12"},
            "steps": 48,
            "blocks": 3,
        },
        {"label": "multistep_wb_s48", "env": MULTISTEP, "steps": 48, "blocks": 3},
    ]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    p.add_argument("--mesh", default="P150x4")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    p.add_argument("--canvas-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default=None, help="comma-separated labels to run (default: all)")
    p.add_argument("--out-dir", default=os.environ.get("DG_AT48_OUT", "/tmp/dg_at48"))
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    configs = at48_configs()
    if args.only:
        want = set(args.only.split(","))
        configs = [c for c in configs if c["label"] in want]

    mesh_device = _open_mesh_device(args.mesh)
    results = []
    try:
        _log_mesh_dram(mesh_device, "baseline")
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        t_load = time.perf_counter()
        bundle = build_tt_model_from_checkpoint_dir(mesh_device, args.checkpoint, **model_kwargs)
        logger.info(f"[at48] model load took {time.perf_counter() - t_load:.1f}s")
        _log_mesh_dram(mesh_device, "post-build")
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        logger.info(f"[at48] prompt_len={int(prompt_tokens.shape[1])}")

        for cfg_spec in configs:
            try:
                r = run_config(bundle, mesh_device, prompt_tokens, cfg_spec, args)
            except BaseException as exc:  # noqa: BLE001
                logger.error(f"DG_AT48_CONFIG_FAILURE label={cfg_spec['label']} err={type(exc).__name__}: {exc}")
                raise
            # Write EACH result to its own file immediately (survive a later config's FATAL/abort).
            with open(os.path.join(args.out_dir, f"{cfg_spec['label']}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=2)
            results.append(r)
    finally:
        with open(os.path.join(args.out_dir, "combined.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        _close_mesh_device(mesh_device)
    print("DG_AT48_DONE configs=" + str(len(results)))
    for r in results:
        print(
            f"  {r['label']}: {r['tokens_per_block_per_s']:.2f} t/s  block={r['steady_block_latency_s']:.3f}s"
            f"  steps={r['denoise_steps_per_block']}  halted={r['halted_per_block']}  sha={r['committed_sha']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
