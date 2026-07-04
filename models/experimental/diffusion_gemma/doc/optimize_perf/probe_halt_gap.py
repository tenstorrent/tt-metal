# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Quantify WHY the adaptive early-halt is a no-op under #48291 (task-3 evidence).

The HF StableAndConfidentStoppingCriteria halts a denoise block when BOTH:
  * stable    — the clean-argmax canvas is unchanged for ``stable_steps_to_halt`` (=1) steps, AND
  * confident — the mean per-position entropy of the temperature-scaled logits is below
                ``entropy_stop_threshold`` (=0.005 nats).

Under #48291 the bf16/MoE/TP=4 backbone produces degenerate decisions, so this probe records the
ACTUAL per-step ``entropy_mean`` and argmax-change count over a full K=48 eager block on the real
30L model, and reports how far each gate sits from firing. This turns the "early-halt is a no-op"
claim into measured evidence: it shows whether the blocker is the confidence gate (entropy nowhere
near 0.005 => a property of the backbone logit DISTRIBUTION, which a higher-precision *terminal*
argmax/entropy re-measures but cannot make more confident) and/or the stability gate (argmax flips
every step). Both gates are downstream of the shared-gemma4 bf16-MoE precision ceiling.

EAGER path only (real StableAndConfident halt, `tt_denoise_block`); no trace region needed.

    DG_SPARSE_MOE=1 DG_DEDUP_ARGMAX=1 DG_SPARSE_MOE_TUNED=1 DG_CKPT=... \
      python -u models/experimental/diffusion_gemma/doc/optimize_perf/probe_halt_gap.py --blocks 2

Markers: RESULT_HALT_GAP <json>. *** DEVICE-OWNERSHIP: run only when QB2 is free. ***
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

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


def _run_one_block_eager(session, config):
    """Replicate serving.decode_block's eager denoise but KEEP the trajectory records."""
    start_pos = session.next_pos
    block_idx = session.block_idx
    gumbel_for_block = session._gumbel_noise_fn(block_idx) if session._gumbel_noise_fn else None
    noise_for_block = session._noise_tokens_fn(block_idx) if session._noise_tokens_fn else None
    init_canvas = session._init_canvas_fn(block_idx, start_pos)
    t0 = time.perf_counter()
    block = denoise_and_commit_block(
        session.tt_model,
        session._logits_fn,
        init_canvas,
        config,
        start_pos=start_pos,
        gumbel_noise_fn=gumbel_for_block,
        noise_tokens_fn=noise_for_block,
        page_table=session.page_table,
        page_tables_per_layer=session.page_tables_per_layer,
    )
    latency_s = time.perf_counter() - t0
    session.next_pos = block.next_pos
    session.block_idx += 1
    return block.trajectory, latency_s


def analyze_trajectory(traj, threshold: float):
    per_step = getattr(traj, "per_step", None)
    if per_step is None:
        per_step = getattr(traj, "records", [])
    ents = [float(r.entropy_mean) for r in per_step]
    accepts = [int(r.num_accepted) for r in per_step]
    # argmax-change count vs previous step (stability gate needs 0 changes for n_stable steps)
    changes = []
    prev = None
    for r in per_step:
        am = r.argmax
        if prev is not None:
            changes.append(int((am != prev).sum().item()))
        prev = am
    confident_steps = sum(1 for e in ents if e < threshold)
    stable_steps = sum(1 for c in changes if c == 0)
    # would-halt: a step whose argmax matches the previous AND entropy < threshold
    would_halt = any((changes[i - 1] == 0) and (ents[i] < threshold) for i in range(1, len(ents)))
    return {
        "num_steps": int(traj.num_steps),
        "halted": bool(traj.halted),
        "entropy_mean_min": min(ents) if ents else None,
        "entropy_mean_median": statistics.median(ents) if ents else None,
        "entropy_mean_max": max(ents) if ents else None,
        "entropy_mean_last": ents[-1] if ents else None,
        "entropy_stop_threshold": threshold,
        "steps_confident_below_threshold": confident_steps,
        "steps_stable_argmax_unchanged": stable_steps,
        "argmax_changes_per_step": changes,
        "entropy_mean_per_step": [round(e, 5) for e in ents],
        "num_accepted_per_step": accepts,
        "would_early_halt": would_halt,
    }


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    p.add_argument("--mesh", default="P150x4")
    p.add_argument("--num-layers", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    p.add_argument("--canvas-length", type=int, default=256)
    p.add_argument("--max-denoise-steps", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--out", default=None)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    mesh_device = _open_mesh_device(args.mesh)
    out = {"config": vars(args), "blocks": []}
    try:
        _log_mesh_dram(mesh_device, "baseline")
        model_kwargs = {"max_seq_len": args.max_seq_len, "create_kv_cache": True}
        if args.num_layers is not None:
            model_kwargs["num_layers"] = args.num_layers
        bundle = build_tt_model_from_checkpoint_dir(mesh_device, args.checkpoint, **model_kwargs)
        prompt_tokens = tokenize_prompt(bundle.tokenizer, args.prompt)
        config = DiffusionConfig(canvas_length=args.canvas_length, max_denoise_steps=args.max_denoise_steps)
        session = BlockDiffusionServingSession(
            bundle.tt_model,
            bundle.state_dict,
            config=config,
            tokenizer=bundle.tokenizer,
            gumbel_mode="argmax",
            seed=args.seed,
            stop_token_ids=[],
        )
        session.prefill(prompt_tokens)
        for b in range(args.blocks):
            traj, latency_s = _run_one_block_eager(session, config)
            info = analyze_trajectory(traj, config.entropy_stop_threshold)
            info["block_idx"] = b
            info["latency_s"] = latency_s
            out["blocks"].append(info)
            logger.info(
                f"[halt-gap] block {b}: num_steps={info['num_steps']} halted={info['halted']} "
                f"entropy_min={info['entropy_mean_min']:.4f} entropy_last={info['entropy_mean_last']:.4f} "
                f"confident_steps={info['steps_confident_below_threshold']} "
                f"stable_steps={info['steps_stable_argmax_unchanged']} would_halt={info['would_early_halt']}"
            )
    finally:
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
        _close_mesh_device(mesh_device)
    print("RESULT_HALT_GAP " + json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
