# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""bf16-vs-bfp8 MoE-experts diffusion-decision agreement harness (dg-07 datatype sweep, #47475).

The datatype-sweep accuracy metric is the DIFFUSION DECISION, not teacher-forcing top-1/top-5:
per-step Gumbel-max argmax agreement, entropy PCC, and end-to-end accept/renoise agreement over
an injected-noise reference trajectory. bfp8 small-probability drift can flip an accept/renoise
decision even when the argmax is unchanged, so the entropy PCC + accept IoU are the sensitive
metrics; argmax agreement + committed-match are the output metrics.

Determinism: one block of ``--max-denoising-steps`` steps is run from a FIXED seeded initial
canvas with FIXED per-step renoise tokens and CLEAN ARGMAX sampling (gumbel noise = None). Every
source of randomness is pinned, so the ONLY difference between a bf16 run and a bfp8 run is the
expert weight dtype (set via the DG_EXPERTS_BFP8 / DG_EXPERTS_DTYPE knob honoured by the DG model
builder). Run "run" twice (bf16 then bfp8) then "compare" the two saved trajectories.

Usage:
  # reference (bf16 experts — no knob)
  python decision_agreement.py run --output /path/traj_bf16.pt --label bf16
  # candidate (bfp8 experts)
  DG_EXPERTS_BFP8=1 python decision_agreement.py run --output /path/traj_bfp8.pt --label bfp8
  # decision agreement of candidate vs reference
  python decision_agreement.py compare --ref /path/traj_bf16.pt --cand /path/traj_bfp8.pt \
      --output /path/agreement.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict

import torch


def _seeded_canvas(seed: int, canvas_len: int, vocab: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (1, canvas_len), dtype=torch.long, generator=g)


def _seeded_noise(seed: int, steps: int, canvas_len: int, vocab: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed + 1000)
    return [torch.randint(0, vocab, (1, canvas_len), dtype=torch.long, generator=g) for _ in range(steps)]


def run(args) -> int:
    from models.experimental.diffusion_gemma.checkpoint import (
        build_tt_model_from_checkpoint_inputs,
        generate_text_from_checkpoint_model_inputs,
        load_checkpoint_inputs,
        text_generation_prefixes_for_layers,
    )
    from models.experimental.diffusion_gemma.config import DiffusionConfig
    from models.experimental.diffusion_gemma.demo.text_demo import (
        _close_mesh_device,
        _log_mesh_dram,
        _open_mesh_device,
    )
    from models.experimental.diffusion_gemma.tt.generate import (
        make_host_canvas_init_fn,
        make_host_noise_tokens_fn,
    )
    from models.experimental.diffusion_gemma.tt.precision_build import dg_experts_dtype_override

    vocab = args.vocab
    canvas = args.canvas_length
    steps = args.max_denoising_steps
    host_canvas = _seeded_canvas(args.seed, canvas, vocab)
    noise = _seeded_noise(args.seed, steps, canvas, vocab)
    # entropy_stop_threshold=-1.0 disables early-halt so bf16 and bfp8 run exactly ``steps``
    # steps (mean entropy is always >= 0, so the < -1.0 halt condition can never fire).
    config = DiffusionConfig(
        canvas_length=canvas,
        max_denoise_steps=steps,
        entropy_stop_threshold=-1.0,
    )

    checkpoint_inputs = load_checkpoint_inputs(
        args.checkpoint,
        tokenizer_kwargs={"local_files_only": True, "trust_remote_code": True},
        state_prefixes=text_generation_prefixes_for_layers(args.num_layers),
        device="cpu",
    )
    mesh = _open_mesh_device(args.mesh)
    try:
        _log_mesh_dram(mesh, "baseline")
        bundle = build_tt_model_from_checkpoint_inputs(
            mesh,
            checkpoint_inputs,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
        )
        _log_mesh_dram(mesh, "post-build")
        override = dg_experts_dtype_override()
        print(f"DG_DA_EXPERTS_DTYPE_OVERRIDE={override}")
        gen = generate_text_from_checkpoint_model_inputs(
            bundle,
            args.prompt,
            num_blocks=1,
            config=config,
            init_canvas_fn=make_host_canvas_init_fn(mesh, [host_canvas]),
            gumbel_noise_fn=lambda block_idx: (lambda step: None),  # clean argmax -> deterministic
            noise_tokens_fn=make_host_noise_tokens_fn(mesh, [noise]),
            max_new_tokens=canvas,
            eos_token_id=None,
            stop_token_ids=None,
            seed=args.seed,
        )
        traj = gen.generation.trajectories[0]
        text = list(gen.text)
        artifact = {
            "traj": traj,
            "committed": traj.committed,
            "text": text,
            "label": args.label,
            "experts_dtype_override": str(override),
            "prompt": args.prompt,
            "seed": args.seed,
            "steps": steps,
            "canvas_length": canvas,
            "num_layers": args.num_layers,
        }
        torch.save(artifact, args.output)
        n_eos = int((traj.committed == 1).sum())
        print(
            f"DG_DA_RUN_DONE label={args.label} steps={steps} committed_eos={n_eos}/{traj.committed.numel()} out={args.output}"
        )
        for t in text:
            print(f"DG_DA_TEXT[{args.label}]: {t!r}")
    finally:
        _close_mesh_device(mesh)
    return 0


def compare(args) -> int:
    from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories

    ref = torch.load(args.ref, weights_only=False)
    cand = torch.load(args.cand, weights_only=False)
    cmp = compare_trajectories(
        ref["traj"],
        cand["traj"],
        min_argmax_agreement=0.0,
        min_sampled_agreement=0.0,
        min_accept_iou=0.0,
        min_canvas_agreement=0.0,
        min_per_step_entropy_pcc=0.0,
        max_entropy_abs_err_threshold=1e9,
        committed_match_threshold=0.0,
        entropy_pcc_threshold=0.0,
    )
    d = asdict(cmp)

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else float("nan")

    summary = {
        "ref_label": ref.get("label"),
        "cand_label": cand.get("label"),
        "ref_experts_dtype": ref.get("experts_dtype_override"),
        "cand_experts_dtype": cand.get("experts_dtype_override"),
        "prompt": ref.get("prompt"),
        "seed": ref.get("seed"),
        "steps": ref.get("steps"),
        "num_layers": ref.get("num_layers"),
        "mean_argmax_agreement": _mean(d["per_step_argmax_agreement"]),
        "min_argmax_agreement": d["min_argmax_agreement"],
        "mean_accept_iou": _mean(d["per_step_accept_iou"]),
        "min_accept_iou": d["min_accept_iou"],
        "mean_canvas_agreement": _mean(d["per_step_canvas_agreement"]),
        "min_canvas_agreement": d["min_canvas_agreement"],
        "mean_entropy_pcc": _mean(d["per_step_entropy_pcc"]),
        "min_entropy_pcc": d["min_entropy_pcc"],
        "entropy_traj_pcc": d.get("entropy_traj_pcc"),
        "committed_match": d["committed_match"],
        "per_step_argmax_agreement": d["per_step_argmax_agreement"],
        "per_step_accept_iou": d["per_step_accept_iou"],
        "per_step_entropy_pcc": d["per_step_entropy_pcc"],
        "per_step_entropy_max_abs": d.get("per_step_entropy_max_abs"),
        "ref_text": ref.get("text"),
        "cand_text": cand.get("text"),
    }
    print("DG_DA_COMPARE " + json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, default=str))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"DG_DA_COMPARE_SAVED {args.output}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--checkpoint", default=os.environ.get("DG_CKPT", "/home/zni/dg_models/diffusiongemma-26B-A4B-it"))
    r.add_argument("--mesh", default="P150x4")
    r.add_argument("--num-layers", type=int, default=None)
    r.add_argument("--max-seq-len", type=int, default=1024)
    r.add_argument("--prompt", default="Explain what a diffusion language model is in one sentence.")
    r.add_argument("--canvas-length", type=int, default=256)
    r.add_argument("--max-denoising-steps", type=int, default=12)
    r.add_argument("--vocab", type=int, default=262144)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--label", default="bf16")
    r.add_argument("--output", required=True)
    r.set_defaults(func=run)

    c = sub.add_parser("compare")
    c.add_argument("--ref", required=True)
    c.add_argument("--cand", required=True)
    c.add_argument("--output", default=None)
    c.set_defaults(func=compare)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
