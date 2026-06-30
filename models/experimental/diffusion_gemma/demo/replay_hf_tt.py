# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Replay one DiffusionGemma denoise block through HF and TT.

This is a focused bring-up harness for R0.5/#48291 fidelity debugging. It drives
both implementations with the same prompt, initial canvas, Gumbel noise, and
renoise tokens, then saves the decision-level trajectory comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import torch

from models.experimental.diffusion_gemma.checkpoint import (
    build_tt_model_from_checkpoint_inputs,
    generate_text_from_checkpoint_model_inputs,
    load_checkpoint_inputs,
    text_generation_prefixes_for_layers,
)
from models.experimental.diffusion_gemma.config import DiffusionConfig
from models.experimental.diffusion_gemma.demo.text_demo import _close_mesh_device, _log_mesh_dram, _open_mesh_device
from models.experimental.diffusion_gemma.reference import sampling as S
from models.experimental.diffusion_gemma.reference.denoise_loop import denoise_block
from models.experimental.diffusion_gemma.tests.trajectory_pcc import compare_trajectories
from models.experimental.diffusion_gemma.tt.generate import (
    make_host_canvas_init_fn,
    make_host_gumbel_noise_fn,
    make_host_noise_tokens_fn,
    tokenize_prompt,
)

DEFAULT_PROMPT = "Complete the sentence: Once upon a time"


def _pad_prompt_tokens_for_hf_prefill(prompt_tokens: torch.Tensor, *, multiple: int = 32) -> torch.Tensor:
    pad = (-prompt_tokens.shape[1]) % multiple
    if pad == 0:
        return prompt_tokens
    padding = torch.zeros((prompt_tokens.shape[0], pad), dtype=prompt_tokens.dtype, device=prompt_tokens.device)
    return torch.cat([prompt_tokens, padding], dim=1)


def _make_config(args) -> DiffusionConfig:
    return DiffusionConfig(
        canvas_length=args.canvas_length,
        max_denoise_steps=args.max_denoising_steps,
        entropy_stop_threshold=args.entropy_stop_threshold,
        stable_steps_to_halt=args.stable_steps_to_halt,
    )


def _top_counts(tokens: torch.Tensor, *, limit: int = 8) -> list[list[int]]:
    values, counts = torch.unique(tokens.cpu(), return_counts=True)
    pairs = sorted([(int(tok), int(count)) for tok, count in zip(values, counts)], key=lambda x: -x[1])
    return [[tok, count] for tok, count in pairs[:limit]]


def _trajectory_summary(prefix: str, trajectory, *, eos_token_id: int | None) -> dict:
    eos_count = None
    non_eos_count = None
    if eos_token_id is not None:
        eos_count = int((trajectory.committed == eos_token_id).sum())
        non_eos_count = int((trajectory.committed != eos_token_id).sum())
    return {
        f"{prefix}_num_steps": trajectory.num_steps,
        f"{prefix}_halted": trajectory.halted,
        f"{prefix}_committed_eos": eos_count,
        f"{prefix}_committed_non_eos": non_eos_count,
        f"{prefix}_committed_top": _top_counts(trajectory.committed),
        f"{prefix}_accept_counts": [int(step.accept_mask.sum()) for step in trajectory.per_step],
    }


def _decision_diff_summary(hf_traj, tt_traj, *, limit: int = 32) -> dict:
    per_step = []
    for hf_step, tt_step in zip(hf_traj.per_step, tt_traj.per_step):
        argmax_diff = (hf_step.argmax != tt_step.argmax).nonzero(as_tuple=False)
        accept_diff = (hf_step.accept_mask != tt_step.accept_mask).nonzero(as_tuple=False)
        canvas_diff = (hf_step.canvas != tt_step.canvas).nonzero(as_tuple=False)
        step_summary = {
            "step": hf_step.step,
            "argmax_diff_count": int(argmax_diff.shape[0]),
            "accept_diff_count": int(accept_diff.shape[0]),
            "canvas_diff_count": int(canvas_diff.shape[0]),
            "argmax_diff": [],
            "accept_diff": [],
        }
        for b, pos in argmax_diff[:limit].tolist():
            step_summary["argmax_diff"].append(
                {
                    "batch": int(b),
                    "pos": int(pos),
                    "hf_argmax": int(hf_step.argmax[b, pos]),
                    "tt_argmax": int(tt_step.argmax[b, pos]),
                    "hf_sampled": int(hf_step.sampled[b, pos]),
                    "tt_sampled": int(tt_step.sampled[b, pos]),
                    "hf_accept": bool(hf_step.accept_mask[b, pos]),
                    "tt_accept": bool(tt_step.accept_mask[b, pos]),
                    "hf_entropy": float(hf_step.entropy[b, pos]),
                    "tt_entropy": float(tt_step.entropy[b, pos]),
                }
            )
        for b, pos in accept_diff[:limit].tolist():
            step_summary["accept_diff"].append(
                {
                    "batch": int(b),
                    "pos": int(pos),
                    "hf_accept": bool(hf_step.accept_mask[b, pos]),
                    "tt_accept": bool(tt_step.accept_mask[b, pos]),
                    "hf_argmax": int(hf_step.argmax[b, pos]),
                    "tt_argmax": int(tt_step.argmax[b, pos]),
                    "hf_entropy": float(hf_step.entropy[b, pos]),
                    "tt_entropy": float(tt_step.entropy[b, pos]),
                }
            )
        per_step.append(step_summary)
    return {"per_step_diffs": per_step}


def _compare_summary(prompt: str, seed: int, hf_traj, tt_traj, *, eos_token_id: int | None) -> tuple[object, dict]:
    comparison = compare_trajectories(
        hf_traj,
        tt_traj,
        min_argmax_agreement=0.0,
        min_sampled_agreement=0.0,
        min_accept_iou=0.0,
        min_canvas_agreement=0.0,
        min_per_step_entropy_pcc=0.0,
        max_entropy_abs_err_threshold=10.0,
        committed_match_threshold=0.0,
        entropy_pcc_threshold=0.0,
    )
    summary = {
        "prompt": prompt,
        "seed": seed,
        **_trajectory_summary("hf", hf_traj, eos_token_id=eos_token_id),
        **_trajectory_summary("tt", tt_traj, eos_token_id=eos_token_id),
        "committed_match": comparison.committed_match,
        "per_step_argmax_agreement": comparison.per_step_argmax_agreement,
        "per_step_sampled_agreement": comparison.per_step_sampled_agreement,
        "per_step_accept_iou": comparison.per_step_accept_iou,
        "per_step_canvas_agreement": comparison.per_step_canvas_agreement,
        "per_step_entropy_pcc": comparison.per_step_entropy_pcc,
        "per_step_entropy_max_abs": comparison.per_step_entropy_max_abs,
        **_decision_diff_summary(hf_traj, tt_traj),
    }
    return comparison, summary


def _load_hf_model(checkpoint: str | Path, *, local_files_only: bool):
    from transformers import AutoTokenizer
    from transformers.models.diffusion_gemma import DiffusionGemmaForBlockDiffusion

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        checkpoint,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=local_files_only,
    ).eval()
    return tokenizer, model


def _hf_text_vocab_size(model, tokenizer) -> int:
    text_config = getattr(model.config, "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    return int(len(tokenizer))


def _run_hf_reference(model, tokenizer, prompt: str, host_canvas: torch.Tensor, config: DiffusionConfig):
    prompt_tokens = tokenize_prompt(tokenizer, prompt)
    prompt_tokens = _pad_prompt_tokens_for_hf_prefill(prompt_tokens)
    cache_len = prompt_tokens.shape[1]
    canvas_len = host_canvas.shape[1]
    position_ids = torch.arange(cache_len, dtype=torch.int64).unsqueeze(0)
    decoder_position_ids = torch.arange(cache_len, cache_len + canvas_len, dtype=torch.int64).unsqueeze(0)
    vocab_size = _hf_text_vocab_size(model, tokenizer)

    class HfLogits:
        def __init__(self) -> None:
            self.prev_raw = None
            self.prev_step = None

        def __call__(self, canvas, step):
            prev_sc = None
            if self.prev_raw is not None:
                temperature = S.temperature_at_step(
                    self.prev_step,
                    config.max_denoise_steps,
                    config.temperature_start,
                    config.temperature_end,
                )
                prev_sc = (self.prev_raw / temperature).to(torch.bfloat16)
            with torch.no_grad():
                out = model(
                    input_ids=prompt_tokens,
                    position_ids=position_ids,
                    decoder_input_ids=canvas,
                    decoder_position_ids=decoder_position_ids,
                    self_conditioning_logits=prev_sc,
                )
            logits = out.logits.float().cpu()
            self.prev_raw = logits
            self.prev_step = step
            return logits

    zero_gumbel = lambda _step: torch.zeros((1, canvas_len, vocab_size), dtype=torch.float32)  # noqa: E731
    zero_noise = lambda _step: torch.zeros((1, canvas_len), dtype=torch.long)  # noqa: E731
    trajectory = denoise_block(
        HfLogits(),
        host_canvas,
        config,
        vocab_size,
        sampler=S.SAMPLER_GUMBEL,
        gumbel_noise_fn=zero_gumbel,
        noise_tokens_fn=zero_noise,
    )
    return prompt_tokens, trajectory, vocab_size


def _run_tt_replay(args, prompt: str, host_canvas: torch.Tensor, config: DiffusionConfig, vocab_size: int):
    zero_gumbel = [[torch.zeros((1, config.canvas_length, vocab_size), dtype=torch.float32)]]
    zero_noise = [[torch.zeros((1, config.canvas_length), dtype=torch.long)]]

    checkpoint_inputs = load_checkpoint_inputs(
        args.checkpoint,
        tokenizer_kwargs={"local_files_only": args.local_files_only, "trust_remote_code": True},
        state_prefixes=text_generation_prefixes_for_layers(args.num_layers),
        device="cpu",
    )
    mesh_device = _open_mesh_device(args.mesh)
    try:
        _log_mesh_dram(mesh_device, "replay-baseline")
        checkpoint_model_inputs = build_tt_model_from_checkpoint_inputs(
            mesh_device,
            checkpoint_inputs,
            max_batch_size=1,
            max_seq_len=args.max_seq_len,
            num_layers=args.num_layers,
            bounded_sliding_kv_cache=args.bounded_sliding_kv_cache,
        )
        _log_mesh_dram(mesh_device, "replay-post-build")
        generation = generate_text_from_checkpoint_model_inputs(
            checkpoint_model_inputs,
            prompt,
            num_blocks=1,
            config=config,
            init_canvas_fn=make_host_canvas_init_fn(mesh_device, [host_canvas]),
            gumbel_noise_fn=make_host_gumbel_noise_fn(mesh_device, zero_gumbel),
            noise_tokens_fn=make_host_noise_tokens_fn(mesh_device, zero_noise),
            max_new_tokens=config.canvas_length,
            eos_token_id=None,
            stop_token_ids=None,
        )
        return generation.generation.trajectories[0]
    finally:
        _close_mesh_device(mesh_device)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay one DiffusionGemma block through HF and TT.")
    parser.add_argument(
        "--checkpoint",
        default=os.getenv("DG_CKPT", "google/diffusiongemma-26B-A4B-it"),
        help="DiffusionGemma checkpoint directory or model id for TT weights/tokenizer.",
    )
    parser.add_argument(
        "--hf-checkpoint",
        default=os.getenv("DG_HF_CKPT"),
        help="Optional separate HF checkpoint path/model id; defaults to --checkpoint.",
    )
    parser.add_argument("--prompt", default=os.getenv("DG_PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--seed", type=int, default=1, help="Seed for the initial host canvas.")
    parser.add_argument("--canvas-length", type=int, default=256)
    parser.add_argument("--max-denoising-steps", type=int, default=1)
    parser.add_argument("--entropy-stop-threshold", type=float, default=-1.0)
    parser.add_argument("--stable-steps-to-halt", type=int, default=1)
    parser.add_argument("--mesh", default=os.getenv("MESH_DEVICE", "P150x4"))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bounded-sliding-kv-cache", action="store_true")
    parser.add_argument("--hf-only", action="store_true", help="Save only the HF reference trajectory.")
    parser.add_argument("--output", default="/tmp/dg_replay_hf_tt_compare.pt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = _make_config(args)
    hf_checkpoint = args.hf_checkpoint or args.checkpoint

    tokenizer, hf_model = _load_hf_model(hf_checkpoint, local_files_only=args.local_files_only)
    vocab_size = _hf_text_vocab_size(hf_model, tokenizer)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    host_canvas = torch.randint(0, vocab_size, (1, config.canvas_length), dtype=torch.long, generator=generator)

    prompt_tokens, hf_traj, vocab_size = _run_hf_reference(hf_model, tokenizer, args.prompt, host_canvas, config)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    artifact = {
        "prompt": args.prompt,
        "seed": args.seed,
        "config": config,
        "prompt_tokens": prompt_tokens,
        "host_canvas": host_canvas,
        "hf_traj": hf_traj,
    }
    summary = {
        "prompt": args.prompt,
        "seed": args.seed,
        "hf_checkpoint": str(hf_checkpoint),
        "tt_checkpoint": str(args.checkpoint),
        "canvas_length": config.canvas_length,
        "max_denoising_steps": config.max_denoise_steps,
        **_trajectory_summary("hf", hf_traj, eos_token_id=eos_token_id),
    }

    if not args.hf_only:
        tt_traj = _run_tt_replay(args, args.prompt, host_canvas, config, vocab_size)
        comparison, summary = _compare_summary(args.prompt, args.seed, hf_traj, tt_traj, eos_token_id=eos_token_id)
        artifact.update({"tt_traj": tt_traj, "comparison": asdict(comparison)})

    artifact["summary"] = summary
    output = Path(args.output)
    torch.save(artifact, output)
    print(json.dumps({"output": str(output), "summary": summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
