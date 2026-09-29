# SPDX-FileCopyrightText: © 2026 Qwen Image 2.1 contributors
# SPDX-License-Identifier: Apache-2.0

"""Export the pinned pipeline's exact sigma schedule and verify captured timesteps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift
from huggingface_hub import snapshot_download

from models.experimental.qwen_image_2_1.checkpoint import MODEL_ID, MODEL_REVISION


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.cuda_dir / "manifest.json").read_text())
    if manifest["model_revision"] != MODEL_REVISION:
        raise ValueError("CUDA capture checkpoint revision differs from the pinned model")
    height, width = manifest["height"], manifest["width"]
    if height < 32 or width < 32 or height % 32 or width % 32:
        raise ValueError("captured height and width must be multiples of 32")
    latent_tokens = (height // 16) * (width // 16)
    captured_latents = torch.load(
        args.cuda_dir / "step_000/transformer/input/hidden_states.pt",
        weights_only=True,
        map_location="cpu",
    )
    if tuple(captured_latents.shape) != (1, latent_tokens, 64):
        raise ValueError("CUDA latent shape does not match the captured image resolution")
    count = manifest["steps"]
    if manifest["capture_steps"] != list(range(count)):
        raise ValueError("all CUDA timesteps must be captured to verify the schedule")
    checkpoint = snapshot_download(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint, subfolder="scheduler", local_files_only=True
    )
    mu = calculate_shift(
        latent_tokens,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    scheduler.set_timesteps(count, device="cpu", sigmas=np.linspace(1.0, 1.0 / count, count), mu=mu)
    sigmas = scheduler.sigmas.cpu().tolist()
    if len(sigmas) != count + 1:
        raise RuntimeError("scheduler did not return one terminal sigma")
    comparisons = []
    for step in range(count):
        captured = (
            torch.load(
                args.cuda_dir / f"step_{step:03d}/transformer/input/timestep.pt",
                weights_only=True,
                map_location="cpu",
            )
            .float()
            .item()
        )
        # The pipeline first rounds the scheduler timestep (1000*sigma) to
        # BF16, then divides by 1000 in BF16 before calling the transformer.
        expected_bf16 = float(scheduler.timesteps[step].to(torch.bfloat16) / 1000)
        difference = abs(captured - expected_bf16)
        comparisons.append(difference)
        if difference > 1e-6:
            raise ValueError(f"step {step}: CUDA timestep {captured} != sigma {expected_bf16}")
    document = {
        "model_revision": MODEL_REVISION,
        "prompt": manifest["prompt"],
        "seed": manifest["seed"],
        "height": manifest["height"],
        "width": manifest["width"],
        "latent_tokens": latent_tokens,
        "steps": count,
        "mu": mu,
        "sigmas": sigmas,
        "max_captured_timestep_abs_error": max(comparisons),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(f"verified {count} CUDA timesteps; wrote {args.output}")


if __name__ == "__main__":
    main()
