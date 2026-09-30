# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""fl2va wall-clock baseline at an explicit canvas and clip length, with per-seed reference artifacts.

Env: BASE_SECONDS (10), BASE_HEIGHT/BASE_WIDTH (1088/1920), BASE_STEPS (50), BASE_WARM_STEPS (2),
BASE_SEEDS ("0"), BASE_FIRST / BASE_LAST (keyframe paths; BASE_LAST optional), BASE_PROMPT,
BASE_VSA_SPARSITY (unset = dense), BASE_OUT (artifact root). Each seed writes mp4, latents, stills and a
timings json as soon as it finishes, so a job cut short by the broker keeps what completed.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

import ttnn

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import GALAXY_MESHES
from .common_av import to_uint8_frames, weights_dir, write_artifacts

DEFAULT_PROMPT = (
    "A slow cinematic camera pan across the scene, natural motion, soft ambient sound, "
    "consistent lighting from the first frame to the last."
)


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_fasth3_fl2va_baseline(mesh_device, reset_seeds):
    seconds = float(os.environ.get("BASE_SECONDS", "10"))
    height = int(os.environ.get("BASE_HEIGHT", "1088"))
    width = int(os.environ.get("BASE_WIDTH", "1920"))
    steps = int(os.environ.get("BASE_STEPS", "50"))
    warm_steps = int(os.environ.get("BASE_WARM_STEPS", "2"))
    seeds = [int(s) for s in os.environ.get("BASE_SEEDS", "0").split(",")]
    prompt = os.environ.get("BASE_PROMPT", DEFAULT_PROMPT)
    sparsity = os.environ.get("BASE_VSA_SPARSITY")
    out_root = Path(os.environ["BASE_OUT"])
    first = Image.open(os.environ["BASE_FIRST"]).convert("RGB")
    last = Image.open(os.environ["BASE_LAST"]).convert("RGB") if os.environ.get("BASE_LAST") else None

    vsa_config = None
    if sparsity is not None:
        from ....models.transformers.minimax_h3.vsa_stages_minimax_h3 import MiniMaxH3VSAConfig

        vsa_config = MiniMaxH3VSAConfig(sparsity=float(sparsity))
    mode = "dense" if vsa_config is None else f"vsa{sparsity}"
    num_frames = align_num_frames(round(seconds * MINIMAX_H3_FPS))
    tag = f"fl2va_{height}p_{seconds:g}s_{mode}_{steps}steps"
    out_dir = out_root / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights, vsa_config=vsa_config)
    gen_kwargs = dict(image=first, last_image=last, num_frames=num_frames, height=height, width=width)
    logger.info(f"BASELINE {tag}: pipeline built in {time.time() - t0:.1f}s")

    # Programs are keyed on the padded length, not the step count, so a short warmup at the real
    # shape and keyframes compiles the same set a 50-step call runs.
    t0 = time.time()
    pipeline.warmup(prompt=prompt, num_inference_steps=warm_steps, **gen_kwargs)
    logger.info(f"BASELINE {tag}: warmup ({warm_steps} steps) wall {time.time() - t0:.1f}s")
    cold_rows = list(pipeline.last_timings)
    warm_padded_len = pipeline.last_padded_len

    for seed in seeds:
        ttnn.synchronize_device(mesh_device)
        t0 = time.time()
        output = pipeline(prompt, seed=seed, num_inference_steps=steps, **gen_kwargs)
        wall = time.time() - t0
        assert pipeline.last_padded_len == warm_padded_len
        rows = list(pipeline.last_timings)
        total = sum(s for _, s in rows)
        denoise = dict(rows).get("Denoise", 0.0)
        record = dict(
            tag=tag,
            seed=seed,
            mesh=list(mesh_device.shape),
            tp=pipeline.tp_factor,
            sp=pipeline.sp_factor,
            height=height,
            width=width,
            num_frames=output.num_frames,
            fps=output.fps,
            video_seconds=output.video_seconds,
            steps=steps,
            forwards=steps - 1,
            padded_len=pipeline.last_padded_len,
            attention=mode,
            prompt=prompt,
            rows=rows,
            total_compute_s=total,
            call_wall_s=wall,
            denoise_per_forward_s=denoise / max(steps - 1, 1),
            warmup_rows=cold_rows,
        )
        logger.info(f"BASELINE {tag} seed={seed} total={total:.1f}s wall={wall:.1f}s rows={rows}")

        stem = f"seed{seed}"
        frames = to_uint8_frames(output)
        write_artifacts(frames, output.audio.float().cpu().numpy(), output.sampling_rate, out_dir, stem=stem)
        for label, index in (("first", 0), ("mid", len(frames) // 2), ("last", len(frames) - 1)):
            Image.fromarray(frames[index]).save(out_dir / f"{stem}_{label}.png")
        torch.save(pipeline.last_latents, out_dir / f"{stem}_latents.pt")
        # Lossless every-16th-frame subset for PSNR; the mp4 is lossy and a full uint8 clip is ~1.5 GB.
        np.save(out_dir / f"{stem}_frames_u8_every16.npy", frames[::16])
        (out_dir / f"{stem}_timings.json").write_text(json.dumps(record, indent=2))
