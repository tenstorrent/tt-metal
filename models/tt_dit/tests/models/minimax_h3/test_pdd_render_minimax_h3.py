# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""One t2va render from a PDD-head ComfyUI file (H3_TRANSFORMER_FILE), e.g. 10Eros-Max beta7 fix1."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import align_num_frames
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import weights_dir

PROMPT = os.environ.get(
    "PDD_PROMPT", "a woman in a red dress dancing slowly in a candlelit room, cinematic, soft light"
)
OUT = Path(os.environ.get("PDD_OUT", "/tmp/pdd-out"))


@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_pdd_render(mesh_device):
    t0 = time.time()
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("transformer", "text_encoder", "vae", "audio_vae"),
        task="t2va",
        dit_fsdp=False,
        vae_output_type="yuv420",
        video_shift=12.0,
        audio_shift=3.0,
        warmup=False,
    )
    logger.info(
        f"pdd load+warm {time.time() - t0:.1f}s file={os.environ.get(chr(72) + chr(51) + str('_TRANSFORMER_FILE'))}"
    )
    num_frames = align_num_frames(round(float(os.environ.get("PDD_SECONDS", "6")) * MINIMAX_H3_FPS))
    OUT.mkdir(parents=True, exist_ok=True)
    for run in range(int(os.environ.get("PDD_RUNS", "2"))):
        seed = 11 + run
        t1 = time.time()
        output = pipeline(PROMPT, num_frames=num_frames, height=768, width=1344, num_inference_steps=9, seed=seed)
        logger.info(f"pdd gen{run} seed={seed} {time.time() - t1:.2f}s")
        mp4 = OUT / f"pdd_s{seed}.mp4"
        export_video_audio_yuv(
            output.video,
            str(mp4),
            fps=output.fps,
            audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
        )
        logger.info(f"pdd wrote {mp4}")
