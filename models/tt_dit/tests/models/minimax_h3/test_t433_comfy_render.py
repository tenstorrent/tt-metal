# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""t433: one ref2va render from a ComfyUI-pruned H3 file (H3_TRANSFORMER_FILE, H3_TRANSFORMER_LAYOUT=comfy_pruned).

The fused 10Eros 8stepTURBO file carries its own contract: 8 forwards, video shift 12, audio shift 3, no guidance.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from loguru import logger
from PIL import Image

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS
from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import align_num_frames
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import weights_dir

PROMPT = os.environ.get(
    "T433_PROMPT",
    "The owl mascot from image 1 waves in front of the FreshMart store from image 2 on a sunny day, "
    "the camera holds steady",
)
REFS = os.environ.get("T433_REFS", "").split(",")
SEED = int(os.environ.get("T433_SEED", "13"))
STEPS = int(os.environ.get("T433_STEPS", "9"))
OUT = Path(os.environ.get("T433_OUT", "/tmp/t433-out"))


@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_t433_render(mesh_device):
    references = [MiniMaxH3Reference(image=Image.open(p).convert("RGB")) for p in REFS if p]
    task = "ref2va" if references else "t2va"
    sub = "transformer_ref" if references else "transformer"
    t0 = time.time()
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir(sub, "text_encoder", "vae", "audio_vae"),
        task=task,
        dit_fsdp=False,
        vae_output_type="yuv420",
        video_shift=float(os.environ.get("T433_VSHIFT", "12")),
        audio_shift=float(os.environ.get("T433_ASHIFT", "3")),
    )
    logger.info(f"t433 load+warm {time.time() - t0:.1f}s file={os.environ.get('H3_TRANSFORMER_FILE')}")
    num_frames = align_num_frames(round(5 * MINIMAX_H3_FPS))
    OUT.mkdir(parents=True, exist_ok=True)
    for run in range(int(os.environ.get("T433_RUNS", "2"))):
        t1 = time.time()
        output = pipeline(
            PROMPT,
            references=references or None,
            num_frames=num_frames,
            height=768,
            width=1344,
            num_inference_steps=STEPS,
            seed=SEED,
        )
        logger.info(f"t433 gen{run} {time.time() - t1:.2f}s")
    mp4 = OUT / f"{task}_s{SEED}.mp4"
    export_video_audio_yuv(
        output.video,
        str(mp4),
        fps=output.fps,
        audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
    )
    logger.info(f"t433 wrote {mp4}")
