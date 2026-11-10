# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""t454: Comfy-Org pruned base H3 (H3_TRANSFORMER_FILE, comfy_pruned) vs our diffusers H3, fl2va, same seeds.

Run once with H3_TRANSFORMER_FILE unset (diffusers) and once with it set; each run renders T454_SEEDS
on one warm pipeline and writes <tag>_s<seed>.mp4 plus a timings line per seed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
from loguru import logger
from PIL import Image

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import align_num_frames
from ....utils.video import Audio, export_video_audio_yuv
from .common import GALAXY_MESHES
from .common_av import weights_dir

IN = Path(os.environ.get("T454_IN", "/home/smarton/t454/in"))
SEEDS = [int(s) for s in os.environ.get("T454_SEEDS", "21,22,23").split(",")]
STEPS = int(os.environ.get("T454_STEPS", "30"))
SECONDS = float(os.environ.get("T454_SECONDS", "5"))
OUT = Path(os.environ.get("T454_OUT", "/home/smarton/t454/out"))


@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES[:1], indirect=["mesh_device", "device_params"])
def test_t454_ab(mesh_device):
    tag = "comfy" if os.environ.get("H3_TRANSFORMER_FILE") else "diffusers"
    prompt = (IN / "ex3.txt").read_text().strip()
    first = Image.open(IN / "first.png").convert("RGB")
    last = Image.open(IN / "last.png").convert("RGB")
    t0 = time.time()
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir("transformer", "text_encoder", "vae", "audio_vae"),
        task="t2va",
        dit_fsdp=False,
        vae_output_type="yuv420",
    )
    logger.info(f"t454 {tag} load+warm {time.time() - t0:.1f}s file={os.environ.get('H3_TRANSFORMER_FILE')}")
    num_frames = align_num_frames(round(SECONDS * MINIMAX_H3_FPS))
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        t1 = time.time()
        output = pipeline(
            prompt,
            image=first,
            last_image=last,
            num_frames=num_frames,
            height=768,
            width=1344,
            num_inference_steps=STEPS,
            seed=seed,
        )
        gen_s = time.time() - t1
        mp4 = OUT / f"{tag}_s{seed}.mp4"
        export_video_audio_yuv(
            output.video,
            str(mp4),
            fps=output.fps,
            audio=Audio(waveform=output.audio[0], sampling_rate=output.sampling_rate),
        )
        row = {
            "tag": tag,
            "seed": seed,
            "gen_s": round(gen_s, 2),
            "steps": STEPS,
            "frames": num_frames,
            "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mp4": str(mp4),
        }
        with open(OUT / "times.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        logger.info(f"t454 {json.dumps(row)}")
