# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN end to end on a Blackhole 4x8 (TP 4 x SP 8): text to audio + video with the released 8-step checkpoint,
checked for ball-park correctness.

Gates: frame and audio sanity (shape, range, motion, silence) and prompt alignment: the mean CLIP score (ViT-B/32,
x100) over 8 evenly spaced frames must reach CLIP_FLOOR. No GPU score for OpenVDN exists yet (stage 3 not run); the
floor sits below the MiniMax-H3 base gate on the same prompt and encoder (33, `test_pipeline_minimax_h3.py`), which
the VDN student is distilled from, and equals Wan's text-to-video floor (30, typical 37).

Env:
  OPENVDN_WEIGHTS_DIR     H3 diffusers snapshot: the VDN repo's h3-base/ plus MiniMax-H3's text_encoder/,
                          tokenizer/, processor/ (see `pipeline_openvdn.py`)
  OPENVDN_CHECKPOINT_DIR  a VDN checkpoint, e.g. <OpenVDN/vdn-minimax-h3>/stage-dmd-step-250
  OPENVDN_ARTIFACTS       where the mp4/wav/frames and scores.json go (default ~/openvdn_artifacts)
  OPENVDN_DURATION_S      seconds of video (default 5)

Run: pytest models/tt_dit/tests/models/openvdn/test_pipeline_openvdn.py -s --tb=short
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from loguru import logger
from PIL import Image

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, resolve_canvas_size
from ....pipelines.minimax_h3.policy import get_num_frames
from ....pipelines.openvdn.pipeline_openvdn import OPENVDN_NUM_FORWARDS, OpenVDNPipeline
from ..minimax_h3.common import MESH_4X8_RING
from ..minimax_h3.common_av import (
    CALIBRATED_FOX_PROMPT,
    check_audio_sanity,
    clip_prompt_alignment,
    is_host,
    to_uint8_frames,
    write_artifacts,
)
from ..wan2_2.common import check_output_sanity

CLIP_FLOOR = 30.0
PROMPT = CALIBRATED_FOX_PROMPT
SEED = 0
NUM_INFERENCE_STEPS = OPENVDN_NUM_FORWARDS + 1  # sigma grid points
ASPECT_RATIO = (16, 9)
DURATION_S = float(os.environ.get("OPENVDN_DURATION_S", 5))


def _required_dir(name: str) -> Path:
    value = os.environ.get(name)
    if not value or not Path(value).is_dir():
        pytest.skip(f"set {name} (see this file's docstring)")
    return Path(value)


@pytest.mark.timeout(14400)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_openvdn_t2va_end_to_end(mesh_device, reset_seeds):
    weights_dir = _required_dir("OPENVDN_WEIGHTS_DIR")
    checkpoint_dir = _required_dir("OPENVDN_CHECKPOINT_DIR")
    pytest.importorskip("open_clip", reason="the CLIP gate needs open_clip, which is not installed")
    artifacts = Path(os.environ.get("OPENVDN_ARTIFACTS", Path.home() / "openvdn_artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = get_num_frames(DURATION_S)
    stem = f"openvdn_t2va_{width}x{height}_{num_frames}f_{OPENVDN_NUM_FORWARDS}nfe_seed{SEED}"

    t0 = time.time()
    pipeline = OpenVDNPipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir,
        checkpoint_dir=checkpoint_dir,
        vae_output_type="float",
    )
    load_s = time.time() - t0

    t0 = time.time()
    output = pipeline(
        PROMPT,
        num_frames=num_frames,
        height=height,
        width=width,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
    generate_s = time.time() - t0
    logger.info(f"OpenVDN: load {load_s:.0f} s, first (cold) generation {generate_s:.0f} s")

    frames = to_uint8_frames(output)
    check_output_sanity(frames, num_frames=num_frames, height=height, width=width)
    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=num_frames / MINIMAX_H3_FPS)

    if not is_host():
        return
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
    # The 8 frames the CLIP score reads, as stills for the results page.
    for i in np.linspace(0, frames.shape[0] - 1, 8).astype(int):
        Image.fromarray(frames[i]).save(artifacts / f"{stem}_frame{i:03d}.png")
    alignment = clip_prompt_alignment(frames, PROMPT)
    scores = {
        "prompt": PROMPT,
        "seed": SEED,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "clip_mean": alignment["mean"],
        "clip_min": alignment["min"],
        "clip_max": alignment["max"],
        "clip_floor": CLIP_FLOOR,
        "passed": alignment["mean"] >= CLIP_FLOOR,
        "load_s": load_s,
        "cold_generation_s": generate_s,
        "merged_adapter_pairs": pipeline.merged_adapter_pairs,
        "branch_tensors": pipeline.branch_tensors,
        "files": {k: str(v) for k, v in paths.items()},
    }
    (artifacts / f"{stem}_scores.json").write_text(json.dumps(scores, indent=2))
    logger.info(
        f"CLIP prompt alignment: mean {alignment['mean']:.2f} (floor {CLIP_FLOOR}), "
        f"min {alignment['min']:.2f}, max {alignment['max']:.2f}; artifacts in {artifacts}"
    )
    assert alignment["mean"] >= CLIP_FLOOR, f"mean CLIP score {alignment['mean']:.2f} is below the {CLIP_FLOOR} floor"
