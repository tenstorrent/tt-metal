# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""OpenVDN final-correctness scores on a Blackhole 4x8 (TP 4 x SP 8): a fixed set of prompts at seed 0 and the
production configuration (t2va, 16:9 at 1344x768, 5 s, the released 8-step checkpoint), each written as an mp4 and
scored with CLIP (mean over 8 evenly spaced frames, ViT-B/32, x100) and VBench (the dimensions H3's tests use).

No GPU or customer reference exists for OpenVDN yet, so the scores are recorded, not compared. The only gate is the
end-to-end test's ball-park CLIP floor (30) and frame/audio sanity.

Prompts: the H3 calibration fox prompt, and short single-scene versions of the three t2va examples the OpenVDN repo
ships (`prompts/example_{0,1,2}.pt`: an animated ronin, an elf with a water fox, a woman in Cologne Cathedral). The
originals are 13-15 s multi-shot scripts, too long for 5 s and for CLIP's 77 tokens.

Env: OPENVDN_WEIGHTS_DIR, OPENVDN_CHECKPOINT_DIR, OPENVDN_ARTIFACTS, OPENVDN_DURATION_S (see test_pipeline_openvdn.py);
RUN_VBENCH=0 skips VBench, which otherwise needs ~/vbench_env (see `run_vbench`).

Run: pytest models/tt_dit/tests/models/openvdn/test_quality_openvdn.py -s --tb=short
"""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from loguru import logger
from PIL import Image

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, resolve_canvas_size
from ....pipelines.minimax_h3.policy import get_num_frames
from ....pipelines.openvdn.pipeline_openvdn import OPENVDN_NUM_FORWARDS, OpenVDNPipeline
from ..minimax_h3.common import MESH_4X8_RING
from ..minimax_h3.common_av import (
    CALIBRATED_FOX_PROMPT,
    VBENCH_PYTHON,
    check_audio_sanity,
    clip_prompt_alignment,
    is_host,
    run_vbench,
    to_uint8_frames,
    write_artifacts,
)
from ..wan2_2.common import check_output_sanity
from .test_pipeline_openvdn import (
    ASPECT_RATIO,
    CLIP_FLOOR,
    DURATION_S,
    NUM_INFERENCE_STEPS,
    SEED,
    _required_dir,
)

PROMPTS = {
    "fox": CALIBRATED_FOX_PROMPT,
    "ronin": (
        "2D animation: a female ronin in black silk robes stands in a dark void of rolling grey smoke and glowing "
        "red embers, then draws her black katana in a fast crimson slash, her long black hair whipping behind her."
    ),
    "water_fox": (
        "3D fantasy animation: a young elf with long blue hair crouches on a mossy stone bridge in a misty jungle at "
        "dawn, as a glowing fox made of water rises from the stream and circles her hand, scattering bright droplets."
    ),
    "cathedral": (
        "Cinematic: a young woman with long wavy dark hair and a cream linen blouse walks slowly into the nave of "
        "Cologne Cathedral, golden light and stained-glass colors falling across the stone columns and her face."
    ),
}

VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
)


@pytest.mark.timeout(21600)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_openvdn_t2va_quality(mesh_device, reset_seeds):
    weights_dir = _required_dir("OPENVDN_WEIGHTS_DIR")
    checkpoint_dir = _required_dir("OPENVDN_CHECKPOINT_DIR")
    pytest.importorskip("open_clip", reason="the CLIP score needs open_clip, which is not installed")
    run_vb = os.environ.get("RUN_VBENCH", "1") not in ("0", "false", "False")
    if run_vb and not VBENCH_PYTHON.is_file():
        pytest.fail(f"no VBench interpreter at {VBENCH_PYTHON} (set RUN_VBENCH=0 to skip VBench)")
    artifacts = Path(os.environ.get("OPENVDN_ARTIFACTS", Path.home() / "openvdn_artifacts"))
    artifacts.mkdir(parents=True, exist_ok=True)

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = get_num_frames(DURATION_S)
    config = dict(num_frames=num_frames, height=height, width=width, seed=SEED)

    pipeline = OpenVDNPipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights_dir,
        checkpoint_dir=checkpoint_dir,
        vae_output_type="float",
    )

    results = {}
    for name, prompt in PROMPTS.items():
        stem = f"openvdn_t2va_{name}_{width}x{height}_{num_frames}f_{OPENVDN_NUM_FORWARDS}nfe_seed{SEED}"
        t0 = time.time()
        output = pipeline(prompt, num_inference_steps=NUM_INFERENCE_STEPS, **config)
        generate_s = time.time() - t0

        frames = to_uint8_frames(output)
        check_output_sanity(frames, num_frames=num_frames, height=height, width=width)
        check_audio_sanity(
            output.audio, sampling_rate=output.sampling_rate, expected_seconds=num_frames / MINIMAX_H3_FPS
        )
        if not is_host():
            continue
        paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
        assert "mp4" in paths, "no mp4 written: neither ffmpeg nor imageio-ffmpeg is available"
        Image.fromarray(frames[num_frames // 2]).save(artifacts / f"{stem}_mid.png")
        alignment = clip_prompt_alignment(frames, prompt)
        results[name] = {
            "prompt": prompt,
            "generate_s": generate_s,
            "clip_mean": alignment["mean"],
            "clip_min": alignment["min"],
            "clip_max": alignment["max"],
            "mp4": str(paths["mp4"]),
        }
        logger.info(f"OpenVDN quality [{name}]: {generate_s:.0f} s, CLIP {alignment['mean']:.2f}")

    if not is_host():
        return
    if run_vb:
        # VBench runs on the CPU in its own interpreter; score all videos at once.
        with ThreadPoolExecutor(max_workers=len(results)) as pool:
            futures = {
                name: pool.submit(run_vbench, Path(r["mp4"]), r["prompt"], VBENCH_DIMENSIONS)
                for name, r in results.items()
            }
            for name, future in futures.items():
                results[name]["vbench"] = future.result()
                logger.info(f"OpenVDN quality [{name}]: VBench {results[name]['vbench']}")

    summary = {
        "config": {**config, "num_inference_steps": NUM_INFERENCE_STEPS, "fps": MINIMAX_H3_FPS},
        "reference": None,
        "clip_floor": CLIP_FLOOR,
        "videos": results,
    }
    (artifacts / "openvdn_quality_scores.json").write_text(json.dumps(summary, indent=2))
    logger.info(f"OpenVDN quality: {json.dumps(summary)}")
    low = {name: r["clip_mean"] for name, r in results.items() if r["clip_mean"] < CLIP_FLOOR}
    assert not low, f"mean CLIP below the {CLIP_FLOOR} floor: {low}"
