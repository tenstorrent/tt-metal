# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Every generation mode under the two-time adapter, swept over clip length.

``test_pipeline_hyperflow_minimax_h3.py`` gates the adapter's contract on ``t2va`` alone. The
contract is task-blind by construction -- the grid, the gate and the endpoint blend are facts about
the adapter -- but the three modes reach the AdaLN table by different routes, and each route has a
level the others do not build:

* ``t2va`` -- two levels per step (video, audio).
* ``fl2va`` -- a third, the keyframe conditioning row pinned at ``max(video_t, noise_aug)``.
* ``ref2va`` -- a fourth, reference soundtrack rows at a literal ``t = 1.0``, off the other
  partition's weights (``transformer_ref/``).

An endpoint that failed to reach one of those levels is not visible in the output: the run
completes and produces video. So the assertion that matters here is
:func:`common_av.assert_hyperflow_applied` on each mode, not the pixels.

**Quality is recorded, not gated**, for the reason ``test_pipeline_hyperflow_minimax_h3.py``
gives: the CLIP and VBench bars elsewhere are calibrated against the 49-forward base model.

Each point writes a JSON sidecar next to its artifacts so a sweep can be tabulated without
scraping logs. One point per process: the broker caps a job at 25 minutes, and three modes'
DiT programs at three padded lengths in one interpreter is a memory risk besides.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames, resolve_canvas_size
from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference, reference_from_video_file
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..wan2_2.common import check_output_sanity
from .common import GALAXY_MESHES, create_fractal_image
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    assert_hyperflow_applied,
    check_audio_sanity,
    check_av_sync,
    check_written_file,
    clip_prompt_alignment,
    log_timing_table,
    run_warm_generation,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)

LORA_PATH_ENV = "MINIMAX_H3_HYPERFLOW_LORA_PATH"

SEED = 0
ASPECT_RATIO = (16, 9)
PROMPT = CALIBRATED_FOX_PROMPT
DURATIONS_S = (5, 10, 15)

# 16384, not the other modes' 65536: the video VAE's taps=3 encoder, which only ref2va reaches,
# clashes with L1 above it. Same reasoning and same value as the ref2va gate, which is why ref2va
# is a second entry point here rather than a third `task` on one parametrize -- the pool is fixed
# when the mesh opens, so it cannot vary per parameter within one test.
_REF2VA_L1_SMALL = 16384
REF2VA_MESHES = [
    pytest.param(shape, {**params, "l1_small_size": _REF2VA_L1_SMALL}, id=param.id)
    for param in GALAXY_MESHES
    for shape, params in [param.values]
]

# Structural only, and generous: it exists to fail a run that silently fell back to the
# 49-forward schedule, not to measure anything. ref2va carries the largest packed sequence.
MAX_S_PER_VIDEO_SECOND = 40.0

# `ref2va` reads the reference set off disk; `fl2va` reads its keyframe from the calibrated t2va
# artifact. Both skip rather than invent content -- a fabricated keyframe would make the recorded
# CLIP number describe a different request than the one the log names.
REFERENCE_MEDIA = Path.home() / "h3_fl2va_artifacts" / "fl2va_first.mp4"
T2VA_ARTIFACT = Path.home() / "h3_t2va_artifacts" / "t2va.mp4"


def _keyframe():
    import imageio.v3 as iio
    from PIL import Image

    if not T2VA_ARTIFACT.is_file():
        pytest.skip(f"no calibrated t2va artifact at {T2VA_ARTIFACT}; run test_pipeline_minimax_h3.py first")
    return Image.fromarray(iio.imread(T2VA_ARTIFACT, index=0, plugin="pyav")).convert("RGB")


def _references() -> list[MiniMaxH3Reference]:
    """Image, then silent video, then bare audio -- the `mixed` set the ref2va gate probes."""
    if not REFERENCE_MEDIA.is_file():
        pytest.skip(f"no reference video at {REFERENCE_MEDIA}; place a clip with a soundtrack there")
    sounded = reference_from_video_file(REFERENCE_MEDIA)
    return [
        MiniMaxH3Reference(image=create_fractal_image(1024, 1024)),
        reference_from_video_file(REFERENCE_MEDIA, with_audio=False),
        MiniMaxH3Reference(audio=sounded.audio, sample_rate=sounded.sample_rate),
    ]


def _request_kwargs(task: str) -> dict:
    if task == "t2va":
        return {}
    if task == "fl2va":
        keyframe = _keyframe()
        return {"image": keyframe, "last_image": keyframe}
    return {"references": _references()}


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"dur{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize("task", ("t2va", "fl2va"))
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_hyperflow_end_to_end(mesh_device, reset_seeds, task, duration_s):
    """``t2va`` and ``fl2va``: both off ``transformer/``, so one mesh configuration serves both."""
    _run_point(mesh_device, task, duration_s)


@pytest.mark.timeout(5400)
@pytest.mark.parametrize("duration_s", DURATIONS_S, ids=[f"dur{d}s" for d in DURATIONS_S])
@pytest.mark.parametrize(("mesh_device", "device_params"), REF2VA_MESHES, indirect=["mesh_device", "device_params"])
def test_ref2va_hyperflow_end_to_end(mesh_device, reset_seeds, duration_s):
    """``ref2va``: the other partition, the fourth AdaLN level, and a smaller L1 pool."""
    _run_point(mesh_device, "ref2va", duration_s)


def _run_point(mesh_device, task: str, duration_s: int) -> None:
    lora_path = os.environ.get(LORA_PATH_ENV)
    if not lora_path:
        pytest.skip(f"set {LORA_PATH_ENV} to a two-time adapter safetensors file")
    strength = float(os.environ.get("FASTH3_LORA_STRENGTH", "1.0"))

    # ref2va runs off the other 62 GB partition; the others share `transformer/`.
    partition = "transformer_ref" if task == "ref2va" else "transformer"
    weights = weights_dir(partition, "text_encoder", "vae", "audio_vae")
    artifacts = artifact_dir("h3_hyperflow_artifacts")
    pytest.importorskip("open_clip", reason="the CLIP measurement needs open_clip, which is not installed")

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = align_num_frames(round(duration_s * MINIMAX_H3_FPS))
    request = _request_kwargs(task)  # before the pipeline: a skip here should not pay a 62 GB load

    if not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning("TT_DIT_CACHE_DIR is unset; every weight load reads safetensors and the run will drag")

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        task="ref2va" if task == "ref2va" else "t2va",  # `t2va` serves fl2va too
        lora_path=lora_path,
        lora_strength=strength,
    )

    contract = pipeline.hyperflow
    assert contract is not None, (
        f"{lora_path} publishes no sampling contract, so this pipeline would run it at 50 sigma "
        f"points; point {LORA_PATH_ENV} at a two-time adapter"
    )
    num_forwards = contract.num_forwards
    stem = f"{task}_hyperflow_{width}x{height}_{duration_s}s_{num_forwards}fwd"
    logger.info(f"adapter {lora_path} at strength {strength}: {contract.identity()}")
    logger.info(f"{task}: {width}x{height}, {num_frames} frames, {num_forwards} forwards, partition {partition}/")

    output = run_warm_generation(
        pipeline, PROMPT, num_frames=num_frames, height=height, width=width, seed=SEED, **request
    )

    assert_hyperflow_applied(pipeline, num_forwards=num_forwards)

    total_s = log_timing_table(
        pipeline,
        stem,
        num_forwards=num_forwards,
        video_seconds=output.video_seconds,
        extra=f" | {task}, partition {partition}/, padded_len {pipeline.last_padded_len}",
    )

    expected_frames = align_num_frames(num_frames)
    frames = to_uint8_frames(output)
    check_output_sanity(frames, num_frames=expected_frames, height=height, width=width)
    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=output.video_seconds)
    check_av_sync(frames, output.audio, sampling_rate=output.sampling_rate, fps=MINIMAX_H3_FPS)

    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
    check_written_file(paths, expected_frames, height=height, width=width)

    alignment = clip_prompt_alignment(frames, PROMPT)
    logger.info(f"{stem} CLIP prompt alignment (RECORDED not gated): {alignment}")

    seconds_per_video_second = total_s / output.video_seconds
    assert seconds_per_video_second < MAX_S_PER_VIDEO_SECOND, (
        f"{seconds_per_video_second:.1f} s of compute per video second is base-model territory; "
        f"the {num_forwards}-forward schedule likely did not take effect"
    )

    # The sidecar carries what a table needs, so tabulating a sweep never reparses a log.
    record = {
        "task": task,
        # Two nodes can sweep into one artifact directory; without this a mixed table reads as single-source.
        "node": socket.gethostname(),
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "num_frames": expected_frames,
        "video_seconds": output.video_seconds,
        "num_forwards": num_forwards,
        "partition": partition,
        "padded_len": pipeline.last_padded_len,
        "mesh": list(pipeline.mesh_device.shape),
        "adapter": Path(lora_path).name,
        "adapter_identity": contract.identity(),
        "timings_s": {label: seconds for label, seconds in pipeline.last_timings},
        "total_compute_s": total_s,
        "s_per_video_second": seconds_per_video_second,
        "clip": alignment,
        "artifacts": {kind: str(path) for kind, path in paths.items()},
    }
    sidecar = artifacts / f"{stem}.json"
    sidecar.write_text(json.dumps(record, indent=2))
    logger.info(f"wrote {sidecar}")
