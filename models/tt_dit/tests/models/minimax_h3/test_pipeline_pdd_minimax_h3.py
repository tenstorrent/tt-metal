# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` e2e with a FastVideo PDD export (FastH3 OmniRef PDD-8): 8 fused forwards.

`MINIMAX_H3_PDD_EXPORT` is the export directory (``transformer_ref/`` + ``fastvideo_inference.json``),
`MINIMAX_H3_MODEL_PATH` the base model; `MINIMAX_H3_PDD_REFERENCE` an optional reference image (default a fractal)
and `MINIMAX_H3_PDD_PROMPT` an optional prompt to go with it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from PIL import Image

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.events import profiler_event_callback
from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference
from ....pipelines.minimax_h3.pipeline_minimax_h3_pdd import MiniMaxH3PDDPipeline
from .common import create_fractal_image
from .common_av import (
    artifact_dir,
    check_audio_sanity,
    log_pipeline_perf,
    log_quality,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)
from .test_pipeline_ref2va_minimax_h3 import _L1_SMALL, MESHES

FPS = 24
SEED = 0
PROMPT = os.environ.get(
    "MINIMAX_H3_PDD_PROMPT",
    "The person in the picture walks through a sunlit market, smiling and talking to the camera.",
)


def _export_dir() -> Path:
    path = os.environ.get("MINIMAX_H3_PDD_EXPORT")
    if not path or not Path(path).is_dir():
        pytest.skip("set MINIMAX_H3_PDD_EXPORT to a PDD export directory")
    return Path(path)


def _reference() -> MiniMaxH3Reference:
    path = os.environ.get("MINIMAX_H3_PDD_REFERENCE")
    image = Image.open(path).convert("RGB") if path else create_fractal_image(1024, 1024)
    return MiniMaxH3Reference(image=image)


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(
    ("num_frames", "height", "width"),
    [
        pytest.param(124, 768, 1344, id="5s_768p"),
        pytest.param(362, 768, 1344, id="15s_768p"),
    ],
)
@pytest.mark.parametrize("attention", ["vsa", "dense"])
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_pdd_ref2va_end_to_end(attention, num_frames, height, width, mesh_device, reset_seeds):
    """`vsa` is the export's trained attention; `dense` drops the trained gates and runs every block, as a baseline."""
    export = _export_dir()
    # The init warmup walks the whole ref2va ladder; its top rungs exceed the sparse op's 64 K pieces per
    # shard on 4x8, and the compile pass below already warms the one rung this case lands on.
    pipeline = MiniMaxH3PDDPipeline.create_pipeline(
        warmup=False,
        vae_output_type="float",
        mesh_device=mesh_device,
        weights_dir=weights_dir("text_encoder", "vae", "audio_vae"),
        task="ref2va",
        pdd_export_dir=export,
        vsa_config=MiniMaxH3PDDPipeline.export_vsa_config(export) if attention == "vsa" else None,
    )
    steps = pipeline.pdd.num_forwards + 1
    kwargs = dict(references=[_reference()], num_frames=num_frames, height=height, width=width, seed=SEED)

    # The schedule is fixed by the export, so the compile pass runs the real step count.
    with pipeline.quiet():
        pipeline(PROMPT, num_inference_steps=steps, **kwargs)
    profiler = BenchmarkProfiler()
    with profiler("run", iteration=0):
        output = pipeline(PROMPT, num_inference_steps=steps, on_event=profiler_event_callback(profiler, 0), **kwargs)
        ttnn.synchronize_device(pipeline.mesh_device)

    assert output.video.shape == (1, 3, num_frames, height, width), tuple(output.video.shape)
    assert torch.isfinite(output.video).all() and torch.isfinite(output.audio).all()

    frames = to_uint8_frames(output)
    directory = artifact_dir("h3_pdd_artifacts")
    stem = f"pdd_ref2va_{attention}_{num_frames}f"
    for index in (0, num_frames // 2, num_frames - 1):
        Image.fromarray(frames[index]).save(directory / f"{stem}_frame_{index}.png")
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, directory, stem=stem)
    log_quality(f"{stem}: artifacts in {directory} ({sorted(paths)})")

    log_pipeline_perf(
        profiler,
        label=f"pdd ref2va {attention} {num_frames}f",
        pipeline=pipeline,
        num_forwards=pipeline.pdd.num_forwards,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=FPS,
        num_inference_steps=steps,
        extra_lines=(f"L1 Small Size: {_L1_SMALL}",),
    )
    check_audio_sanity(
        output.audio, sampling_rate=output.sampling_rate, expected_seconds=num_frames / FPS, tolerance_seconds=0.05
    )
