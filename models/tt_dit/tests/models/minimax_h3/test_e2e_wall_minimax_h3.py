# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Prompt-to-files wall clock for one t2va working point, after warmup, with a determinism check.

Unlike `test_t2va_performance`, which times only the pipeline call and leaves weight loads and file
writing out, this times what a user waits for: from the prompt to the video and audio files on disk.
Each measured run writes `video.y4m` (8-bit 4:2:0) and `audio.wav` (16-bit PCM) and is timed to their
close; muxing an mp4 is timed separately because it is a delivery-format choice, not generation.

Runs: `H3_E2E_WARMUP_STEPS` (default 3, 0 = a full warmup at the measured step count) for a quiet
warmup at the exact shape, then `H3_E2E_RUNS` (default 2) measured generations at the same seed. Run 2+
is compared with run 1 value for value (video bytes and audio samples): the pipeline must be
deterministic for a seed. The CLIP prompt alignment of run 1 is scored; `RUN_VBENCH=1` adds VBench.
`H3_E2E_INIT_WARMUP=0` skips the serving warmup of every bucket rung.
`H3_E2E_OUT` appends one JSON row per run (stage seconds, per-step seconds, total, determinism).

    TT_DIT_CACHE_DIR=~/tt_dit_cache pytest models/tt_dit/tests/models/minimax_h3/test_e2e_wall_minimax_h3.py -s
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.events import DenoiseStep, profiler_event_callback
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....pipelines.minimax_h3.policy import align_num_frames, get_num_frames
from .common import GALAXY_MESHES
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    _ffmpeg,
    artifact_dir,
    assert_generation_ok,
    clip_prompt_alignment,
    frames_for_export,
    is_host,
    run_vbench,
    weights_dir,
)

NUM_INFERENCE_STEPS = int(os.environ.get("H3_E2E_STEPS", 50))
SEED = int(os.environ.get("H3_E2E_SEED", 0))
ASPECT_RATIO = (16, 9)
DURATION_S = float(os.environ.get("H3_E2E_DURATION_S", 15))
VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
)


def write_y4m_wav(video_yuv: np.ndarray, audio: np.ndarray, sampling_rate: int, directory: Path) -> dict[str, Path]:
    """`video.y4m` (C420jpeg-free 4:2:0 planes as the pipeline produced them) and `audio.wav`."""
    import wave

    frames, planar_height, width = video_yuv.shape
    height = planar_height * 2 // 3
    y4m = directory / "video.y4m"
    with open(y4m, "wb") as handle:
        handle.write(f"YUV4MPEG2 W{width} H{height} F{MINIMAX_H3_FPS}:1 Ip A1:1 C420\n".encode())
        payload = np.ascontiguousarray(video_yuv, dtype=np.uint8)
        header = b"FRAME\n"
        for frame in payload:
            handle.write(header)
            handle.write(frame.tobytes())
    wav = directory / "audio.wav"
    interleaved = np.ascontiguousarray(audio[0].T if audio.ndim == 3 else audio.T)
    pcm = (np.clip(interleaved, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(wav), "wb") as handle:
        handle.setnchannels(pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sampling_rate)
        handle.writeframes(pcm.tobytes())
    return {"y4m": y4m, "wav": wav}


def rgb_frames_from_y4m(y4m: Path, count: int, total: int, height: int, width: int) -> np.ndarray:
    """`count` evenly spaced frames of the lossless y4m as (count, H, W, 3) uint8 RGB, decoded by ffmpeg
    (the pipeline's planes are BT.601 yuv420p, ffmpeg's default for this conversion) -- what CLIP
    should see, rather than the gray frames `decoded_frames` returns."""
    exe = _ffmpeg()
    indices = np.linspace(0, total - 1, count).astype(int)
    select = "+".join(f"eq(n\\,{i})" for i in indices)
    result = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-i",
            str(y4m),
            "-vf",
            f"select='{select}'",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    frames = np.frombuffer(result.stdout, dtype=np.uint8).reshape(-1, height, width, 3)
    assert frames.shape[0] == count, f"decoded {frames.shape[0]} of {count} sampled frames"
    return frames


def mux_mp4(paths: dict[str, Path], directory: Path) -> Path | None:
    exe = _ffmpeg()
    if exe is None:
        return None
    mp4 = directory / "video.mp4"
    subprocess.run(
        [
            exe,
            "-y",
            "-v",
            "error",
            "-i",
            str(paths["y4m"]),
            "-i",
            str(paths["wav"]),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(mp4),
        ],
        check=True,
        capture_output=True,
    )
    return mp4


@pytest.mark.timeout(14400)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_e2e_wall(mesh_device, reset_seeds):
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    prompt = os.environ.get("H3_E2E_PROMPT", CALIBRATED_FOX_PROMPT)
    runs = int(os.environ.get("H3_E2E_RUNS", 2))
    warmup_steps = int(os.environ.get("H3_E2E_WARMUP_STEPS", 3)) or NUM_INFERENCE_STEPS
    out_rows = os.environ.get("H3_E2E_OUT")
    label = os.environ.get("H3_E2E_LABEL", "")

    height, width = resolve_canvas_size(*ASPECT_RATIO)
    num_frames = get_num_frames(DURATION_S)
    expected_frames = align_num_frames(num_frames)
    gen = dict(num_frames=num_frames, height=height, width=width, seed=SEED)

    t_create = time.perf_counter()
    # H3_E2E_INIT_WARMUP=0 skips the serving warmup of every bucket rung (~40 min); the quiet warmup
    # below still compiles everything this shape runs, and the program-cache check proves it.
    init_warmup = os.environ.get("H3_E2E_INIT_WARMUP", "1") == "1"
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights, vae_output_type="yuv420", warmup=init_warmup
    )
    create_s = time.perf_counter() - t_create

    t_warm = time.perf_counter()
    with pipeline.quiet():
        pipeline(prompt, num_inference_steps=warmup_steps, **gen)
    ttnn.synchronize_device(pipeline.mesh_device)
    warmup_s = time.perf_counter() - t_warm

    root = artifact_dir("h3_e2e_wall")
    reference = None
    rows = []
    for run in range(runs):
        directory = root / f"run{run}"
        directory.mkdir(parents=True, exist_ok=True)
        profiler = BenchmarkProfiler()
        sections = profiler_event_callback(profiler, 0)
        step_marks: list[float] = []

        def on_event(event):
            if isinstance(event, DenoiseStep):
                step_marks.append(time.perf_counter())
            sections(event)

        t0 = time.perf_counter()
        output = pipeline(prompt, num_inference_steps=NUM_INFERENCE_STEPS, on_event=on_event, **gen)
        ttnn.synchronize_device(pipeline.mesh_device)
        t_generated = time.perf_counter()
        paths = {}
        if is_host():
            video = frames_for_export(output)
            audio = output.audio.float().cpu().numpy()
            paths = write_y4m_wav(video, audio, output.sampling_rate, directory)
        t_files = time.perf_counter()
        mp4 = mux_mp4(paths, directory) if is_host() else None
        t_mp4 = time.perf_counter()

        missed = {name: count for name, count in pipeline.last_program_cache_misses.items() if count}
        assert not missed, f"run {run} compiled programs after warmup, so it is not warm: {missed}"
        assert_generation_ok(output, expected_frames)

        stages = {}
        for name in ("encoder", "denoising", "vae", "audio"):
            if profiler.contains_step(name, 0):
                stages[name] = round(profiler.get_duration(name, 0), 3)
        step_s = [b - a for a, b in zip(step_marks, step_marks[1:])]
        row = {
            "label": label,
            "run": run,
            "duration_s": DURATION_S,
            "frames": expected_frames,
            "canvas": f"{width}x{height}",
            "steps": NUM_INFERENCE_STEPS,
            "seed": SEED,
            "seq_padded": pipeline.last_seq_len.padded if pipeline.last_seq_len else None,
            "create_s": round(create_s, 1),
            "warmup_s": round(warmup_s, 1),
            "generate_s": round(t_generated - t0, 3),
            "files_s": round(t_files - t_generated, 3),
            "prompt_to_files_s": round(t_files - t0, 3),
            "mp4_mux_s": round(t_mp4 - t_files, 3),
            **{f"stage_{k}_s": v for k, v in stages.items()},
            "other_s": round((t_generated - t0) - sum(stages.values()), 3),
            "step_median_s": round(statistics.median(step_s), 4) if step_s else None,
            "step_min_s": round(min(step_s), 4) if step_s else None,
            "step_max_s": round(max(step_s), 4) if step_s else None,
        }

        current = (np.asarray(frames_for_export(output)), output.audio.float().cpu())
        if reference is None:
            reference = current
            row["deterministic"] = None
        else:
            video_diff = int(np.count_nonzero(reference[0] != current[0]))
            audio_diff = int(torch.count_nonzero(reference[1] != current[1]).item())
            row["video_bytes_differ"] = video_diff
            row["audio_samples_differ"] = audio_diff
            row["deterministic"] = video_diff == 0 and audio_diff == 0

        if run == 0 and is_host() and paths:
            rgb = rgb_frames_from_y4m(paths["y4m"], 8, expected_frames, height, width)
            try:
                clip = clip_prompt_alignment(rgb, prompt, num_frames=8)
                row["clip_mean"] = round(clip["mean"], 3)
                row["clip_min"] = round(clip["min"], 3)
            except ImportError as exc:
                logger.warning(f"CLIP not scored: {exc}")
            if mp4 is not None and os.environ.get("RUN_VBENCH", "0") in ("1", "true", "True"):
                scores = run_vbench(mp4, prompt, VBENCH_DIMENSIONS)
                row.update({f"vbench_{k}": round(v, 4) for k, v in scores.items()})

        rows.append(row)
        if is_host():
            logger.info("H3_E2E " + json.dumps(row))
            if out_rows:
                with open(out_rows, "a") as handle:
                    handle.write(json.dumps(row) + "\n")

    pipeline.release_traces()
    for row in rows[1:]:
        assert row["deterministic"], f"run {row['run']} differs from run 0: {row}"
