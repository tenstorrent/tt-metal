# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end `t2va`: a prompt in, an mp4 with a soundtrack out, at 768P for 5 s and 15 s.

There is no torch reference for a whole generation -- 50 layers over 37749 rows (109101 at 15 s) for
49 steps is not a CPU computation -- so correctness here is established by the tiers that do not need
one, in the order they catch things:

  tier 4  reference-free sanity: geometry, finiteness, range, spatial variance, inter-frame motion
          (`tests/models/wan2_2/common.py`), plus the audio analogue and an A/V sync check
          (`common_av.py`)
  tier 5  the written *file* re-decoded with ffmpeg -- so container, pixel format and frame count
          are checked, not just the tensor -- plus a temporal-seam score and a log-spectrum shape
  tier 6  CLIP prompt alignment in-process, and VBench dimensions in a separate interpreter --
          VBench pins numpy < 2 and transformers 4.33, so it cannot share this environment. Both
          env-gated and defaulting **on**.

Per-component numerics are gated elsewhere and are not repeated here: the conditioner in
`test_text_encoder_minimax_h3.py`, the DiT in `test_transformer_minimax_h3.py`, both VAEs in
`test_vae_*` and `test_audio_minimax_h3.py`, all at the 768P/5s working point. Nothing gates a
*component* at 15 s, so the 15 s case here is the only thing standing between a 15 s request and
whatever it produces -- which is why its tier-4 and tier-5 checks assert rather than log.

Artifacts are written to a stable path so the output can be *looked at*. Every numeric gate below
can pass on video that is visibly wrong, and the two failure modes whole-tensor statistics hide best
are both live here: seams at the video VAE's 4x7 = 28 spatial tiles and its 17-frame temporal chunks,
and temporal flicker from per-frame GroupNorm statistics or chunk-boundary stitching. Reading the artifact rubric
against the real frames is part of the gate, not an optional follow-up.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, align_num_frames
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..wan2_2.common import check_output_sanity
from .common import MESH_PARAMS
from .common_av import check_audio_sanity, check_av_sync, check_spatial_seams, log_spectral_flatness

# The production working point: 1344x768 at 16:9, 124 frames @ 24 fps (5.17 s), 50 scheduler
# steps -> 49 forwards.
HEIGHT, WIDTH = 768, 1344
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
SEED = 0

# The durations gated end to end. `align_num_frames(round(duration * MINIMAX_H3_FPS))` under the
# model's 17n + 5 rule gives 124 and 362, i.e. n = 7 and n = 21. The frame counts are written out
# rather than computed so the test ids are stable and greppable.
#
# 15s is here because it is the case the wider mesh exists for -- 107 latent frames pack to 109101
# rows, which is 13664 per device at SP=8 but only 3424 at SP=32 -- and because the perf suite has
# had a 15s row for a while with no correctness gate under it.
#
# **Only 5s is tier-6 gated.** `CLIP_THRESHOLD` and `VBENCH_THRESHOLDS` were calibrated with `PROMPT`
# as a matched pair at 124 frames; reusing those numbers at 362 would be asserting a bar nobody
# measured, and `imaging_quality` in particular has swung 0.6896 -> 0.4884 between two
# correct-looking scenes. The 15s case therefore *measures and logs* tier 6 and gates on tiers 4
# and 5, which are prompt- and duration-independent. Set the 15s bars from the logged numbers once
# there are a few runs to set them from.
DURATIONS = [
    pytest.param(NUM_FRAMES, id="5s"),
    pytest.param(362, id="15s"),
]

# Dense: a moving camera, a reflective wet surface, several independent light sources at
# different colour temperatures, volumetric haze, and foreground/background motion at different depths.
# Those are the things a video model is most likely to get wrong, and they are also what the artifact
# rubric reads best -- banding shows in the haze gradients, seams show in the reflections, and flicker
# shows in the neon.
# The gated prompt and the tier-6 thresholds are a **matched pair**. Both were calibrated together on
# this prompt (amendment 80); swapping one without recalibrating the other breaks the gate. Measured
# here: CLIP 37.37, VBench imaging_quality 0.6896.
#
# `imaging_quality` in particular is prompt-dependent, not just model-dependent: it is a no-reference
# IQA model that rewards sharp, well-lit frames. A dark rain-at-night scene scored **0.4884** against
# this same 0.64 bar (amendment 87) while looking entirely correct. So a showcase prompt belongs in a
# manual run, not in this constant.
PROMPT = (
    "A red fox trots across a snowy field at dawn, its breath visible in the cold air. "
    "The low sun throws long blue shadows behind it, and loose snow lifts from each footfall."
)


WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
DEFAULT_WEIGHTS = "/data/cglagovich/MiniMax-H3-diffusers"
ARTIFACT_ENV = "MINIMAX_H3_ARTIFACT_DIR"


def _weights_dir():
    base = os.environ.get(WEIGHTS_ENV, DEFAULT_WEIGHTS)
    missing = [p for p in ("transformer", "text_encoder", "vae", "audio_vae") if not os.path.isdir(f"{base}/{p}")]
    if missing:
        pytest.skip(f"MiniMax-H3 snapshot at {base} is missing {missing}; set {WEIGHTS_ENV}")
    return base


def _artifact_dir():
    directory = Path(os.environ.get(ARTIFACT_ENV) or Path.home() / "h3_t2va_artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _ffmpeg():
    """An ffmpeg executable: the system binary, else imageio-ffmpeg's bundled static build."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"],
            check=False,
            capture_output=True,
        )
        try:
            import imageio_ffmpeg
        except ImportError:
            return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    """`(1, 3, F, H, W)` in [0, 1] -> `(F, H, W, 3)` uint8, which is what the checkers take."""
    assert video.ndim == 5 and video.shape[0] == 1, f"unexpected video shape {tuple(video.shape)}"
    frames = video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    return frames.cpu().numpy()


def _write_artifacts(frames, audio, sampling_rate, directory: Path, stem: str = "t2va"):
    """Write the video, the soundtrack, and a muxed mp4. Returns the paths that were produced.

    `stem` names the files. It defaults to `t2va` so every existing caller and every recorded artifact
    path is unchanged; the fl2va gate passes its own so the two tasks' artifacts coexist in one
    directory rather than overwriting each other.
    """
    import wave

    paths = {}
    wav_path = directory / f"{stem}.wav"
    interleaved = np.ascontiguousarray(audio[0].T if audio.ndim == 3 else audio.T)
    pcm = np.clip(interleaved, -1.0, 1.0)
    with wave.open(str(wav_path), "wb") as handle:
        handle.setnchannels(pcm.shape[1])
        handle.setsampwidth(2)
        handle.setframerate(sampling_rate)
        handle.writeframes((pcm * 32767.0).astype("<i2").tobytes())
    paths["wav"] = wav_path
    logger.info(f"wrote {wav_path}")

    exe = _ffmpeg()
    if exe is None:
        logger.warning("no ffmpeg available; skipping mp4 and the file-level checks")
        return paths

    silent = directory / f"{stem}_silent.mp4"
    num_frames, height, width, _ = frames.shape
    subprocess.run(
        [
            exe,
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(MINIMAX_H3_FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            str(silent),
        ],
        input=frames.tobytes(),
        check=True,
        capture_output=True,
    )
    paths["silent_mp4"] = silent

    muxed = directory / f"{stem}.mp4"
    subprocess.run(
        [
            exe,
            "-y",
            "-v",
            "error",
            "-i",
            str(silent),
            "-i",
            str(wav_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(muxed),
        ],
        check=True,
        capture_output=True,
    )
    paths["mp4"] = muxed
    logger.info(f"wrote {muxed} and {silent}")
    return paths


def _probe_streams(path: Path) -> dict:
    """Per-stream duration from the written container, via ffprobe if it is available."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return {}
    import json as _json

    result = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,duration,nb_frames,sample_rate,channels,width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = {}
    for stream in _json.loads(result.stdout).get("streams", []):
        streams[stream.get("codec_type", "?")] = stream
    return streams


def _decoded_frames(path: Path, count: int) -> np.ndarray:
    """Sample `count` frames evenly out of the written file, as uint8 luma."""
    exe = _ffmpeg()
    if exe is None:
        return np.empty((0,))
    result = subprocess.run(
        [
            exe,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select='not(mod(n\\,{max(1, count)}))'",
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    probe = _probe_streams(path).get("video", {})
    height = int(probe.get("height") or HEIGHT)
    width = int(probe.get("width") or WIDTH)
    buffer = np.frombuffer(result.stdout, dtype=np.uint8)
    usable = (buffer.size // (height * width)) * height * width
    return buffer[:usable].reshape(-1, height, width)


# VBench dimensions and the bars, **calibrated on this model at this working point** (2026-08-04,
# seed 0, the fox prompt) rather than copied from LTX's 1088p set:
#
#   subject_consistency 0.9820   background_consistency 0.9831   motion_smoothness 0.9905
#   dynamic_degree 1.0           imaging_quality 0.6896
#
# For reference, LTX's calibrated 1088p bars are 0.92 / 0.93 / 0.955 / 1.0 / 0.645, which H3 clears on
# every dimension, so those values would gate nothing here.
#
# The margins below are generous because this is a **single-sample** calibration: one
# prompt, one seed. They are set to catch a broken pipeline, not to certify quality. `dynamic_degree`
# stays at 1.0 because over one video it is effectively binary -- either the clip has real motion or
# it does not, and "it does not" is the frozen-video failure.
VBENCH_THRESHOLDS = {
    "subject_consistency": 0.95,
    "background_consistency": 0.95,
    "motion_smoothness": 0.97,
    "dynamic_degree": 1.0,
    "imaging_quality": 0.64,
}
# Measured mean 37.05 over 8 evenly-spaced frames (min 36.16), which sits at wan2.2's ~37 baseline
# rather than LTX's ~31.3. 33.0 leaves ~4 points, matching the headroom LTX allows for run-to-run
# variance.
CLIP_THRESHOLD = 33.0
VBENCH_VENV_ENV = "MINIMAX_H3_VBENCH_PYTHON"
DEFAULT_VBENCH_PYTHON = "/data/kevinmi/vbench_env/bin/python"


def _run_vbench(video: Path, prompt: str, dimensions) -> dict[str, float]:
    """Score `video` by running VBench in its own interpreter, and return the dimension scores.

    VBench cannot live in `python_env`: it pins numpy < 2 and transformers 4.33, so installing it
    would downgrade numpy 2.2.6 -> 1.26.4 and transformers 5.12.1 -> 4.33.2, breaking `ttnn`'s numpy
    ABI and the Qwen3-VL reference the text-encoder gate depends on. It evaluates a *file* and needs
    nothing from this process, so it runs out-of-process instead of not at all.
    """
    interpreter = os.environ.get(VBENCH_VENV_ENV, DEFAULT_VBENCH_PYTHON)
    if not os.path.isfile(interpreter):
        pytest.skip(
            f"no VBench interpreter at {interpreter}; set {VBENCH_VENV_ENV}, or create it with "
            "`uv venv --python 3.10 <path> && uv pip install --python <path>/bin/python vbench decord "
            "'numpy==1.26.4' 'opencv-python-headless<4.11' 'setuptools<81'` (set RUN_VBENCH=0 to skip)"
        )
    runner = Path(__file__).with_name("vbench_runner.py")
    result = subprocess.run(
        [interpreter, str(runner), str(video), ",".join(dimensions), "--prompt", prompt],
        capture_output=True,
        text=True,
        timeout=5400,
    )
    marker = "VBENCH_JSON "
    line = next((l for l in result.stdout.splitlines() if l.startswith(marker)), None)
    if line is None:
        raise AssertionError(
            f"VBench produced no scores (exit {result.returncode}).\n"
            f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
        )
    import json as _json

    return _json.loads(line[len(marker) :])


def _clip_prompt_alignment(frames: np.ndarray, prompt: str, num_frames: int = 8) -> dict[str, float]:
    """Mean CLIP similarity between the prompt and evenly-spaced frames, x100.

    Uses `open_clip` (already in `python_env`) over the frames this test already decoded, so unlike
    the wan2.2 and LTX versions it needs no `decord`.
    """
    from PIL import Image

    from ...dataset_eval.clip_encoder import CLIPEncoder

    indices = np.linspace(0, frames.shape[0] - 1, num_frames).astype(int)
    encoder = CLIPEncoder()
    scores = [encoder.get_clip_score(prompt, Image.fromarray(frames[i])).item() * 100.0 for i in indices]
    return {"mean": float(np.mean(scores)), "min": float(min(scores)), "max": float(max(scores))}


def _temporal_seam_score(frames: np.ndarray, period: int) -> float:
    """Ratio of inter-frame delta *at* chunk boundaries to the delta everywhere else.

    The video VAE decodes in temporal chunks and cross-fades them on the host. If that stitching is
    wrong, the discontinuity concentrates at multiples of the chunk's pixel-frame count -- which a
    whole-video mean delta averages away entirely. ~1.0 means boundaries look like ordinary frames.
    """
    deltas = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2))
    if len(deltas) < 2 * period:
        return float("nan")
    index = np.arange(1, len(deltas) + 1)
    at_boundary = deltas[index % period == 0]
    elsewhere = deltas[index % period != 0]
    if not len(at_boundary) or not elsewhere.mean():
        return float("nan")
    return float(at_boundary.mean() / elsewhere.mean())


@pytest.mark.timeout(14400)
@pytest.mark.parametrize("num_frames", DURATIONS)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_PARAMS, indirect=["mesh_device", "device_params"])
def test_t2va_end_to_end(mesh_device, reset_seeds, num_frames):
    weights_dir = _weights_dir()
    artifacts = _artifact_dir()

    # On a multi-host mesh every rank runs this function. Device work is collective and must happen
    # on all of them, but the host-side artifacts must not: four ranks writing the same mp4 over NFS
    # race, and scoring it four times is three wasted VBench interpreters.
    is_root = not ttnn.using_distributed_env() or int(ttnn.distributed_context_get_rank()) == 0

    # Tier 6 is calibrated at 5s against `PROMPT` only; see `DURATIONS`.
    tier6_calibrated = num_frames == NUM_FRAMES

    run_vbench = os.environ.get("RUN_VBENCH", "1") in ("1", "true", "True")
    run_clip = os.environ.get("RUN_CLIP", "1") in ("1", "true", "True")

    # `MINIMAX_H3_PROMPT` overrides the gated prompt for a manual showcase run, which is where the
    # constant's own note says a showcase prompt belongs. The tier-6 thresholds are calibrated
    # *against that constant*, so an override forces CLIP and VBench off rather than reporting a
    # failure that says nothing about the model -- `imaging_quality` alone has swung 0.6896 -> 0.4884
    # between two correct-looking scenes. Tiers 4 and 5 are prompt-independent and still run.
    prompt = os.environ.get("MINIMAX_H3_PROMPT") or PROMPT
    if prompt is not PROMPT:
        logger.info("MINIMAX_H3_PROMPT override in use; disabling the prompt-calibrated tier-6 gates")
        run_vbench = run_clip = False
    # A missing dependency must report SKIPPED, never a silent pass: a quality gate that no-ops
    # reads green, which is worse than not having it. VBench is checked as an *interpreter* rather
    # than an import, because it does not live in this environment (see `_run_vbench`);
    # CLIP needs only `open_clip`, which is already here, and this test's own ffmpeg frames.
    if run_clip:
        pytest.importorskip("open_clip", reason="RUN_CLIP=1 but open_clip is not installed (set RUN_CLIP=0)")

    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights_dir)
    output = pipeline(
        prompt,
        num_frames=num_frames,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    expected_frames = align_num_frames(num_frames)
    logger.info(
        f"generated {output.num_frames} frames ({output.video_seconds:.3f} s) and "
        f"{output.audio_seconds:.3f} s of audio at {output.sampling_rate} Hz"
    )
    # Latency is recorded, not gated: this is a bringup run at current perf.
    for name, seconds in output.timings.items():
        logger.info(f"LATENCY {name}: {seconds:.1f} s")

    frames = _to_uint8_frames(output.video)

    # ---- tier 4: reference-free sanity, video then audio then the pairing ----
    check_output_sanity(frames, num_frames=expected_frames, height=HEIGHT, width=WIDTH)
    check_audio_sanity(
        output.audio,
        sampling_rate=output.sampling_rate,
        expected_seconds=expected_frames / MINIMAX_H3_FPS,
    )
    sync = check_av_sync(
        frames,
        output.audio,
        sampling_rate=output.sampling_rate,
        fps=MINIMAX_H3_FPS,
    )
    log_spectral_flatness(output.audio, sampling_rate=output.sampling_rate)

    # Spatial seams, at the tile geometry the VAE **actually used**. Asked of the VAE object rather
    # than re-derived: an earlier version of this recomputed `split_tiles(HEIGHT, 256, 32, ratio)`
    # with a hardcoded overlap of 32 against the real `DEFAULT_TILE_OVERLAP = 64`, which produced a
    # 4x6 grid instead of 4x7 and checked columns that are not tile boundaries at all. The gate
    # passed while measuring nothing. Duplicating a derivation is how that happens; asking the
    # object that owns it is how it does not.
    ratio = pipeline.vae_config.spatial_compression_ratio
    (y_starts, _, _), (x_starts, _, _) = pipeline.vae.decode_tile_grid(HEIGHT // ratio, WIDTH // ratio)
    check_spatial_seams(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])

    # ---- tier 5: the written file, not just the tensor ----
    # Root only, so the ranks do not race on the same paths. Tiers 4 and 6 are pure reads of tensors
    # every rank already holds, but tier 5 is *about* the file, so on a non-root rank there is
    # nothing to check rather than something being skipped.
    if not is_root:
        logger.info(f"rank {ttnn.distributed_context_get_rank()}: not root, skipping artifacts and scoring")
        assert sync["video_seconds"] > 0
        return

    stem = "t2va" if num_frames == NUM_FRAMES else f"t2va_{num_frames}f"
    paths = _write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
    if "mp4" in paths:
        streams = _probe_streams(paths["mp4"])
        if streams:
            logger.info(f"container streams: { {k: v.get('duration') for k, v in streams.items()} }")
            assert "video" in streams and "audio" in streams, f"muxed file is missing a stream: {list(streams)}"
            durations = {k: float(v["duration"]) for k, v in streams.items() if v.get("duration")}
            if {"video", "audio"} <= set(durations):
                skew = durations["audio"] - durations["video"]
                # AAC pads to a frame boundary, so allow a little more than the tensor-level check.
                assert abs(skew) < 0.15, f"muxed A/V skew {skew:+.3f} s"
                logger.info(f"muxed A/V skew: {skew:+.4f} s")

        decoded = _decoded_frames(paths["mp4"], count=1)
        if decoded.size:
            assert (
                decoded.shape[0] >= expected_frames - 1
            ), f"the written mp4 decodes to {decoded.shape[0]} frames, expected ~{expected_frames}"
            # The VAE's temporal chunk covers clip_length (17) pixel frames.
            seam = _temporal_seam_score(decoded, period=17)
            logger.info(f"temporal seam score at the 17-frame chunk period: {seam:.3f} (1.0 = no seam)")
            if np.isfinite(seam):
                assert seam < 3.0, (
                    f"inter-frame delta at chunk boundaries is {seam:.2f}x the delta elsewhere; "
                    "suspect temporal stitching (see the artifact rubric)"
                )

    # ---- tier 6: generative quality ----
    if run_clip:
        alignment = _clip_prompt_alignment(frames, prompt)
        bar = f"bar {CLIP_THRESHOLD}" if tier6_calibrated else "uncalibrated at this duration"
        logger.info(
            f"CLIP prompt alignment: mean={alignment['mean']:.2f} "
            f"min={alignment['min']:.2f} max={alignment['max']:.2f} ({bar})"
        )
        if tier6_calibrated:
            assert (
                alignment["mean"] >= CLIP_THRESHOLD
            ), f"CLIP mean {alignment['mean']:.2f} < {CLIP_THRESHOLD}; the video does not match the prompt"
    else:
        logger.info("RUN_CLIP=0, skipping the CLIP prompt-alignment gate")

    if run_vbench:
        assert "mp4" in paths, "RUN_VBENCH=1 needs the muxed mp4, which ffmpeg did not produce"
        scores = _run_vbench(paths["mp4"], prompt, VBENCH_THRESHOLDS.keys())
        for dimension, threshold in VBENCH_THRESHOLDS.items():
            # A requested dimension with no returned score is an *ungated* dimension, not a pass.
            assert dimension in scores, f"VBench returned no score for {dimension}"
            bar = f"bar {threshold}" if tier6_calibrated else "uncalibrated at this duration"
            logger.info(f"VBench {dimension} = {scores[dimension]:.4f} ({bar})")
        if tier6_calibrated:
            failures = [
                f"{d} = {scores[d]:.4f} < {threshold:.4f}"
                for d, threshold in VBENCH_THRESHOLDS.items()
                if scores[d] < threshold
            ]
            assert not failures, "VBench below threshold: " + "; ".join(failures)
    else:
        logger.info("RUN_VBENCH=0, skipping the VBench gate")

    if not tier6_calibrated:
        logger.info(
            f"tier 6 measured but not gated at {num_frames} frames: the CLIP and VBench bars are "
            f"calibrated against PROMPT at {NUM_FRAMES} frames only. Set the bars for this duration "
            "from the numbers above once there are enough runs to separate signal from the +/-8% "
            "run-to-run variance."
        )

    logger.info(f"artifacts in {artifacts}: {sorted(p.name for p in artifacts.iterdir())}")
    logger.info(
        "REMINDER: read the artifact rubric against these frames -- the seam and flicker scores above "
        "are statistics, and only looking at the output catches what they average away"
    )
    assert sync["video_seconds"] > 0
