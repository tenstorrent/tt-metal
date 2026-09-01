# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Reference-free A/V sanity checks plus the scaffolding shared by the MiniMax-H3 e2e gates.
A/V sync is checked structurally (duration/ordering); envelope-vs-motion correlation is diagnostic only."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS


def check_audio_sanity(audio, *, sampling_rate, expected_seconds, tolerance_seconds=0.05):
    """Guard against a soundtrack that is silent, clipped, constant, or the wrong length."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.ndim == 3:
        assert audio.shape[0] == 1, f"expected a single generation, got batch {audio.shape[0]}"
        audio = audio[0]
    assert audio.ndim == 2, f"expected (channels, samples), got shape {audio.shape}"

    channels, samples = audio.shape
    assert channels == 2, f"H3 generates stereo; got {channels} channel(s)"
    assert np.isfinite(audio).all(), "soundtrack contains NaN/Inf"

    seconds = samples / sampling_rate
    assert abs(seconds - expected_seconds) <= tolerance_seconds, (
        f"soundtrack is {seconds:.3f} s against {expected_seconds:.3f} s of video "
        f"(off by {seconds - expected_seconds:+.3f} s, tolerance {tolerance_seconds:.3f} s)"
    )

    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((audio.astype(np.float64) ** 2).mean()))
    assert peak > 1e-3, f"soundtrack is silent (peak {peak:.2e})"
    assert rms > 1e-4, f"soundtrack is near-silent (rms {rms:.2e})"
    for index in range(channels):
        channel_std = float(audio[index].std())
        assert channel_std > 1e-4, f"channel {index} is constant (std {channel_std:.2e})"
    # Widespread clipping means the denormalization is wrong, not that the mix is loud.
    clipped = float((np.abs(audio) >= 0.999).mean())
    assert clipped < 0.01, f"{clipped:.1%} of samples are at full scale; suspect a scaling error"

    logger.info(
        f"Audio sanity OK: {channels}ch {seconds:.3f} s @ {sampling_rate} Hz, "
        f"peak={peak:.3f}, rms={rms:.4f}, clipped={clipped:.3%}"
    )


def check_av_sync(frames, audio, *, sampling_rate, fps, tolerance_seconds=0.05):
    """Durations must agree and stereo channels must differ -- identical channels mean
    `unpack_audio_tokens` collapsed the channel-major layout. Envelope-motion lag is logged only."""
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    frames = np.asarray(frames)
    audio = np.asarray(audio)
    if audio.ndim == 3:
        audio = audio[0]

    num_frames = frames.shape[0]
    video_seconds = num_frames / fps
    audio_seconds = audio.shape[-1] / sampling_rate
    assert num_frames > 1, f"need more than one frame to talk about sync, got {num_frames}"
    assert audio.shape[-1] > sampling_rate // fps, "soundtrack is shorter than a single frame"
    assert abs(video_seconds - audio_seconds) <= tolerance_seconds, (
        f"video is {video_seconds:.3f} s but audio is {audio_seconds:.3f} s "
        f"(off by {audio_seconds - video_seconds:+.3f} s)"
    )

    if audio.shape[0] == 2:
        assert not np.allclose(audio[0], audio[1]), "stereo channels are identical; suspect audio row unpacking"

    # Diagnostic only.
    motion = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2, 3))
    samples_per_frame = audio.shape[-1] / num_frames
    envelope = np.array(
        [
            np.abs(audio[:, int(i * samples_per_frame) : int((i + 1) * samples_per_frame)]).mean()
            for i in range(1, num_frames)
        ]
    )
    lag_frames = 0
    if motion.std() > 1e-8 and envelope.std() > 1e-8:
        m = (motion - motion.mean()) / motion.std()
        e = (envelope - envelope.mean()) / envelope.std()
        correlation = np.correlate(m, e, mode="full") / len(m)
        lag_frames = int(np.argmax(correlation)) - (len(m) - 1)
        logger.info(
            f"A/V envelope-motion best lag: {lag_frames:+d} frames ({lag_frames / fps:+.3f} s), "
            f"peak r={correlation.max():.3f} (diagnostic, not asserted)"
        )

    logger.info(
        f"A/V sync OK: video {video_seconds:.3f} s / {num_frames} frames @ {fps} fps, "
        f"audio {audio_seconds:.3f} s @ {sampling_rate} Hz, delta {audio_seconds - video_seconds:+.4f} s"
    )
    return {"video_seconds": video_seconds, "audio_seconds": audio_seconds, "lag_frames": lag_frames}


def check_spatial_seams(frames, *, vertical_boundaries, horizontal_boundaries, max_ratio=2.0):
    """Gradient energy at the VAE's tile boundaries vs everywhere else (~1.0 = no seam; whole-frame means hide seams)."""
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames).astype(np.float32)
    if frames.ndim == 4:
        frames = frames.mean(axis=-1)

    def ratio(gradient, boundaries):
        inside = np.array([b for b in boundaries if 1 <= b < len(gradient) - 1], dtype=int)
        if not len(inside):
            return float("nan")
        # Exclude +-2 px around each boundary so a smeared seam cannot inflate its own baseline.
        baseline = np.ones(len(gradient), dtype=bool)
        for b in inside:
            baseline[max(0, b - 2) : b + 3] = False
        if not baseline.any() or gradient[baseline].mean() == 0:
            return float("nan")
        return float(gradient[inside].mean() / gradient[baseline].mean())

    column_gradient = np.abs(np.diff(frames, axis=2)).mean(axis=(0, 1))
    row_gradient = np.abs(np.diff(frames, axis=1)).mean(axis=(0, 2))
    vertical = ratio(column_gradient, vertical_boundaries)
    horizontal = ratio(row_gradient, horizontal_boundaries)

    logger.info(
        f"Spatial seam ratios (1.0 = no seam): vertical {vertical:.3f} at x={list(vertical_boundaries)}, "
        f"horizontal {horizontal:.3f} at y={list(horizontal_boundaries)}"
    )
    for name, value in (("vertical", vertical), ("horizontal", horizontal)):
        if np.isfinite(value):
            assert value < max_ratio, (
                f"{name} tile-boundary gradient is {value:.2f}x the surrounding image; "
                "suspect the tile blend extent or per-tile normalization (artifact rubric: seams)"
            )
    return {"vertical": vertical, "horizontal": horizontal}


def log_spectral_flatness(audio, *, sampling_rate, num_bands=64):
    """Log-spectrum shape, logged only: white noise drives flatness toward 1.0, a stuck tone toward 0."""
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()
    audio = np.asarray(audio)
    if audio.ndim == 3:
        audio = audio[0]
    mono = audio.mean(axis=0).astype(np.float64)

    window = 2048
    hop = window // 2
    frames = [mono[i : i + window] for i in range(0, max(1, len(mono) - window), hop)]
    if not frames:
        return {"flatness": float("nan")}
    spectrum = np.abs(np.fft.rfft(np.stack(frames) * np.hanning(window), axis=-1)) ** 2
    power = spectrum.mean(axis=0)[1:]
    flatness = float(np.exp(np.log(power + 1e-20).mean()) / (power.mean() + 1e-20))
    band_edges = np.linspace(0, len(power), num_bands + 1).astype(int)
    bands = np.array([power[a:b].mean() for a, b in zip(band_edges[:-1], band_edges[1:]) if b > a])
    logger.info(
        f"Audio log-spectrum: flatness={flatness:.4f}, "
        f"band dB range=[{10 * np.log10(bands.min() + 1e-20):.1f}, {10 * np.log10(bands.max() + 1e-20):.1f}]"
    )
    return {"flatness": flatness}


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_artifacts(frames, audio, sampling_rate, directory: Path, stem: str = "t2va"):
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
            # No -shortest. Video is quantised to 1/24 s and audio to 1/40 s (latent rate), so which
            # stream is shorter depends on the duration (+8 ms at 5 s, 0 at 10 s, -8 ms at 15 s) and
            # -shortest would clip whichever it is. That sub-frame trim is not what shortened the
            # decoded frame count -- an mp4 edit list from AAC priming is, see `decoded_frames` --
            # but there is no reason to let the muxer discard either stream's tail either.
            str(muxed),
        ],
        check=True,
        capture_output=True,
    )
    paths["mp4"] = muxed
    logger.info(f"wrote {muxed} and {silent}")
    return paths


def probe_streams(path: Path) -> dict:
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


# Only reached when ffprobe is unavailable.
_FALLBACK_HEIGHT = 768
_FALLBACK_WIDTH = 1344


def decoded_frames(path: Path, count: int, height: int = 0, width: int = 0) -> np.ndarray:
    """Sample `count` frames evenly out of the written file, as uint8 luma.

    `height`/`width` are the caller's known canvas. They matter: the frame count is derived by
    dividing the raw byte stream by the frame size, and the only other source for that size is
    `probe_streams`, which needs `ffprobe` -- absent wherever ffmpeg comes from imageio_ffmpeg.
    Without them this fell back to a hardcoded 768x1344, which silently miscounts every canvas
    with a different pixel total (4:3 read 94 frames for a correct 124-frame file). 16:9 and 21:9
    both happen to be 1032192 px, so the constant looked fine until a sweep left that budget.
    """
    exe = _ffmpeg()
    if exe is None:
        return np.empty((0,))
    result = subprocess.run(
        [
            exe,
            "-v",
            "error",
            # Count what the file STORES. Muxing AAC writes an mp4 edit list (the codec's priming
            # shifts the timeline), and a decoder honouring it drops frames at the edges: a 362-frame
            # write reads back as 362 stored / 361 in-container / 359 after the edit list. The gate
            # asks "were the frames written", so the edit list must not be applied here.
            "-ignore_editlist",
            "1",
            "-i",
            str(path),
            # `select` with count<=1 is a no-op (`not(mod(n,1))` is always true) but combined with
            # `-vsync 0` on an edit-list-shifted timeline it silently drops frames at the edges --
            # 2 of them on a muxed 15 s clip. Only ask for the filter when actually subsampling.
            *(["-vf", f"select='not(mod(n\\,{count}))'", "-vsync", "0"] if count > 1 else []),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    probe = probe_streams(path).get("video", {})
    height = int(probe.get("height") or height or _FALLBACK_HEIGHT)
    width = int(probe.get("width") or width or _FALLBACK_WIDTH)
    buffer = np.frombuffer(result.stdout, dtype=np.uint8)
    usable = (buffer.size // (height * width)) * height * width
    return buffer[:usable].reshape(-1, height, width)


VBENCH_PYTHON = Path.home() / "vbench_env" / "bin" / "python"


def run_vbench(video: Path, prompt: str, dimensions) -> dict[str, float]:
    """Run VBench in its own interpreter (~/vbench_env): it pins numpy < 2 / transformers 4.33."""
    if not VBENCH_PYTHON.is_file():
        pytest.skip(
            f"no VBench interpreter at {VBENCH_PYTHON}; create it with "
            "`uv venv --python 3.10 ~/vbench_env && uv pip install --python ~/vbench_env/bin/python "
            "vbench decord 'numpy==1.26.4' 'opencv-python-headless<4.11' 'setuptools<81'`"
        )
    interpreter = str(VBENCH_PYTHON)
    runner = Path(__file__).with_name("tools") / "vbench_runner.py"
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


def clip_prompt_alignment(frames: np.ndarray, prompt: str, num_frames: int = 8) -> dict[str, float]:
    """Mean CLIP similarity between the prompt and evenly-spaced frames, x100."""
    from PIL import Image

    from ...dataset_eval.clip_encoder import CLIPEncoder

    indices = np.linspace(0, frames.shape[0] - 1, num_frames).astype(int)
    encoder = CLIPEncoder()
    scores = [encoder.get_clip_score(prompt, Image.fromarray(frames[i])).item() * 100.0 for i in indices]
    return {"mean": float(np.mean(scores)), "min": float(min(scores)), "max": float(max(scores))}


def temporal_seam_score(frames: np.ndarray, period: int) -> float:
    """Inter-frame delta at temporal chunk boundaries vs elsewhere (~1.0 = no seam)."""
    deltas = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2))
    if len(deltas) < 2 * period:
        return float("nan")
    index = np.arange(1, len(deltas) + 1)
    at_boundary = deltas[index % period == 0]
    elsewhere = deltas[index % period != 0]
    if not len(at_boundary) or not elsewhere.mean():
        return float("nan")
    return float(at_boundary.mean() / elsewhere.mean())


# ------------------------------------------------------------------ shared e2e gate scaffolding

# Matched pair with the tier-6 bars (CLIP 37.37, imaging_quality 0.6896); imported by fl2va so it cannot drift.
CALIBRATED_FOX_PROMPT = (
    "A red fox trots across a snowy field at dawn, its breath visible in the cold air. "
    "The low sun throws long blue shadows behind it, and loose snow lifts from each footfall."
)


def weights_dir(*required_subdirs: str) -> Path:
    """The snapshot dir from MINIMAX_H3_MODEL_PATH; skips when it or a required partition is missing."""
    root = os.environ.get("MINIMAX_H3_MODEL_PATH", "")
    if not root or not Path(root).is_dir():
        pytest.skip("set MINIMAX_H3_MODEL_PATH to a MiniMax-H3 diffusers snapshot")
    directory = Path(root)
    missing = [name for name in required_subdirs if not (directory / name).is_dir()]
    if missing:
        pytest.skip(f"MiniMax-H3 snapshot at {directory} is missing {missing}")
    return directory


def artifact_dir(name: str) -> Path:
    """The gate's artifact directory `~/{name}`, created if absent."""
    directory = Path.home() / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_warm_generation(pipeline, prompt: str, *, seed: int, **gen_kwargs):
    """Warmup then the timed generation with identical kwargs; asserts padded-length agreement (programs are keyed on it)."""
    pipeline.warmup(prompt=prompt, **gen_kwargs)
    warm_padded_len = pipeline.last_padded_len

    # The warmup is a full generation too, so its stage timings are the *cold* numbers -- program
    # compilation, per-shape conv3d blocking and buffer allocation all land in them. Log them
    # before the measured call overwrites `last_timings`. Not gated: cold cost depends on what the
    # weight/JIT caches already held, so it is a diagnostic, not a bar.
    cold_rows = list(pipeline.last_timings)
    if cold_rows:
        cold_total = sum(seconds for _, seconds in cold_rows)
        logger.info("WARMUP (cold: program compile + buffer alloc included, not a perf target)")
        for row_label, seconds in cold_rows:
            share = 100 * seconds / cold_total if cold_total else 0.0
            logger.info(f"  {row_label:<18} {seconds:8.1f} s  ({share:4.1f} %)")
        logger.info(f"  {'Total (warmup)':<18} {cold_total:8.1f} s")

    ttnn.synchronize_device(pipeline.mesh_device)
    if ttnn.using_distributed_env():
        ttnn.distributed_context_barrier()

    output = pipeline(prompt, seed=seed, **gen_kwargs)

    assert pipeline.last_padded_len == warm_padded_len, (
        f"warmup ran at padded_len {warm_padded_len} but the measured call ran at "
        f"{pipeline.last_padded_len}; this number is not warm"
    )
    return output


def log_timing_table(pipeline, label: str, num_forwards: int, video_seconds: float, expected_total_s=None, extra=""):
    """The MEASUREMENT block; `expected_total_s`, when given, asserts the total. Returns the total."""
    rows = pipeline.last_timings
    total = sum(seconds for _, seconds in rows)
    shape = tuple(pipeline.mesh_device.shape)
    logger.info(
        f"MEASUREMENT {label} fully warm | mesh {shape[0]}x{shape[1]} Blackhole, "
        f"TP={pipeline.tp_factor} axis {pipeline.tp_axis} / SP={pipeline.sp_factor} axis {pipeline.sp_axis}, "
        f"{pipeline.ccl_manager.topology}, {pipeline.ccl_manager.num_links} links{extra} "
        f"| warm window: one full warmup generation at this shape, prepares and export excluded"
    )
    for row_label, seconds in rows:
        logger.info(f"  {row_label:<18} {seconds:8.1f} s  ({100 * seconds / total:4.1f} %)")
    logger.info(f"  {'Total (compute)':<18} {total:8.1f} s")
    denoise = dict(rows).get("Denoise")
    if denoise:
        logger.info(
            f"  per forward        {denoise / num_forwards * 1000:8.1f} ms  "
            f"({num_forwards} forwards over {denoise:.1f} s)"
        )
    logger.info(f"  realtime factor    {total / video_seconds:8.1f} x  (compute / video seconds)")
    if expected_total_s is not None:
        assert (
            total < expected_total_s
        ), f"fully-warm total {total:.1f} s exceeds the {expected_total_s:.0f} s floor bar"
    return total


def to_uint8_frames(output) -> np.ndarray:
    """`(1, 3, F, H, W)` [0, 1] -> `(F, H, W, 3)` uint8; clamp + round first -- a bare astype wraps out-of-range values."""
    video = output.video
    assert video.ndim == 5 and video.shape[0] == 1, f"unexpected video shape {tuple(video.shape)}"
    frames = video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    return frames.cpu().numpy()


def check_written_file(paths: dict, expected_frames: int, seam_period: int = 17, height: int = 0, width: int = 0):
    """Gate the written *file* (streams, A/V skew, frame count, temporal seam); silently no-ops without an mp4."""
    if "mp4" not in paths:
        return
    streams = probe_streams(paths["mp4"])
    if streams:
        logger.info(f"container streams: { {k: v.get('duration') for k, v in streams.items()} }")
        assert "video" in streams and "audio" in streams, f"muxed file is missing a stream: {list(streams)}"
        durations = {k: float(v["duration"]) for k, v in streams.items() if v.get("duration")}
        if {"video", "audio"} <= set(durations):
            skew = durations["audio"] - durations["video"]
            # AAC pads to a frame boundary, so allow a little more than the tensor-level check.
            assert abs(skew) < 0.15, f"muxed A/V skew {skew:+.3f} s"
            logger.info(f"muxed A/V skew: {skew:+.4f} s")

    decoded = decoded_frames(paths["mp4"], count=1, height=height, width=width)
    if decoded.size:
        assert (
            decoded.shape[0] >= expected_frames - 1
        ), f"the written mp4 decodes to {decoded.shape[0]} frames, expected ~{expected_frames}"
        # The VAE's temporal chunk covers clip_length (17) pixel frames.
        seam = temporal_seam_score(decoded, period=seam_period)
        logger.info(f"temporal seam score at the {seam_period}-frame chunk period: {seam:.3f} (1.0 = no seam)")
        if np.isfinite(seam):
            assert seam < 3.0, (
                f"inter-frame delta at chunk boundaries is {seam:.2f}x the delta elsewhere; "
                "suspect temporal stitching (see the artifact rubric)"
            )


def gate_clip(frames: np.ndarray, prompt: str, threshold: float, label: str):
    """CLIP prompt-alignment gate; skips only if `open_clip` is missing."""
    pytest.importorskip("open_clip", reason="the CLIP gate needs open_clip, which is not installed")
    alignment = clip_prompt_alignment(frames, prompt)
    logger.info(
        f"{label} CLIP prompt alignment: mean={alignment['mean']:.2f} "
        f"min={alignment['min']:.2f} max={alignment['max']:.2f} (bar {threshold})"
    )
    assert alignment["mean"] >= threshold, (
        f"{label} CLIP prompt alignment {alignment['mean']:.2f} is below the {threshold} bar; "
        "the video does not match the prompt"
    )
    return alignment


def gate_vbench(paths: dict, prompt: str, thresholds: dict, label: str, skip_without_mp4=False):
    """VBench gate (skips only when its interpreter is missing); a requested dimension with no returned score FAILS, never silently passes."""
    if "mp4" not in paths:
        message = "the VBench gate needs the muxed mp4, which ffmpeg did not produce"
        if skip_without_mp4:
            pytest.skip(message)
        raise AssertionError(message)
    scores = run_vbench(paths["mp4"], prompt, tuple(thresholds))
    for dimension, bar in thresholds.items():
        assert dimension in scores, f"VBench returned no score for {dimension}"
        logger.info(f"{label} VBench {dimension} = {scores[dimension]:.4f} (bar {bar})")
    failures = [f"{d} = {scores[d]:.4f} < {bar:.4f}" for d, bar in thresholds.items() if scores[d] < bar]
    assert not failures, "VBench below threshold: " + "; ".join(failures)
    return scores
