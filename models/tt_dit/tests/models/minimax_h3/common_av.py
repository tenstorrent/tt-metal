# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Reference-free sanity checks for a joint video + audio generation.

`tests/models/wan2_2/common.py::check_output_sanity` covers the video side; this module adds the
soundtrack and the relationship between the two streams. Every threshold here
sits far below any real output, so these fire on genuine corruption and not on run-to-run noise --
they answer "is this a real video with a real soundtrack, aligned to it", not "is it good".

A/V sync is checked structurally rather than perceptually. MiniMax-H3 puts audio and video rows on
one shared rotary clock (40 audio latents/s against 24 fps, i.e. 5/3 rotary units per frame), so the
thing that can actually go wrong is a *duration* or *ordering* error in packing or decode -- a
half-clip offset, a channel swap, a soundtrack for a different length of video. Cross-correlating
an audio envelope against frame-to-frame motion energy would test something the model is not
trained to guarantee (generated audio need not be causally tied to visible motion), so it is
reported as a diagnostic and never asserted on.

The artifact and metric helpers shared by the e2e gates (`write_artifacts`, `run_vbench`,
`clip_prompt_alignment`, the warm-generation / timing-table / tier-5 / tier-6 scaffolding at the
bottom of this module) also live here, so no test module has to import from another test module.
Helpers used by a single gate live in that gate's file, not here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS


def check_audio_sanity(audio, *, sampling_rate, expected_seconds, tolerance_seconds=0.05):
    """Guard against a soundtrack that is silent, clipped, constant, or the wrong length.

    Args:
        audio: `(1, channels, samples)` or `(channels, samples)` float waveform.
        sampling_rate: samples per second.
        expected_seconds: duration the video covers.
        tolerance_seconds: allowed disagreement, default one 24 fps frame's worth (~0.042 s) rounded up.
    """
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
    # A decoder stuck on a constant, or one channel dead.
    for index in range(channels):
        channel_std = float(audio[index].std())
        assert channel_std > 1e-4, f"channel {index} is constant (std {channel_std:.2e})"
    # Hard clipping across most of the waveform means the denormalization is wrong, not that the mix
    # is loud. A real waveform touches the rails rarely if at all.
    clipped = float((np.abs(audio) >= 0.999).mean())
    assert clipped < 0.01, f"{clipped:.1%} of samples are at full scale; suspect a scaling error"

    logger.info(
        f"Audio sanity OK: {channels}ch {seconds:.3f} s @ {sampling_rate} Hz, "
        f"peak={peak:.3f}, rms={rms:.4f}, clipped={clipped:.3%}"
    )


def check_av_sync(frames, audio, *, sampling_rate, fps, tolerance_seconds=0.05):
    """The two streams must describe the same span of time, and the pairing must not be degenerate.

    Asserted:
      * video and audio durations agree within `tolerance_seconds` (default ~1 frame).
      * the two stereo channels are not identical -- a mono-duplicated soundtrack means
        `unpack_audio_tokens` collapsed the channel-major layout.
      * neither stream is empty.

    Reported only (see the module docstring): the lag at which audio envelope energy best correlates
    with frame-to-frame motion energy. Useful for spotting a gross half-clip offset by eye; not a
    pass/fail criterion, because generated audio is not required to track visible motion.
    """
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
        # Channel-major packing means the left channel is the first block of rows and the right the
        # second. Getting that wrong duplicates one channel, which is inaudible in a mono playback
        # check but is a real unpacking bug.
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
    """Gradient energy *at* the VAE's tile boundaries against the gradient everywhere else.

    The video VAE decodes spatial tiles independently and cross-fades the overlaps on the host. A
    wrong blend extent, a wrong tile origin, or per-tile rather than per-image normalization
    statistics all concentrate their error on those boundary columns and rows -- and a whole-frame
    mean or PCC averages it straight out, which is exactly the rubric's first entry.

    ~1.0 means a boundary column looks like any other column. A real seam runs several times that,
    because a visible edge is a large gradient in a place the image did not put one.

    Args:
        frames: `(F, H, W)` luma or `(F, H, W, 3)`.
        vertical_boundaries: interior tile start columns, in pixels.
        horizontal_boundaries: interior tile start rows, in pixels.
        max_ratio: fail above this. Default 2.0 against ~1.0 measured.
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    frames = np.asarray(frames).astype(np.float32)
    if frames.ndim == 4:
        frames = frames.mean(axis=-1)

    def ratio(gradient, boundaries):
        inside = np.array([b for b in boundaries if 1 <= b < len(gradient) - 1], dtype=int)
        if not len(inside):
            return float("nan")
        # Exclude a couple of pixels either side from the baseline, so a smeared seam cannot
        # inflate the denominator it is being compared against.
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
    """Report a coarse log-spectrum shape, to catch a soundtrack that is noise or a single tone.

    Not asserted as a quality bar -- there is no reference to compare a *generated* soundtrack
    against, so what this provides is a number that moves visibly when the decoder breaks: white
    noise flattens the spectrum toward 1.0, a stuck tone drives it toward 0.
    """
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
    power = spectrum.mean(axis=0)[1:]  # drop DC
    # Geometric over arithmetic mean: Wiener entropy / spectral flatness.
    flatness = float(np.exp(np.log(power + 1e-20).mean()) / (power.mean() + 1e-20))
    band_edges = np.linspace(0, len(power), num_bands + 1).astype(int)
    bands = np.array([power[a:b].mean() for a, b in zip(band_edges[:-1], band_edges[1:]) if b > a])
    logger.info(
        f"Audio log-spectrum: flatness={flatness:.4f}, "
        f"band dB range=[{10 * np.log10(bands.min() + 1e-20):.1f}, {10 * np.log10(bands.max() + 1e-20):.1f}]"
    )
    return {"flatness": flatness}


def _ffmpeg():
    """An ffmpeg executable: the system binary, else imageio-ffmpeg's bundled static build."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def write_artifacts(frames, audio, sampling_rate, directory: Path, stem: str = "t2va"):
    """Write the video, the soundtrack, and a muxed mp4. Returns the paths that were produced.

    `stem` names the files; each gate passes its own (the default is the t2va gate's) so the tasks'
    artifacts coexist in one directory rather than overwriting each other.
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


def probe_streams(path: Path) -> dict:
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


# Only reached when ffprobe is unavailable: the production 768x1344 working point every e2e gate
# runs at.
_FALLBACK_HEIGHT = 768
_FALLBACK_WIDTH = 1344


def decoded_frames(path: Path, count: int) -> np.ndarray:
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
    probe = probe_streams(path).get("video", {})
    height = int(probe.get("height") or _FALLBACK_HEIGHT)
    width = int(probe.get("width") or _FALLBACK_WIDTH)
    buffer = np.frombuffer(result.stdout, dtype=np.uint8)
    usable = (buffer.size // (height * width)) * height * width
    return buffer[:usable].reshape(-1, height, width)


VBENCH_VENV_ENV = "MINIMAX_H3_VBENCH_PYTHON"
DEFAULT_VBENCH_PYTHON = "/data/kevinmi/vbench_env/bin/python"


def run_vbench(video: Path, prompt: str, dimensions) -> dict[str, float]:
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


def temporal_seam_score(frames: np.ndarray, period: int) -> float:
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


# ------------------------------------------------------------------ shared e2e gate scaffolding
#
# The pieces every pipeline e2e gate (t2va / fl2va / ref2va) runs the same way. Per-pipeline
# judgement -- gate values, header wording, which tiers run -- stays at the call sites; what lives
# here is only the mechanics all three had verbatim.

WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
DEFAULT_WEIGHTS = "/data/cglagovich/MiniMax-H3-diffusers"

# Dense: a moving camera, a reflective wet surface, several independent light sources at
# different colour temperatures, volumetric haze, and foreground/background motion at different depths.
# Those are the things a video model is most likely to get wrong, and they are also what the artifact
# rubric reads best -- banding shows in the haze gradients, seams show in the reflections, and flicker
# shows in the neon.
# The gated prompt and the t2va tier-6 thresholds are a **matched pair**. Both were calibrated
# together on this prompt; swapping one without recalibrating the other breaks the gate. Measured
# on it: CLIP 37.37, VBench imaging_quality 0.6896.
#
# `imaging_quality` in particular is prompt-dependent, not just model-dependent: it is a no-reference
# IQA model that rewards sharp, well-lit frames. A dark rain-at-night scene scored **0.4884** against
# this same 0.64 bar while looking entirely correct. So a showcase prompt belongs in a
# manual run, not in this constant.
#
# Exported from here so the fl2va gate, whose keyframe is frame 0 of the calibrated t2va generation,
# uses the *identical* string by import rather than by a copy that can drift.
CALIBRATED_FOX_PROMPT = (
    "A red fox trots across a snowy field at dawn, its breath visible in the cold air. "
    "The low sun throws long blue shadows behind it, and loose snow lifts from each footfall."
)

# One truthy set for the RUN_CLIP / RUN_VBENCH switches, so every gate parses them identically.
_ENV_TRUTHY = ("1", "true", "True")


def clip_gate_enabled() -> bool:
    """RUN_CLIP, defaulting **on**."""
    return os.environ.get("RUN_CLIP", "1") in _ENV_TRUTHY


def vbench_gate_enabled() -> bool:
    """RUN_VBENCH, defaulting **on**."""
    return os.environ.get("RUN_VBENCH", "1") in _ENV_TRUTHY


def weights_dir(*required_subdirs: str, default: str = DEFAULT_WEIGHTS) -> Path:
    """The diffusers snapshot directory; `pytest.skip`s when it or a required subdir is missing.

    `required_subdirs` are the partitions the caller's gate actually loads (the ref2va gate
    requires `transformer_ref`, and passes `default=""` because it has no default snapshot).
    """
    directory = Path(os.environ.get(WEIGHTS_ENV, default))
    if not directory.is_dir():
        pytest.skip(f"no MiniMax-H3 snapshot at {directory}; set {WEIGHTS_ENV}")
    missing = [name for name in required_subdirs if not (directory / name).is_dir()]
    if missing:
        pytest.skip(f"MiniMax-H3 snapshot at {directory} is missing {missing}; set {WEIGHTS_ENV}")
    return directory


def artifact_dir(env: str, default_name: str) -> Path:
    """The gate's artifact directory (`$env`, else `~/{default_name}`), created if absent."""
    directory = Path(os.environ.get(env) or Path.home() / default_name)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_warm_generation(pipeline, prompt: str, *, seed: int, **gen_kwargs):
    """One full warmup at the real shape, then the timed generation, asserted to be warm.

    `gen_kwargs` is everything that shapes the request (num_frames/height/width/steps, plus
    keyframes or references), passed identically to `pipeline.warmup` and to the timed call --
    every program in the 50-block stack is keyed on the padded packed length, so a warmup at any
    other shape warms nothing. The padded-length agreement is asserted rather than assumed;
    it is what makes `pipeline.last_timings` a fully-warm measurement.
    """
    pipeline.warmup(prompt=prompt, **gen_kwargs)
    warm_padded_len = pipeline.last_padded_len

    output = pipeline(prompt, seed=seed, **gen_kwargs)

    assert pipeline.last_padded_len == warm_padded_len, (
        f"warmup ran at padded_len {warm_padded_len} but the measured call ran at "
        f"{pipeline.last_padded_len}; this number is not warm"
    )
    return output


def log_timing_table(pipeline, label: str, num_forwards: int, video_seconds: float, expected_total_s=None, extra=""):
    """The MEASUREMENT block: per-stage rows, total, per-forward, realtime factor.

    `extra` is spliced into the header right after "2 links" and carries everything that
    differs per gate -- geometry, anchors, `l1_small_size` -- so those meaningful differences
    stay at the call site instead of being regenerated here. `expected_total_s`, when given,
    asserts the total against the gate's did-something-collapse bar. Returns the total.
    """
    rows = pipeline.last_timings
    total = sum(seconds for _, seconds in rows)
    logger.info(
        f"MEASUREMENT {label} fully warm | mesh 4x8 Blackhole, TP=4 axis 0 / SP=8 axis 1, ring, 2 links{extra} "
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
    """`output.video` `(1, 3, F, H, W)` in [0, 1] -> `(F, H, W, 3)` uint8, which is what the checkers take.

    Clamps then rounds before the uint8 cast: a plain `.astype(np.uint8)` truncates and, worse,
    *wraps around* on any value that strays outside [0, 1], turning a barely-out-of-range white
    pixel into a black one.
    """
    video = output.video
    assert video.ndim == 5 and video.shape[0] == 1, f"unexpected video shape {tuple(video.shape)}"
    frames = video[0].permute(1, 2, 3, 0).clamp(0, 1).mul(255).round().to(torch.uint8)
    return frames.cpu().numpy()


def check_written_file(paths: dict, expected_frames: int, seam_period: int = 17):
    """Tier 5: gate the written *file*, not just the tensor. No-op when ffmpeg produced no mp4.

    Probes the container (both streams present, muxed A/V skew), re-decodes it (frame count), and
    scores the temporal seam at the video VAE's chunk period.
    """
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

    decoded = decoded_frames(paths["mp4"], count=1)
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


def gate_clip(frames: np.ndarray, prompt: str, threshold: float, label: str, enabled=None):
    """The CLIP prompt-alignment gate. May `pytest.skip` when `open_clip` is missing.

    `enabled=None` reads RUN_CLIP (default on); a caller that computes its own switch (the t2va
    gate forces tier 6 off under a prompt override) passes it explicitly. `threshold` stays at
    the call site because the bars are calibrated per pipeline.
    """
    if enabled is None:
        enabled = clip_gate_enabled()
    if not enabled:
        logger.info("RUN_CLIP=0, skipping the CLIP prompt-alignment gate")
        return None
    pytest.importorskip("open_clip", reason="RUN_CLIP=1 but open_clip is not installed (set RUN_CLIP=0)")
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


def gate_vbench(paths: dict, prompt: str, thresholds: dict, label: str, enabled=None, skip_without_mp4=False):
    """The VBench gate, with the per-dimension failure list in the assert message.

    `enabled=None` reads RUN_VBENCH (default on). A requested dimension with no returned score is
    an *ungated* dimension, not a pass, so it fails. `skip_without_mp4` chooses what a missing
    muxed mp4 means: the t2va gate fails (its own ffmpeg step should have produced it), the ref2va
    gate skips.
    """
    if enabled is None:
        enabled = vbench_gate_enabled()
    if not enabled:
        logger.info("RUN_VBENCH=0, skipping the VBench gate")
        return None
    if "mp4" not in paths:
        message = "RUN_VBENCH=1 needs the muxed mp4, which ffmpeg did not produce"
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
