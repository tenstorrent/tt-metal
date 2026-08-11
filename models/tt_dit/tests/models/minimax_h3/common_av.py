# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Reference-free sanity checks for a joint video + audio generation.

`tests/models/wan2_2/common.py::check_output_sanity` covers the video side; nothing in tree covered
the soundtrack or the relationship between the two, which is what this adds. Every threshold here
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
`clip_prompt_alignment`, the ref2va reference sets) also live here, so no test module has to
import from another test module.
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

from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, prepare_keyframe_image
from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference, reference_from_video_file
from .common import create_fractal_image


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
    return {"flatness": flatness, "bands_db": 10 * np.log10(bands + 1e-20)}


def check_keyframe_anchor(frames, keyframe, *, index, stretch, width, height, pcc_floor=0.3):
    """A decoded frame must correlate with the keyframe that anchored it.

    The `fl2va` analogue of `wan2_2.common.check_first_frame_matches_seed`, and it exists separately
    because that helper resizes the seed with a plain `PIL.resize`, i.e. a **stretch**. That is right
    for MiniMax-H3's *first* keyframe and wrong for any other: `prepare_keyframe_image` stretches only
    the geometry anchor (the first keyframe given) and **cover-crops** every later one -- scale by
    `max(W/w, H/h)`, then centre-crop. Comparing a cover-cropped keyframe against a stretched
    reference would fail on a correct pipeline.

    So the canvas rule is applied here rather than assumed, by calling `prepare_keyframe_image`
    itself. That also means this helper cannot drift from the pipeline's own preparation.

    This is a real correctness signal rather than a formality: the anchors are noised only to
    `t = 0.999`, so `0.999 * x0 + 0.001 * noise` is essentially the clean VAE latent of the keyframe,
    and a decoded anchor frame that does not resemble it means the conditioning path is broken --
    wrong rows written, anchors overwritten during denoising, or the conditioning block placed at the
    wrong sequence position.

    Args:
        frames: decoded video, `(F, H, W, 3)`, batch dim removed.
        keyframe: the PIL keyframe *as supplied to the pipeline*, before preparation.
        index: which decoded frame to compare -- `0` for a `first` anchor, `-1` for a `last` one.
        stretch: how the pipeline prepared this keyframe. `True` for the first keyframe given.
        pcc_floor: minimum Pearson correlation. Provisional; tighten once real values are recorded.
    """
    frame = frames[index]
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    frame = np.asarray(frame).astype(np.float64)

    prepared = prepare_keyframe_image(keyframe.convert("RGB"), height, width, stretch)
    expected = np.asarray(prepared).astype(np.float64)
    assert frame.shape == expected.shape, f"frame {index} shape {frame.shape} != keyframe {expected.shape}"

    pcc = float(np.corrcoef(frame.ravel(), expected.ravel())[0, 1])
    label = "first" if index == 0 else "last"
    logger.info(f"fl2va {label}-keyframe anchor: decoded frame {index} vs keyframe PCC = {pcc:.4f}")
    assert pcc > pcc_floor, (
        f"decoded frame {index} barely correlates with the {label} keyframe (PCC={pcc:.3f}); "
        "the fl2va conditioning path is likely broken"
    )
    return pcc


def check_tile_boundary_gradient(frames, *, vertical_boundaries, horizontal_boundaries, max_ratio=3.0):
    """One-pixel gradient at each tile boundary against its own neighbourhood.

    The sensitive complement to :func:`check_spatial_seams`, which compares *block-mean* activity either
    side of a boundary and therefore cannot see a seam narrower than its blocks. Measured on a clean
    production frame: `check_spatial_seams` reports 1.03 while every one of the six vertical boundaries
    carries a per-column gradient 1.2-1.5x its neighbourhood. Both numbers are correct; they measure
    different things, and only this one would notice a one-pixel discontinuity from the tiled VAE decode.

    A control matters here and is built in: non-boundary columns are measured the same way and must sit
    near 1.0, otherwise the statistic is picking up ordinary image structure rather than a seam.

    The bar is loose (3.0) because a ratio of 1.2-1.5 is the *known good* state, not a
    defect: linear cross-fading two independently decoded tiles leaves a derivative discontinuity at the
    ends of the blend, and at production geometry that measures ~0.3/255 of luma step -- 0.12 % of full
    scale, invisible at 8x zoom, and identical in `t2va`. This gate exists to catch that becoming
    *visible*, which is a several-fold change, not to police the floor.
    """
    frames = np.asarray(frames)
    luma = frames.astype(np.float64).mean(-1) if frames.ndim == 4 else frames.astype(np.float64)
    gx = np.abs(np.diff(luma, axis=2)).mean(axis=(0, 1))
    gy = np.abs(np.diff(luma, axis=1)).mean(axis=(0, 2))

    def ratio(profile, index):
        near = np.concatenate([profile[index - 12 : index - 3], profile[index + 3 : index + 12]])
        return float(profile[index - 1] / max(float(np.median(near)), 1e-9))

    results = {}
    for name, profile, boundaries in (("vertical", gx, vertical_boundaries), ("horizontal", gy, horizontal_boundaries)):
        ratios = {int(b): ratio(profile, int(b)) for b in boundaries if 12 < int(b) < len(profile) - 12}
        results[name] = ratios
        if ratios:
            logger.info(
                f"{name} tile-boundary gradient ratios (1.0 = no seam): "
                + ", ".join(f"x={b}:{r:.3f}" if name == "vertical" else f"y={b}:{r:.3f}" for b, r in ratios.items())
            )

    # Control: columns that are not boundaries must read ~1.0, or the measurement is meaningless.
    generator = np.random.default_rng(0)
    candidates = generator.integers(30, len(gx) - 30, 24)
    control = [c for c in candidates if all(abs(int(c) - int(b)) > 16 for b in vertical_boundaries)][:12]
    control_ratios = [ratio(gx, int(c)) for c in control]
    mean_control = float(np.mean(control_ratios))
    logger.info(f"control non-boundary columns: mean ratio {mean_control:.3f}, max {max(control_ratios):.3f}")
    assert mean_control < 1.15, (
        f"control columns average {mean_control:.3f}; this statistic is tracking image structure rather "
        "than tile boundaries, so its boundary numbers mean nothing"
    )

    worst = max((r, f"{n} {b}") for n, rs in results.items() for b, r in rs.items())
    assert worst[0] < max_ratio, (
        f"tile-boundary gradient at {worst[1]} is {worst[0]:.2f}x its neighbourhood (control "
        f"{mean_control:.2f}); a visible seam. See the artifact rubric"
    )
    results["control"] = mean_control
    return results


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


def write_artifacts(frames, audio, sampling_rate, directory: Path, stem: str = "t2va"):
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


def decoded_frames(path: Path, count: int, *, fallback_height: int = 768, fallback_width: int = 1344) -> np.ndarray:
    """Sample `count` frames evenly out of the written file, as uint8 luma.

    The fallbacks are only reached when ffprobe is unavailable, and default to the production
    768x1344 working point every e2e gate runs at.
    """
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
    height = int(probe.get("height") or fallback_height)
    width = int(probe.get("width") or fallback_width)
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


REFERENCE_MEDIA_ENV = "MINIMAX_H3_REFERENCE_MEDIA"
DEFAULT_REFERENCE_MEDIA = Path.home() / "h3_fl2va_artifacts" / "fl2va_first.mp4"


def reference_video() -> Path:
    path = Path(os.environ.get(REFERENCE_MEDIA_ENV) or DEFAULT_REFERENCE_MEDIA)
    if not path.is_file():
        pytest.skip(f"no reference video at {path}; set {REFERENCE_MEDIA_ENV} to a clip with a soundtrack")
    return path


def ref2va_references(case: str) -> list[MiniMaxH3Reference]:
    """The reference set per e2e case, shared by the ref2va correctness and perf gates.

    ``one_image`` and ``mixed`` condition on a Mandelbrot fractal: for a shape-and-sanity gate the
    useful property is a reference nothing in the prompt could produce. It is also the most
    adversarial of the three and sits at the bottom of the quality table in
    `test_pipeline_ref2va_minimax_h3.py` -- 0.4826 imaging_quality is this case, not ref2va
    generally, which reaches 0.6575 on a photographic reference. The discriminator uses real
    photographs instead.
    """
    if case == "one_image":
        return [MiniMaxH3Reference(image=create_fractal_image(1024, 1024))]
    if case == "video_with_sound":
        return [reference_from_video_file(reference_video())]
    if case == "mixed":
        # One of each, and in an order that is not the natural one: the image first,
        # then a SILENT video, then a standalone audio reference. So the request
        # exercises a video block with no soundtrack rows of its own next to an audio
        # block with no video rows, which is where the per-modality row cursors of
        # `split_condition_blocks` can disagree with the layout.
        sounded = reference_from_video_file(reference_video())
        return [
            MiniMaxH3Reference(image=create_fractal_image(1024, 1024)),
            reference_from_video_file(reference_video(), with_audio=False),
            MiniMaxH3Reference(audio=sounded.audio, sample_rate=sounded.sample_rate),
        ]
    raise ValueError(case)
