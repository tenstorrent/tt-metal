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
"""

from __future__ import annotations

import numpy as np
import torch
from loguru import logger

from ....pipelines.minimax_h3.packing import prepare_keyframe_image


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
