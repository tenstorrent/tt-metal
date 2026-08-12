# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 `fl2va` end to end: a prompt plus a first and/or last keyframe in, video and audio out.

ONE combined perf + quality e2e, on the `first_and_last` case: it strictly supersets the lone
`first` and lone `last` cases on packing and scatter coverage -- two conditioning blocks (2016
rows), a two-run vision scatter in the conditioner, and both preparation paths (stretch for the
geometry anchor, cover-crop for the follower). The one behaviour `first_and_last` cannot gate --
a lone `last_image` being the geometry anchor and therefore *stretched*, not cover-cropped -- is
pinned host-side by `test_lone_last_keyframe_is_stretched_not_cover_cropped`.

The test carries both the fully-warm latency measurement and the quality gates on one weight load:
warmup at the real shape (keyframes included), then time the measured generation and gate its
output. The timing method is `pipelines/ltx`'s -- prepares and export excluded,
`Total (compute)` = the sum of `pipeline.last_timings` rows -- so the number is comparable to the
t2va row in `test_pipeline_minimax_h3.py`. The timed Encoder row is a real device encode, vision
tower included. Two warmth conditions are asserted rather than assumed (see the comments in the
test); each can silently invalidate the measurement.

A separate file from `test_pipeline_minimax_h3.py` rather than more cases in it, so the
two e2e gates default to separate processes. One process holding the 50-block DiT's programs *and*
its CCL persistent buffers at several different padded sequence lengths (37888 for t2va, 40448
for the two-anchor fl2va layout) is a memory risk nobody has measured.

What this gates beyond the t2va tiers
-------------------------------------
The reference-free A/V checks are shared with t2va. Two things are specific to `fl2va` and are the
reason this file exists:

- **the keyframe anchor actually anchors.** Decoded frame 0 (or -1) must correlate with the keyframe.
  At `noise_aug = 0.999` the anchor rows are essentially the clean VAE latent of the keyframe, so this
  is a strong signal, and it is the only one that would notice conditioning rows placed at the wrong
  sequence position.
- **the anchors survive the loop.** Nothing re-imposes them; they persist only because the loop never
  writes them. `t2va` cannot test this, having no conditioning rows at all.

Tier 6 and the keyframe's content
---------------------------------
The gated prompt and the tier-6 thresholds are a **matched pair**, and
`imaging_quality` is a no-reference IQA metric that moves with content, not just with correctness.
A keyframe *forces* the content, so an arbitrary photograph would invalidate the t2va calibration
outright.

So the gated keyframe is **frame 0 of the calibrated t2va generation** -- the fox scene those
thresholds were calibrated against. The content distribution is then the one they were measured on,
the same prompt still applies, and the anchor check is maximally meaningful because the keyframe is
in-distribution for the model. It also means this file needs the t2va gate to have run; it skips
rather than inventing content when the artifact is absent.

Tier 6 is calibrated by measurement: CLIP measures 36.63 / 37.30 / 37.00 across the three anchor
cases against t2va's 37.37, so the t2va bar of 33.0 transfers by measurement rather than assumption.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

from ....pipelines.minimax_h3.packing import align_num_frames, prepare_keyframe_image
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.test import ring_params_req_exact_devices
from ..wan2_2.common import check_output_sanity
from .common import create_fractal_image
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    check_written_file,
    gate_clip,
    log_spectral_flatness,
    log_timing_table,
    run_warm_generation,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)

# The t2va working point, unchanged, so the two are comparable. One anchor adds 1008 conditioning rows.
HEIGHT, WIDTH = 768, 1344
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
SEED = 0

# The prompt the t2va tier-6 thresholds were calibrated against. Imported rather than copied, so
# it stays identical to t2va's by construction (see the module docstring on why it must).
PROMPT = CALIBRATED_FOX_PROMPT

ARTIFACT_ENV = "MINIMAX_H3_ARTIFACT_DIR"
# Where the *calibrated t2va* artifact lives, which is where the gated keyframe comes from. Separate
# from ARTIFACT_ENV: that one says where fl2va *writes*, and pointing it at a fresh
# directory must not silently turn the keyframe source into a missing file.
T2VA_ARTIFACT_ENV = "MINIMAX_H3_T2VA_ARTIFACT_DIR"

# Ring collectives, so the fabric must be FABRIC_1D_RING -- taken from the shared helper, same as
# the t2va gate.
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**ring_params_req_exact_devices, "l1_small_size": 65536},
        id="4x8",
    )
]

# Set from measurement, not inherited. The wan2_2 analogue ships a provisional 0.3 with a note to
# tighten it once real values are observed; these are the observed values, all three cases:
#
#   first            frame  0 vs keyframe   0.9971
#   last             frame -1 vs keyframe   0.9943
#   first_and_last   frame  0 / frame -1    0.9971 / 0.9946
#
# 0.95 keeps ~9x margin on `1 - PCC` against the worst of those, which is the same margin convention
# the conditioner bar uses. Not tighter: the anchors go through a VAE round trip
# and 49 denoising steps of a *neighbouring* frame's influence, so some spread is expected, and a
# genuinely broken conditioning path scores near zero rather than 0.9.
ANCHOR_PCC_FLOOR = 0.95

# t2va's bar, applied here because the first fl2va run measured 36.63 / 37.30 / 37.00 against t2va's
# 37.37 -- i.e. it transfers. It transfers *because* the gated keyframe comes from the calibrated t2va
# generation; an arbitrary keyframe would need its own calibration.
CLIP_THRESHOLD = 33.0

# Generous: a regression bar, not a target -- same convention as the t2va gate's. There is no tuned
# perf target yet; the point is to notice a collapse (a lost cache, a fallback kernel). Comfortable
# even with the real conditioner encode (vision tower included) inside the timed window.
EXPECTED_TOTAL_S = 400.0


# ------------------------------------------------------------------ checkers only this gate uses


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

    The sensitive complement to :func:`common_av.check_spatial_seams`, which compares *block-mean*
    activity either side of a boundary and therefore cannot see a seam narrower than its blocks.
    Measured on a clean production frame: `check_spatial_seams` reports 1.03 while every one of the
    six vertical boundaries carries a per-column gradient 1.2-1.5x its neighbourhood. Both numbers
    are correct; they measure different things, and only this one would notice a one-pixel
    discontinuity from the tiled VAE decode.

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


def _gated_keyframe() -> Image.Image:
    """Frame 0 of the calibrated t2va generation, at the production canvas.

    **This keyframe cannot on its own show that conditioning works, and the anchor correlation it
    produces is confounded.** It is taken from t2va's *own output*, so a pipeline that ignored the
    keyframe entirely and merely re-ran t2va would also score ~0.997 against it. That is what
    `test_fl2va_follows_the_keyframe` exists to rule out, with a keyframe the model would never
    produce. What this one is for is tier 6: the CLIP and VBench bars were calibrated on this content,
    so it is the only keyframe those bars legitimately apply to.

    Read from the t2va gate's artifact rather than generated here: a second generation would cost
    another 90 s of Galaxy time to produce content this file only needs to *read*, and reusing the
    calibrated one is what makes the thresholds applicable (see the module docstring on tier 6).
    """
    source_dir = Path(os.environ.get(T2VA_ARTIFACT_ENV) or Path.home() / "h3_t2va_artifacts")
    artifact = source_dir / "t2va.mp4"
    if not artifact.is_file():
        pytest.skip(
            f"no calibrated t2va artifact at {artifact}; run test_pipeline_minimax_h3.py first so the "
            "fl2va keyframe comes from content the tier-6 thresholds were calibrated on"
        )
    frame = _first_frame(artifact)
    assert frame.size == (WIDTH, HEIGHT), f"t2va artifact is {frame.size}, expected {(WIDTH, HEIGHT)}"
    return frame


def _first_frame(path: Path) -> Image.Image:
    """Decode frame 0 of an mp4 with imageio, which the tier-5 checks already depend on."""
    import imageio.v3 as iio

    return Image.fromarray(np.asarray(iio.imread(path, index=0, plugin="pyav"))).convert("RGB")


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_fl2va_end_to_end(mesh_device, reset_seeds):
    """The `first_and_last` case: two conditioning blocks (2016 rows), a two-run vision scatter in
    the conditioner, and both keyframe-preparation paths -- stretch for the geometry anchor and
    cover-crop for the follower. Perf and quality on the same timed generation.
    """
    case = "first_and_last"
    keyframe = _gated_keyframe()
    image = keyframe
    last_image = keyframe

    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights_dir())

    # ---- fully-warm latency, `pipelines/ltx`'s method, via `run_warm_generation`. Two things have
    # to be right or the number means nothing:
    # 1. The warmup must be fl2va-shaped, keyframes included: every program in the 50-block stack
    #    is keyed on the padded packed length, so a t2va warmup warms nothing for fl2va.
    # 2. The warmup must run at the same prompt length -- asserted by the helper rather than
    #    assumed. For t2va a one-token "warmup" prompt works purely by coincidence (1 and 39 tokens
    #    both round up to 37888); with two ~1008-row vision blocks it does not.
    # The measured run pays the full device conditioner encode -- vision tower included -- inside the
    # timed Encoder row: there is no prompt-embedding cache, so that row is a genuine measurement.
    # No reference number is recorded for the fl2va encode (t2va's text-only encode measures ~2.8 s;
    # the vision tower adds to that here), and the warmup already compiled the encoder path.
    output = run_warm_generation(
        pipeline,
        PROMPT,
        image=image,
        last_image=last_image,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    expected_frames = align_num_frames(NUM_FRAMES)
    logger.info(
        f"fl2va[{case}] padded_len={pipeline.last_padded_len} "
        f"video={tuple(output.video.shape)} audio={tuple(output.audio.shape)}"
    )

    num_forwards = NUM_INFERENCE_STEPS - 1
    video_seconds = expected_frames / output.fps
    # No tuned target yet (bringup at current perf); EXPECTED_TOTAL_S is a loose
    # did-something-collapse bar.
    log_timing_table(
        pipeline,
        "fl2va",
        num_forwards=num_forwards,
        video_seconds=video_seconds,
        expected_total_s=EXPECTED_TOTAL_S,
        extra=(
            f" | {WIDTH}x{HEIGHT}, {expected_frames} frames @ {output.fps} fps "
            f"({video_seconds:.2f} s), {num_forwards} forwards, first+last anchors, "
            f"padded_len {pipeline.last_padded_len}"
        ),
    )
    assert output.video.shape[2] == expected_frames, f"{output.video.shape[2]} frames, expected {expected_frames}"

    frames = to_uint8_frames(output)

    # ---- tier 4: reference-free sanity, shared with t2va ----
    check_output_sanity(frames, num_frames=expected_frames, height=HEIGHT, width=WIDTH)
    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=expected_frames / output.fps)
    check_av_sync(frames, output.audio, sampling_rate=output.sampling_rate, fps=output.fps)
    log_spectral_flatness(output.audio, sampling_rate=output.sampling_rate)

    # Boundaries from the VAE's own tile grid rather than re-derived: a re-derivation with overlap 32
    # against the real 64 checks positions that are not boundaries at all. Note the
    # return shape -- `((y_starts, lengths, overlaps), (x_starts, ...))`, rows first -- and that
    # vertical seams come from the *x* starts.
    ratio = pipeline.vae_config.spatial_compression_ratio
    (y_starts, _, _), (x_starts, _, _) = pipeline.vae.decode_tile_grid(HEIGHT // ratio, WIDTH // ratio)
    check_spatial_seams(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])
    # The sensitive complement: block-mean seam ratios cannot see a one-pixel discontinuity, and the
    # tiled VAE decode leaves one at ~0.12 % of full scale. Sub-visible, and gated so it stays that way.
    check_tile_boundary_gradient(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])

    # ---- fl2va-specific: the anchors ----
    # `stretch` follows the pipeline's rule: the FIRST keyframe given is the geometry anchor.
    if image is not None:
        check_keyframe_anchor(
            frames, image, index=0, stretch=True, width=WIDTH, height=HEIGHT, pcc_floor=ANCHOR_PCC_FLOOR
        )
    if last_image is not None:
        check_keyframe_anchor(
            frames,
            last_image,
            index=-1,
            stretch=image is None,
            width=WIDTH,
            height=HEIGHT,
            pcc_floor=ANCHOR_PCC_FLOOR,
        )

    # ---- tier 5: the written file, not just the tensor ----
    artifacts = artifact_dir(ARTIFACT_ENV, "h3_t2va_artifacts")
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=f"fl2va_{case}")
    check_written_file(paths, expected_frames)
    if "mp4" in paths:
        # Frames for the artifact rubric. Statistics average away exactly the two defects that matter
        # -- a seam and a flicker -- so the reminder below is not rhetorical.
        for index in (0, 17, 62, expected_frames - 1):
            Image.fromarray(frames[index]).save(artifacts / f"fl2va_{case}_frame_{index}.png")

    # ---- tier 6: generative quality ----
    # The bar is t2va's, and it is applied because it was *measured* to transfer rather than assumed to.
    # First-run values, all three cases: mean 36.63 / 37.30 / 37.00 against t2va's 37.37, min 34.86.
    # That the keyframe is frame 0 of the calibrated t2va generation is what makes this legitimate --
    # the recorded counterexample is a night scene that scored imaging_quality 0.4884 against a
    # 0.64 bar while being visually perfect, purely because the content moved.
    gate_clip(frames, PROMPT, CLIP_THRESHOLD, f"fl2va[{case}]")

    logger.info(
        "REMINDER: read the artifact rubric against these frames -- seams and flicker are what every "
        "whole-tensor metric averages away, and both are parallelism bugs"
    )


# ------------------------------------------------------------------ host: keyframe preparation
#
# Which preparation path a lone `last_image` takes -- the one behaviour the `first_and_last` e2e
# case cannot gate. Pinned on host instead of with a third 90-second generation.


def test_lone_last_keyframe_is_stretched_not_cover_cropped():
    """A lone `last_image` is the geometry anchor, so it is *stretched* -- not cover-cropped.

    The pipeline keys `stretch` on position in the keyframe list, not on the anchor name:
    `sources = [k for k in (image, last_image) if k is not None]` followed by
    `prepare_keyframe_image(k, height, width, stretch=(i == 0))`. With `image=None` a lone
    `last_image` sits at index 0 and is stretched, despite being the "last" anchor. That looks
    like a bug in `prepare_keyframe_image` until this rule is read, so this pins it host-side.

    The source is 1:1 against the 16:9 canvas with a marker band on its top edge, so the two
    preparation paths are genuinely distinguishable: a stretch distorts aspect but keeps every
    source pixel, while a cover-crop (scale by `max(W/w, H/h)`, centre-crop) discards the top and
    bottom of a 1:1 source outright.
    """
    source = np.zeros((512, 512, 3), dtype=np.uint8)
    source[:64, :, 0] = 255  # top band: red
    source[-64:, :, 2] = 255  # bottom band: blue
    source[:, :, 1] = 32  # non-zero interior so the crop is not comparing zeros to zeros
    keyframe = Image.fromarray(source)

    # The pipeline's own rule, replicated for the lone-last request (image=None).
    image, last_image = None, keyframe
    sources = [k for k in (image, last_image) if k is not None]
    prepared = [prepare_keyframe_image(k, HEIGHT, WIDTH, stretch=(i == 0)) for i, k in enumerate(sources)]
    assert len(prepared) == 1

    stretched = prepare_keyframe_image(keyframe, HEIGHT, WIDTH, True)
    cropped = prepare_keyframe_image(keyframe, HEIGHT, WIDTH, False)
    got = np.asarray(prepared[0])

    # Both paths land on the canvas, and they are genuinely different preparations -- otherwise
    # this test could not distinguish them and would gate nothing.
    assert prepared[0].size == stretched.size == cropped.size == (WIDTH, HEIGHT)
    assert not np.array_equal(np.asarray(stretched), np.asarray(cropped))

    # The rule: the lone-last anchor took the stretch path, bit for bit.
    assert np.array_equal(got, np.asarray(stretched)), "a lone last_image must be stretched (it is the geometry anchor)"

    # And the semantics that make the distinction matter: the stretch keeps the source's edges
    # (the marker bands survive), where the cover-crop of a 1:1 source onto 16:9 discards them.
    assert np.asarray(stretched)[0, :, 0].mean() > 200, "stretch must keep the source's top edge"
    assert np.asarray(stretched)[-1, :, 2].mean() > 200, "stretch must keep the source's bottom edge"
    assert np.asarray(cropped)[0, :, 0].mean() < 50, "cover-crop of a 1:1 source must discard the top band"
    assert np.asarray(cropped)[-1, :, 2].mean() < 50, "cover-crop of a 1:1 source must discard the bottom band"


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_fl2va_follows_the_keyframe(mesh_device, reset_seeds):
    """The keyframe *drives* the generation -- it is not merely consistent with it.

    This is the test that makes the anchor numbers mean something, and it exists because the gated
    keyframe cannot do that job. That keyframe is frame 0 of the calibrated t2va generation, so a
    pipeline that ignored keyframes altogether and just re-ran t2va would score ~0.997 against it. The
    correlation is real but confounded, and reporting it alone overstates what has been shown.

    Here the keyframe is a **Mandelbrot fractal** -- content the model will never produce for a prompt
    about a fox in snow. Three claims, and only the three together are conclusive:

    1. decoded frame 0 resembles the fractal. Impossible to satisfy by ignoring the keyframe.
    2. decoded frame 0 resembles the fractal *much better* than it resembles t2va's own frame 0, which
       is what it would resemble if conditioning were a no-op. This is the discriminating comparison.
    3. the tail of the clip has left the fractal behind. A pipeline that pinned every frame rather than
       just the anchor would also pass 1 and 2, and would not be generating a video at all.

    The prompt still describes the fox, so the keyframe and the prompt disagree, which is the
    sharpest possible test of whether the conditioning rows are actually reaching the DiT: a no-op
    keyframe leaves the prompt to win outright.
    """
    fractal = create_fractal_image(WIDTH, HEIGHT)
    pipeline = MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=weights_dir())
    output = pipeline(
        PROMPT,
        image=fractal,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
    frames = to_uint8_frames(output)

    def pcc(a, b):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        return float(np.corrcoef(a, b)[0, 1])

    prepared = np.asarray(prepare_keyframe_image(fractal, HEIGHT, WIDTH, True))
    to_keyframe = pcc(frames[0], prepared)
    to_t2va = pcc(frames[0], np.asarray(_gated_keyframe()))
    tail = pcc(frames[-1], prepared)

    logger.info(
        f"fl2va keyframe-drives-generation: frame 0 vs fractal keyframe {to_keyframe:.4f}, "
        f"frame 0 vs t2va's own frame 0 {to_t2va:.4f}, frame -1 vs fractal keyframe {tail:.4f}"
    )
    artifacts = artifact_dir(ARTIFACT_ENV, "h3_t2va_artifacts")
    for index in (0, 17, 62, align_num_frames(NUM_FRAMES) - 1):
        Image.fromarray(frames[index]).save(artifacts / f"fl2va_fractal_frame_{index}.png")

    # (1) the anchor followed the supplied keyframe.
    assert to_keyframe > ANCHOR_PCC_FLOOR, (
        f"decoded frame 0 correlates only {to_keyframe:.3f} with the fractal keyframe it was "
        "conditioned on; the keyframe is not reaching the DiT"
    )
    # (2) and it followed *that* keyframe rather than reproducing t2va. The margin is what rules out
    # the confound; a no-op pipeline inverts this comparison.
    assert to_keyframe > to_t2va + 0.30, (
        f"frame 0 resembles the fractal keyframe ({to_keyframe:.3f}) barely more than it resembles "
        f"t2va's own frame 0 ({to_t2va:.3f}); conditioning may be a no-op"
    )
    # (3) it is still a video, not 124 copies of the keyframe.
    assert tail < to_keyframe - 0.20, (
        f"the last frame still correlates {tail:.3f} with the keyframe against frame 0's "
        f"{to_keyframe:.3f}; the clip may be pinned throughout rather than anchored at one end"
    )
