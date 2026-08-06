# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` end to end: omni-reference conditioning on the 4x8 mesh.

A separate file from the ``t2va`` / ``fl2va`` e2e gates so it defaults to its own
process: a ref2va request is 1.2x-3.0x t2va's packed length (campaign am. 114) and
one process holding DiT programs plus CCL buffers at several of those lengths is a
memory risk nobody had measured before this campaign's shape probe.

WHY THE fl2va QUALITY GATE DOES NOT TRANSFER

``fl2va``'s keyframe is *pinned*: it occupies frame 0 of the output, so "decoded
frame 0 correlates with the keyframe" has a floor to hold to. A ``ref2va``
reference is **not** pinned to any output row -- it conditions, it is not copied --
so there is no similarity number that a correct implementation must reach, and any
absolute bar would be invented rather than derived.

So the gate here is built the other way round, from a **measured floor**:

1. Generate twice from the *same* prompt and seed with the *same* reference. The
   difference between those two runs is the pipeline's own run-to-run floor.
2. Generate again with a *different* reference of **identical geometry** -- both
   2048x2048, so the packed layout is identical row for row and the noise stream is
   bit-identical, because the draw shapes and their order are unchanged.
3. If conditioning were a no-op, run 3 would be indistinguishable from runs 1 and 2.
   It has to differ by far more than the floor from step 1.

That is a comparison the null hypothesis fails, with no threshold chosen by hand:
the bar is a ratio against a number measured in the same test. The directional
check that follows -- each output resembles the reference it was *given* more than
the one it was not -- is likewise threshold-free.

And the standing rule: **look at the frames.** Seams and flicker are what
whole-tensor metrics average away, and both are parallelism bugs. Artifacts land in
``MINIMAX_H3_REF2VA_ARTIFACT_DIR``.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference, reference_from_video_file
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ....utils.test import ring_params_req_exact_devices
from .common_av import check_audio_sanity, check_av_sync, check_spatial_seams
from .test_pipeline_fl2va_minimax_h3 import create_fractal_image

WEIGHTS_ENV = "MINIMAX_H3_DIFFUSERS_DIR"
ARTIFACT_ENV = "MINIMAX_H3_REF2VA_ARTIFACT_DIR"
MEDIA_ENV = "MINIMAX_H3_REFERENCE_MEDIA"
DEFAULT_MEDIA = Path.home() / "h3_fl2va_artifacts" / "fl2va_first.mp4"

# The campaign working point. Fixed: changing it invalidates the numbers.
WIDTH, HEIGHT, NUM_FRAMES, STEPS, SEED = 1344, 768, 124, 50, 0
FPS = 24

PROMPT = "a slow push-in through a quiet room as afternoon light moves across the floor"

# `l1_small_size` 16384 rather than the 65536 the t2va/fl2va gates use. Measured, not chosen: a video
# reference goes through the video VAE's **taps=3** encoder, whose static circular buffers clash with
# L1 at 65536 and at 32768, and 16384 is the first value that fits (am. 124/126). t2va and fl2va never
# reach that encoder -- a keyframe is one frame and takes the taps=1 path -- which is why they were
# free to reserve more. One process holds every stage, so this one value has to serve all of them,
# including the audio decode STATE.md records 65536 for.
_L1_SMALL = int(os.environ.get("MINIMAX_H3_L1_SMALL", 16384))

# `ring_params_req_exact_devices`, exactly as the t2va and fl2va e2e gates use -- the DiT attends in a
# ring on the SP axis, and a LINE fabric config fails its CCL ops outright with
# `fabric.cpp:174 forwarding_direction.has_value()`. Only `l1_small_size` differs, and that difference
# is measured (above).
MESH_4X8 = [
    pytest.param(
        (4, 8),
        {**ring_params_req_exact_devices, "l1_small_size": _L1_SMALL},
        id="4x8",
    )
]


def _weights_dir() -> Path:
    directory = Path(os.environ.get(WEIGHTS_ENV, ""))
    if not directory.is_dir():
        pytest.skip(f"set {WEIGHTS_ENV} to a diffusers snapshot holding transformer_ref/")
    if not (directory / "transformer_ref").is_dir():
        pytest.skip(f"{directory} has no transformer_ref/; ref2va runs against that partition")
    return directory


def _artifact_dir() -> Path:
    directory = Path(os.environ.get(ARTIFACT_ENV) or Path.home() / "h3_ref2va_artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _reference_video() -> Path:
    path = Path(os.environ.get(MEDIA_ENV) or DEFAULT_MEDIA)
    if not path.is_file():
        pytest.skip(f"no reference video at {path}; set {MEDIA_ENV} to a clip with a soundtrack")
    return path


def _stripe_image(width: int, height: int) -> Image.Image:
    """A high-contrast diagonal stripe field: structured, and nothing a prompt produces.

    The counterpart to the fractal in the discriminator. It must have *structure* --
    a flat field has zero variance and no correlation is defined against it -- and it
    must be the same size, so the two requests share a layout row for row.
    """
    y, x = np.mgrid[0:height, 0:width]
    stripes = (((x + y) // 64) % 2).astype(np.float32)
    rgb = np.stack([stripes, 1.0 - stripes, ((x - y) // 96 % 2).astype(np.float32)], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def _grey_frames(frames: torch.Tensor) -> np.ndarray:
    """``(F, H, W)`` luminance in [0, 1] from a ``(1, 3, F, H, W)`` output."""
    video = frames[0].float().cpu().numpy()
    return (0.299 * video[0] + 0.587 * video[1] + 0.114 * video[2]).astype(np.float64)


def _resemblance(frames: torch.Tensor, image: Image.Image) -> float:
    """Mean per-frame correlation between an output and a reference image.

    Deliberately crude and deliberately whole-frame: this is not a quality metric,
    it is a *direction* indicator, and it is only ever used to compare two numbers
    measured the same way in the same test.
    """
    grey = _grey_frames(frames)
    target = np.asarray(image.convert("L").resize((grey.shape[2], grey.shape[1]), Image.Resampling.LANCZOS))
    target = (target.astype(np.float64) / 255.0).reshape(-1)
    target = target - target.mean()
    if not np.any(target):
        raise ValueError("the reference image has no structure, so no correlation is defined against it")
    scores = []
    for frame in grey:
        centred = frame.reshape(-1) - frame.mean()
        denominator = np.linalg.norm(centred) * np.linalg.norm(target)
        scores.append(0.0 if denominator == 0 else float(centred @ target / denominator))
    return float(np.mean(scores))


def _divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute per-pixel difference between two outputs, over [0, 1] pixels."""
    return float((a.float() - b.float()).abs().mean())


def _write(output, stem: str) -> None:
    """Frames and audio for a human to look at, plus four sampled PNGs."""
    directory = _artifact_dir()
    frames = (output.video[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)
    for index in (0, 17, NUM_FRAMES // 2, NUM_FRAMES - 1):
        Image.fromarray(frames[index]).save(directory / f"{stem}_frame_{index}.png")
    np.save(directory / f"{stem}_audio.npy", output.audio.float().cpu().numpy())
    logger.info(f"{stem}: artifacts in {directory}")


def _pipeline(mesh_device) -> MiniMaxH3Pipeline:
    """A pipeline bound to the `transformer_ref` partition.

    The partition is fixed at construction rather than per call: the two are 62 GB
    each, so switching inside one process would mean a full reload.
    """
    return MiniMaxH3Pipeline.create_pipeline(mesh_device=mesh_device, weights_dir=_weights_dir(), task="ref2va")


# The e2e case list. Trimmed to what the campaign's Phase 0 shape probe admits --
# see `campaigns/minimax-h3-ref2va/ledgers/amendments.md`. `padded` is the measured
# packed length each case runs at, asserted below so a case cannot silently drift
# onto a different shape than the one that was probed.
CASES = {
    "one_image": 46080,
    "video_with_sound": 81664,
    "mixed": 90112,
}


def _references(case: str) -> list[MiniMaxH3Reference]:
    if case == "one_image":
        return [MiniMaxH3Reference(image=create_fractal_image(1024, 1024))]
    if case == "video_with_sound":
        return [reference_from_video_file(_reference_video())]
    if case == "mixed":
        # One of each, and in an order that is not the natural one: the image first,
        # then a SILENT video, then a standalone audio reference. So the request
        # exercises a video block with no soundtrack rows of its own next to an audio
        # block with no video rows, which is where the per-modality row cursors of
        # `split_condition_blocks` can disagree with the layout.
        sounded = reference_from_video_file(_reference_video())
        return [
            MiniMaxH3Reference(image=create_fractal_image(1024, 1024)),
            reference_from_video_file(_reference_video(), with_audio=False),
            MiniMaxH3Reference(audio=sounded.audio, sample_rate=sounded.sample_rate),
        ]
    raise ValueError(case)


@pytest.mark.timeout(9000)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
@pytest.mark.parametrize("case", list(CASES), ids=list(CASES))
def test_ref2va_end_to_end(mesh_device, case, reset_seeds):
    """A full ref2va generation: video plus its synchronized soundtrack.

    What this gates is that every path a reference touches agrees on geometry -- the
    conditioner's vision blocks, the VAE encode, the packed layout, the rotary clock
    and both decoders. A mismatch anywhere shows up as a wrong shape, a failed
    assert, or a desynchronized soundtrack, none of which needs a quality bar.
    """
    pipeline = _pipeline(mesh_device)
    output = pipeline(
        PROMPT,
        references=_references(case),
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        seed=SEED,
    )

    assert output.video.shape == (1, 3, NUM_FRAMES, HEIGHT, WIDTH), tuple(output.video.shape)
    assert output.video.min() >= 0.0 and output.video.max() <= 1.0, "decoded video must be in [0, 1]"
    assert torch.isfinite(output.video).all() and torch.isfinite(output.audio).all()
    # The shape the campaign probed. A drift here means the case is no longer the
    # request that was measured, so the memory verdict no longer covers it.
    assert (
        pipeline.last_padded_len == CASES[case]
    ), f"{case} ran at padded_len {pipeline.last_padded_len}, not the probed {CASES[case]}"

    frames = (output.video[0].permute(1, 2, 3, 0).float().cpu().numpy() * 255).astype(np.uint8)
    check_audio_sanity(
        output.audio, sampling_rate=output.sampling_rate, expected_seconds=NUM_FRAMES / FPS, tolerance_seconds=0.05
    )
    check_av_sync(frames, output.audio, sampling_rate=output.sampling_rate, fps=FPS)
    # Seams and flicker are the two defects a whole-tensor metric averages away, and
    # both are parallelism bugs -- which is exactly what a 3x longer sequence stresses.
    check_spatial_seams(frames, vertical_boundaries=(448, 896), horizontal_boundaries=(384,))
    _write(output, f"ref2va_{case}")
    logger.info(f"ref2va[{case}] padded_len={pipeline.last_padded_len} timings={pipeline.last_timings}")


@pytest.mark.timeout(9000)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_ref2va_conditioning_is_not_a_no_op(mesh_device, reset_seeds):
    """The discriminator, against a floor measured in this same test.

    Three generations, all at the smallest ref2va shape:

    ``A``  fractal reference, seed 0
    ``A'`` the same request again -- this measures the pipeline's run-to-run floor
    ``B``  a stripe reference of **identical geometry**, seed 0

    Identical geometry is what makes ``B`` comparable: both references are 2048x2048,
    so the packed layout is identical row for row and every noise draw has the same
    shape in the same order, i.e. the noise is bit-identical. The only thing that
    differs between ``A`` and ``B`` is the *content* of the reference.

    So if conditioning were ignored, ``A`` and ``B`` would be as close as ``A`` and
    ``A'``. The gate is that they are not, by a wide margin -- a ratio against a
    measured floor rather than a threshold anyone chose.
    """
    pipeline = _pipeline(mesh_device)
    fractal = create_fractal_image(1024, 1024)
    stripes = _stripe_image(1024, 1024)

    def generate(image):
        return pipeline(
            PROMPT,
            references=[MiniMaxH3Reference(image=image)],
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            seed=SEED,
        )

    a = generate(fractal)
    a_repeat = generate(fractal)
    b = generate(stripes)

    floor = _divergence(a.video, a_repeat.video)
    signal = _divergence(a.video, b.video)
    to_fractal = (_resemblance(a.video, fractal), _resemblance(b.video, fractal))
    to_stripes = (_resemblance(a.video, stripes), _resemblance(b.video, stripes))

    logger.info(
        f"ref2va discriminator: run-to-run floor {floor:.6f}, reference-swap signal {signal:.6f} "
        f"(ratio {signal / max(floor, 1e-9):.1f}x) | resemblance to fractal A={to_fractal[0]:.4f} "
        f"B={to_fractal[1]:.4f} | to stripes A={to_stripes[0]:.4f} B={to_stripes[1]:.4f}"
    )
    for stem, output in (("discriminate_fractal", a), ("discriminate_stripes", b)):
        _write(output, stem)

    # 1. The same request twice must be close. If this is large the floor is
    #    meaningless and so is everything below it, so it fails loudly rather than
    #    silently widening the bar.
    assert floor < 0.02, f"the same request twice diverged by {floor:.6f}; the pipeline is not reproducible"

    # 2. Swapping the reference must move the output by far more than that floor.
    #    A no-op implementation scores signal == floor exactly, since the layout and
    #    the noise are identical between the two requests.
    assert signal > 10 * floor, (
        f"swapping the reference moved the output by {signal:.6f} against a run-to-run floor of "
        f"{floor:.6f} ({signal / max(floor, 1e-9):.1f}x): the reference is not conditioning the output"
    )

    # 3. Directional, and threshold-free: each output must resemble the reference it
    #    was actually given more than the one it was not.
    assert to_fractal[0] > to_fractal[1], (
        f"the fractal-conditioned output resembles the fractal {to_fractal[0]:.4f}, no more than the "
        f"stripe-conditioned one does ({to_fractal[1]:.4f})"
    )
    assert to_stripes[1] > to_stripes[0], (
        f"the stripe-conditioned output resembles the stripes {to_stripes[1]:.4f}, no more than the "
        f"fractal-conditioned one does ({to_stripes[0]:.4f})"
    )


@pytest.mark.timeout(9000)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESH_4X8, indirect=["mesh_device", "device_params"])
def test_ref2va_reference_order_changes_the_request(mesh_device, reset_seeds):
    """Reordering the same two references is a different request, and must generate differently.

    The host gate proves the *layout* changes; this proves the change reaches the
    output. Both references are 2048x2048, so the two requests have identical row
    counts and an identical noise stream -- only the rotary clock and the presentation
    labels differ.
    """
    pipeline = _pipeline(mesh_device)
    fractal = create_fractal_image(1024, 1024)
    stripes = _stripe_image(1024, 1024)

    def generate(references):
        return pipeline(
            PROMPT,
            references=references,
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=STEPS,
            seed=SEED,
        )

    forward = generate([MiniMaxH3Reference(image=fractal), MiniMaxH3Reference(image=stripes)])
    reversed_ = generate([MiniMaxH3Reference(image=stripes), MiniMaxH3Reference(image=fractal)])

    divergence = _divergence(forward.video, reversed_.video)
    logger.info(f"ref2va reference order: divergence {divergence:.6f}, padded_len={pipeline.last_padded_len}")
    assert forward.video.shape == reversed_.video.shape
    assert divergence > 0.01, (
        f"reordering the references changed the output by only {divergence:.6f}; the order is supposed to "
        "advance the shared rotary clock and relabel the presentation"
    )
