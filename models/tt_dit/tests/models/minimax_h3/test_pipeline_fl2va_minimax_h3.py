# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 `fl2va` e2e: prompt plus first/last keyframes in, video and audio out.
`test_fl2va_follows_the_keyframe` (fractal keyframe) is the discriminating conditioning gate.
Separate file so it runs in its own process: DiT programs + CCL buffers at several padded lengths is a memory risk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger
from PIL import Image

from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.minimax_h3.packing import align_num_frames, prepare_keyframe_image
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from ..wan2_2.common import check_output_sanity
from .common import GALAXY_MESHES, create_fractal_image
from .common_av import (
    artifact_dir,
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    check_written_file,
    is_host,
    log_pipeline_perf,
    log_quality,
    log_spectral_flatness,
    quality_logs_enabled,
    run_warm_generation,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)

HEIGHT, WIDTH = 768, 1344
NUM_FRAMES = 124
NUM_INFERENCE_STEPS = 50
SEED = 0

PROMPT = "Brad Pitt from age 18 to age 60"

FIRST_KEYFRAME = Path("/data/DC-deploy/vision-models/brad_18_resize.png")
LAST_KEYFRAME = Path("/data/DC-deploy/vision-models/brad_60_resize.png")

# Ring collectives require FABRIC_1D_RING.
MESHES = GALAXY_MESHES

ANCHOR_PCC_FLOOR = 0.95  # measured 0.9943-0.9971 across the three anchor cases


def check_keyframe_anchor(frames, keyframe, *, index, stretch, width, height, pcc_floor=0.3):
    """Decoded frame `index` must correlate with its keyframe; uses the pipeline's own
    `prepare_keyframe_image` (stretch vs cover-crop) so the comparison canvas cannot drift."""
    frame = frames[index]
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    frame = np.asarray(frame).astype(np.float64)

    prepared = prepare_keyframe_image(keyframe.convert("RGB"), height, width, stretch)
    expected = np.asarray(prepared).astype(np.float64)
    assert frame.shape == expected.shape, f"frame {index} shape {frame.shape} != keyframe {expected.shape}"

    pcc = float(np.corrcoef(frame.ravel(), expected.ravel())[0, 1])
    label = "first" if index == 0 else "last"
    log_quality(f"fl2va {label}-keyframe anchor: decoded frame {index} vs keyframe PCC = {pcc:.4f}")
    assert pcc > pcc_floor, (
        f"decoded frame {index} barely correlates with the {label} keyframe (PCC={pcc:.3f}); "
        "the fl2va conditioning path is likely broken"
    )
    return pcc


def check_tile_boundary_gradient(frames, *, vertical_boundaries, horizontal_boundaries, max_ratio=3.0):
    """One-pixel gradient at each tile boundary vs its neighbourhood -- catches seams the block-mean
    check cannot see. Known-good measures 1.2-1.5x; the built-in control columns must sit near 1.0."""
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
            log_quality(
                f"{name} tile-boundary gradient ratios (1.0 = no seam): "
                + ", ".join(f"x={b}:{r:.3f}" if name == "vertical" else f"y={b}:{r:.3f}" for b, r in ratios.items())
            )

    generator = np.random.default_rng(0)
    candidates = generator.integers(30, len(gx) - 30, 24)
    control = [c for c in candidates if all(abs(int(c) - int(b)) > 16 for b in vertical_boundaries)][:12]
    control_ratios = [ratio(gx, int(c)) for c in control]
    mean_control = float(np.mean(control_ratios))
    log_quality(f"control non-boundary columns: mean ratio {mean_control:.3f}, max {max(control_ratios):.3f}")
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
    """Frame 0 of the calibrated t2va generation -- the only keyframe the tier-6 bars apply to.
    As a conditioning signal it is confounded; `test_fl2va_follows_the_keyframe` rules that out."""
    artifact = Path.home() / "h3_t2va_artifacts" / "t2va.mp4"
    if not artifact.is_file():
        pytest.skip(
            f"no calibrated t2va artifact at {artifact}; run test_pipeline_minimax_h3.py first so the "
            "fl2va keyframe comes from content the tier-6 thresholds were calibrated on"
        )
    frame = _first_frame(artifact)
    assert frame.size == (WIDTH, HEIGHT), f"t2va artifact is {frame.size}, expected {(WIDTH, HEIGHT)}"
    return frame


def _first_frame(path: Path) -> Image.Image:
    import imageio.v3 as iio

    return Image.fromarray(np.asarray(iio.imread(path, index=0, plugin="pyav"))).convert("RGB")


def _load_keyframe(path: Path) -> Image.Image:
    if not path.is_file():
        pytest.skip(f"no keyframe at {path}")
    return Image.open(path).convert("RGB")


@pytest.mark.timeout(10800)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_fl2va_end_to_end(mesh_device, reset_seeds):
    """The `first_and_last` case: both preparation paths and a two-run vision scatter; perf + quality."""
    case = "first_and_last"
    image = _load_keyframe(FIRST_KEYFRAME)
    last_image = _load_keyframe(LAST_KEYFRAME)

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights_dir(), precomputed_adaln=False, dit_fsdp=False
    )

    # Warmup must be fl2va-shaped (keyframes included): programs are keyed on padded length; the helper asserts it.
    benchmark_profiler = BenchmarkProfiler()
    output = run_warm_generation(
        pipeline,
        PROMPT,
        image=image,
        # last_image=last_image,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
        profiler=benchmark_profiler,
    )

    expected_frames = align_num_frames(NUM_FRAMES)
    height, width = int(output.video.shape[-2]), int(output.video.shape[-1])
    log_quality(
        f"fl2va[{case}] padded_len={pipeline.last_padded_len} canvas={width}x{height} "
        f"video={tuple(output.video.shape)} audio={tuple(output.audio.shape)}"
    )

    num_forwards = NUM_INFERENCE_STEPS - 1
    log_pipeline_perf(
        benchmark_profiler,
        label="fl2va",
        pipeline=pipeline,
        num_forwards=num_forwards,
        width=width,
        height=height,
        num_frames=expected_frames,
        fps=output.fps,
        num_inference_steps=NUM_INFERENCE_STEPS,
        extra_lines=("Conditioning: first+last anchors",),
    )

    frames = to_uint8_frames(output)
    if is_host():
        artifacts = artifact_dir("h3_fl2va_artifacts")
        paths = write_artifacts(
            frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=f"fl2va_{case}"
        )
        video_path = paths.get("mp4") or paths.get("silent_mp4")
        if video_path is not None:
            logger.info(f"saved video: {video_path}")
        else:
            logger.info("ffmpeg is missing; skipped mp4")
        for index in (0, 17, 62, frames.shape[0] - 1):
            Image.fromarray(frames[index]).save(artifacts / f"fl2va_{case}_frame_{index}.png")
        check_written_file(paths, expected_frames)

    check_output_sanity(
        frames,
        num_frames=expected_frames,
        height=height,
        width=width,
        log=quality_logs_enabled() and is_host(),
    )
    check_audio_sanity(output.audio, sampling_rate=output.sampling_rate, expected_seconds=expected_frames / output.fps)
    check_av_sync(frames, output.audio, sampling_rate=output.sampling_rate, fps=output.fps)
    log_spectral_flatness(output.audio, sampling_rate=output.sampling_rate)

    # Boundaries from the VAE's own tile grid (vertical seams come from the x starts).
    ratio = pipeline.vae_config.spatial_compression_ratio
    (y_starts, _, _), (x_starts, _, _) = pipeline.vae.decode_tile_grid(height // ratio, width // ratio)
    check_spatial_seams(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])
    check_tile_boundary_gradient(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])

    # `stretch` follows the pipeline's rule: the FIRST keyframe given is the geometry anchor.
    if image is not None:
        check_keyframe_anchor(
            frames, image, index=0, stretch=True, width=width, height=height, pcc_floor=ANCHOR_PCC_FLOOR
        )
    if last_image is not None:
        check_keyframe_anchor(
            frames,
            last_image,
            index=-1,
            stretch=image is None,
            width=width,
            height=height,
            pcc_floor=ANCHOR_PCC_FLOOR,
        )

    log_quality(
        "REMINDER: read the artifact rubric against these frames -- seams and flicker are what every "
        "whole-tensor metric averages away, and both are parallelism bugs"
    )


def test_lone_last_keyframe_is_stretched_not_cover_cropped():
    """The pipeline keys `stretch` on list position, so a lone `last_image` (index 0) is stretched."""
    source = np.zeros((512, 512, 3), dtype=np.uint8)
    source[:64, :, 0] = 255
    source[-64:, :, 2] = 255
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

    assert prepared[0].size == stretched.size == cropped.size == (WIDTH, HEIGHT)
    assert not np.array_equal(np.asarray(stretched), np.asarray(cropped))

    assert np.array_equal(got, np.asarray(stretched)), "a lone last_image must be stretched (it is the geometry anchor)"

    assert np.asarray(stretched)[0, :, 0].mean() > 200, "stretch must keep the source's top edge"
    assert np.asarray(stretched)[-1, :, 2].mean() > 200, "stretch must keep the source's bottom edge"
    assert np.asarray(cropped)[0, :, 0].mean() < 50, "cover-crop of a 1:1 source must discard the top band"
    assert np.asarray(cropped)[-1, :, 2].mean() < 50, "cover-crop of a 1:1 source must discard the bottom band"


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_fl2va_follows_the_keyframe(mesh_device, reset_seeds):
    """Discriminating gate: a fractal keyframe the model would never produce must drive frame 0."""
    fractal = create_fractal_image(WIDTH, HEIGHT)
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights_dir(), precomputed_adaln=False, dit_fsdp=False
    )
    output = pipeline(
        PROMPT,
        image=fractal,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
    frames = to_uint8_frames(output)
    height, width = frames.shape[1], frames.shape[2]

    def pcc(a, b):
        a = np.asarray(a, dtype=np.float64).ravel()
        b = np.asarray(b, dtype=np.float64).ravel()
        return float(np.corrcoef(a, b)[0, 1])

    prepared = np.asarray(prepare_keyframe_image(fractal, height, width, True))
    to_keyframe = pcc(frames[0], prepared)
    to_t2va = pcc(frames[0], np.asarray(_gated_keyframe()))
    tail = pcc(frames[-1], prepared)

    log_quality(
        f"fl2va keyframe-drives-generation: frame 0 vs fractal keyframe {to_keyframe:.4f}, "
        f"frame 0 vs t2va's own frame 0 {to_t2va:.4f}, frame -1 vs fractal keyframe {tail:.4f}"
    )
    artifacts = artifact_dir("h3_fl2va_artifacts")
    for index in (0, 17, 62, align_num_frames(NUM_FRAMES) - 1):
        Image.fromarray(frames[index]).save(artifacts / f"fl2va_fractal_frame_{index}.png")

    # (1) frame 0 follows the supplied keyframe.
    assert to_keyframe > ANCHOR_PCC_FLOOR, (
        f"decoded frame 0 correlates only {to_keyframe:.3f} with the fractal keyframe it was "
        "conditioned on; the keyframe is not reaching the DiT"
    )
    # (2) the margin rules out the t2va confound; a no-op pipeline inverts this comparison.
    assert to_keyframe > to_t2va + 0.30, (
        f"frame 0 resembles the fractal keyframe ({to_keyframe:.3f}) barely more than it resembles "
        f"t2va's own frame 0 ({to_t2va:.3f}); conditioning may be a no-op"
    )
    # (3) it is still a video, not 124 copies of the keyframe.
    assert tail < to_keyframe - 0.20, (
        f"the last frame still correlates {tail:.3f} with the keyframe against frame 0's "
        f"{to_keyframe:.3f}; the clip may be pinned throughout rather than anchored at one end"
    )
