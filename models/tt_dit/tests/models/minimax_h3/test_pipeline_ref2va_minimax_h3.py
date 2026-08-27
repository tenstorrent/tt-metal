# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""MiniMax-H3 ``ref2va`` e2e on the ``mixed`` case (largest shape, all three reference modalities).
``test_ref2va_conditioning_is_not_a_no_op`` is the discriminating gate, judged against a measured run-to-run floor.
Separate file so it runs in its own process: ref2va padded lengths are 1.2-3.0x t2va's, a memory risk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from models.perf.benchmarking_utils import BenchmarkProfiler

from ....pipelines.minimax_h3.packing_ref2va import MiniMaxH3Reference, reference_from_video_file
from ....pipelines.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from .common import GALAXY_MESHES, create_fractal_image
from .common_av import (
    artifact_dir,
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    gate_clip,
    gate_vbench,
    log_pipeline_perf,
    log_quality,
    run_warm_generation,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)

WIDTH, HEIGHT, NUM_FRAMES, STEPS, SEED = 1344, 768, 124, 50, 0
FPS = 24

PROMPT = "a slow push-in through a quiet room as afternoon light moves across the floor"

# Bars set below the minimum measured across the three ref2va cases; t2va's bars do NOT transfer.
REF2VA_CLIP_THRESHOLD = 25.0  # min measured 29.05
REF2VA_VBENCH_THRESHOLDS = {
    "subject_consistency": 0.90,
    "background_consistency": 0.89,
    "motion_smoothness": 0.97,  # t2va's, and ref2va measures *better* than t2va on it
    "dynamic_degree": 1.0,  # binary over one video: 1.0 or the clip is frozen
    "imaging_quality": 0.44,
}

# 16384, not the other gates' 65536: the video VAE's taps=3 encoder (only ref2va reaches it) clashes with L1 above it.
_L1_SMALL = 16384

# Ring collectives require FABRIC_1D_RING; only `l1_small_size` differs from the other gates.
# The shared shapes, but this suite runs a smaller L1_SMALL pool than they carry and reports it in
# the measurement line, so the size is overridden rather than inherited.
MESHES = [
    pytest.param(shape, {**params, "l1_small_size": _L1_SMALL}, id=param.id)
    for param in GALAXY_MESHES
    for shape, params in [param.values]
]


REFERENCE_MEDIA = Path.home() / "h3_fl2va_artifacts" / "fl2va_first.mp4"


def reference_video() -> Path:
    if not REFERENCE_MEDIA.is_file():
        pytest.skip(f"no reference video at {REFERENCE_MEDIA}; place a clip with a soundtrack there")
    return REFERENCE_MEDIA


def ref2va_references(case: str) -> list[MiniMaxH3Reference]:
    """The reference set per e2e case (a fractal: content nothing in the prompt could produce)."""
    if case == "one_image":
        return [MiniMaxH3Reference(image=create_fractal_image(1024, 1024))]
    if case == "video_with_sound":
        return [reference_from_video_file(reference_video())]
    if case == "mixed":
        # image, then SILENT video, then bare audio: stresses split_condition_blocks' per-modality row cursors.
        sounded = reference_from_video_file(reference_video())
        return [
            MiniMaxH3Reference(image=create_fractal_image(1024, 1024)),
            reference_from_video_file(reference_video(), with_audio=False),
            MiniMaxH3Reference(audio=sounded.audio, sample_rate=sounded.sample_rate),
        ]
    raise ValueError(case)


def _real_frame_image() -> Image.Image:
    from ....pipelines.minimax_h3.packing_ref2va import decode_reference_video

    frames, _, _ = decode_reference_video(reference_video())
    return Image.fromarray(frames[0])


def _inverted(image: Image.Image) -> Image.Image:
    """Colour-inverted copy: same size, so the packed layout and noise stream are unchanged."""
    return Image.fromarray(255 - np.asarray(image.convert("RGB")))


def _clip_resemblance(output, image: Image.Image, num_frames: int = 8) -> float:
    from ...dataset_eval.clip_encoder import CLIPEncoder

    encoder = CLIPEncoder()
    frames = to_uint8_frames(output)
    indices = np.linspace(0, frames.shape[0] - 1, num_frames).round().astype(int)

    with torch.no_grad():
        reference_features = encoder.model.encode_image(encoder.preprocess(image).unsqueeze(0)).float()
        reference_features /= reference_features.norm(dim=-1, keepdim=True)
        scores = []
        for index in indices:
            frame = encoder.preprocess(Image.fromarray(frames[index])).unsqueeze(0)
            features = encoder.model.encode_image(frame).float()
            features /= features.norm(dim=-1, keepdim=True)
            scores.append(float((features @ reference_features.T).squeeze()))
    return float(np.mean(scores))


def _colour_distance(output, image: Image.Image) -> float:
    """Euclidean distance between mean RGBs, over [0, 1]. Logged only, never asserted."""
    output_mean = to_uint8_frames(output).reshape(-1, 3).mean(axis=0) / 255.0
    reference_mean = np.asarray(image.convert("RGB")).reshape(-1, 3).mean(axis=0) / 255.0
    return float(np.linalg.norm(output_mean - reference_mean))


def _divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean())


def _write(output, stem: str) -> dict:
    directory = artifact_dir("h3_ref2va_artifacts")
    frames = to_uint8_frames(output)
    for index in (0, 17, NUM_FRAMES // 2, NUM_FRAMES - 1):
        Image.fromarray(frames[index]).save(directory / f"{stem}_frame_{index}.png")
    paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, directory, stem=stem)
    log_quality(f"{stem}: artifacts in {directory} ({sorted(paths)})")
    return paths


def _record_quality(frames: np.ndarray, paths: dict, case: str) -> None:
    """CLIP prompt alignment and the five VBench dimensions, against the ref2va-calibrated bars."""
    gate_clip(frames, PROMPT, REF2VA_CLIP_THRESHOLD, f"QUALITY ref2va[{case}]")
    # A missing muxed mp4 skips rather than fails here, unlike t2va.
    gate_vbench(paths, PROMPT, REF2VA_VBENCH_THRESHOLDS, f"QUALITY ref2va[{case}]", skip_without_mp4=True)


def _pipeline(mesh_device) -> MiniMaxH3Pipeline:
    """Pipeline bound to the `transformer_ref` partition (62 GB, fixed at construction)."""
    return MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights_dir("transformer_ref"), task="ref2va"
    )


# Per-case padded sequence length, asserted so a case cannot silently drift off its probed shape.
# `mixed` measured 89856 e2e (a host estimate says 90112); one_image / video_with_sound are from the
# ref2va warm-latency measurements. If a case's assert trips, its message prints the actual length.
_EXPECTED_PADDED_LEN = {"one_image": 46080, "video_with_sound": 81664, "mixed": 89856}


@pytest.mark.timeout(10800)
@pytest.mark.parametrize("case", list(_EXPECTED_PADDED_LEN), ids=list(_EXPECTED_PADDED_LEN))
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_ref2va_end_to_end(case, mesh_device, reset_seeds):
    """Full ref2va generation per reference case (one_image / video_with_sound / mixed): geometry
    agreement across every reference path, plus warm latency."""
    references = ref2va_references(case)
    pipeline = _pipeline(mesh_device)

    # Warmup must use the SAME references: `padded_len` depends on them, and the helper asserts agreement.
    benchmark_profiler = BenchmarkProfiler()
    output = run_warm_generation(
        pipeline,
        PROMPT,
        references=references,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=STEPS,
        seed=SEED,
        profiler=benchmark_profiler,
    )

    assert output.video.shape == (1, 3, NUM_FRAMES, HEIGHT, WIDTH), tuple(output.video.shape)
    assert output.video.min() >= 0.0 and output.video.max() <= 1.0, "decoded video must be in [0, 1]"
    assert torch.isfinite(output.video).all() and torch.isfinite(output.audio).all()
    assert (
        pipeline.last_padded_len == _EXPECTED_PADDED_LEN[case]
    ), f"{case} ran at padded_len {pipeline.last_padded_len}, not the probed {_EXPECTED_PADDED_LEN[case]}"

    frames = to_uint8_frames(output)
    # Artifacts before the checks, so a failing check still leaves frames to inspect.
    paths = _write(output, f"ref2va_{case}")

    num_forwards = STEPS - 1
    log_pipeline_perf(
        benchmark_profiler,
        label=f"ref2va[{case}]",
        pipeline=pipeline,
        num_forwards=num_forwards,
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES,
        fps=FPS,
        num_inference_steps=STEPS,
        extra_lines=(f"L1 Small Size: {_L1_SMALL}",),
    )

    check_audio_sanity(
        output.audio, sampling_rate=output.sampling_rate, expected_seconds=NUM_FRAMES / FPS, tolerance_seconds=0.05
    )
    check_av_sync(frames, output.audio, sampling_rate=output.sampling_rate, fps=FPS)
    # Horizontal bar is 3.0: `video_with_sound` measures 2.29x there from scene content, not a seam.
    check_spatial_seams(frames, vertical_boundaries=(448, 896), horizontal_boundaries=(), max_ratio=2.0)
    check_spatial_seams(frames, vertical_boundaries=(), horizontal_boundaries=(384,), max_ratio=3.0)

    _record_quality(frames, paths, case)


@pytest.mark.timeout(9000)
@pytest.mark.parametrize(("mesh_device", "device_params"), MESHES, indirect=["mesh_device", "device_params"])
def test_ref2va_conditioning_is_not_a_no_op(mesh_device, reset_seeds):
    """Discriminator: a same-size inverted reference keeps the noise bit-identical, so any output
    divergence above the measured run-to-run floor is attributable to reference content alone."""
    pipeline = _pipeline(mesh_device)
    normal = _real_frame_image()
    inverted = _inverted(normal)

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

    a = generate(normal)
    a_repeat = generate(normal)
    b = generate(inverted)

    floor = _divergence(a.video, a_repeat.video)
    signal = _divergence(a.video, b.video)
    to_normal = (_clip_resemblance(a, normal), _clip_resemblance(b, normal))
    to_inverted = (_clip_resemblance(a, inverted), _clip_resemblance(b, inverted))
    colour = (_colour_distance(a, normal), _colour_distance(b, inverted))
    crossed = (_colour_distance(a, inverted), _colour_distance(b, normal))

    log_quality(
        f"ref2va discriminator: run-to-run floor {floor:.6f}, reference-swap signal {signal:.6f} "
        f"(ratio {signal / max(floor, 1e-9):.1f}x) | CLIP to normal A={to_normal[0]:.4f} "
        f"B={to_normal[1]:.4f} | CLIP to inverted A={to_inverted[0]:.4f} B={to_inverted[1]:.4f} | "
        f"colour distance to own reference A={colour[0]:.4f} B={colour[1]:.4f}, "
        f"to the other A={crossed[0]:.4f} B={crossed[1]:.4f}"
    )
    for stem, output in (("discriminate_normal", a), ("discriminate_inverted", b)):
        _write(output, stem)

    assert floor < 0.02, f"the same request twice diverged by {floor:.6f}; the pipeline is not reproducible"

    # The null hypothesis (reference ignored) scores signal == floor exactly, the noise being bit-identical.
    assert signal > 10 * floor, (
        f"swapping the reference moved the output by {signal:.6f} against a run-to-run floor of "
        f"{floor:.6f}: the reference is not conditioning the output"
    )
    assert signal > 0.01, (
        f"swapping the reference moved the output by only {signal:.6f} mean absolute pixel value; "
        "the effect is present but too small to call conditioning"
    )

    # Direction is logged, not asserted: no known instrument here can fail on a correct pipeline only.
    log_quality(
        f"ref2va direction (recorded, not asserted): CLIP own-vs-other "
        f"A {to_normal[0]:.4f} vs {to_inverted[0]:.4f}, B {to_inverted[1]:.4f} vs {to_normal[1]:.4f}; "
        f"colour own-vs-other A {colour[0]:.4f} vs {crossed[0]:.4f}, B {colour[1]:.4f} vs {crossed[1]:.4f}"
    )
