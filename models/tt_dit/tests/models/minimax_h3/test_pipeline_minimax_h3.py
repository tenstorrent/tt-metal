# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""End-to-end `t2va` quality gate, swept over the published working points.

Six aspect ratios (21:9 .. 9:16) x three durations (5 / 10 / 15 s), 50 steps. The canvas comes from
`resolve_canvas_size` -- short edge 768 from 16:9 through 9:16, ~1 MPix for wider -- and the frame
count from `align_num_frames`, so neither is tabulated here. Each case writes its own artifact stem,
so a sweep does not overwrite itself.

Pipeline wall-clock lives in `test_performance_minimax_h3.py`. Sanity, seam, CLIP, and reminder logs
are silent unless `H3_LOG_QUALITY=1`.
"""

from __future__ import annotations

import os

import pytest
from loguru import logger

import ttnn

from ....models.transformers.minimax_h3.length_classes import length_class
from ....pipelines.minimax_h3.packing import MINIMAX_H3_FPS, resolve_canvas_size
from ....pipelines.minimax_h3.pipeline_minimax_h3 import (
    _DIT_FSDP_ENV,
    _DYNAMIC_LENGTH_ENV,
    _PRESETS_BH,
    _PRESETS_WH,
    MiniMaxH3Pipeline,
    _DenoisePath,
    _requested_audio_t_factor,
    _requested_dynamic_length,
    _resolve_audio_t_shard,
    _resolve_denoise_path,
    _resolve_dit_fsdp,
    validate_bucket_ladder,
)
from ....pipelines.minimax_h3.policy import align_num_frames, get_num_frames
from ..wan2_2.common import check_output_sanity
from .common import GALAXY_MESHES, MESH_4X8_RING
from .common_av import (
    CALIBRATED_FOX_PROMPT,
    artifact_dir,
    assert_generation_ok,
    check_audio_sanity,
    check_av_sync,
    check_spatial_seams,
    check_written_file,
    gate_clip,
    gate_vbench,
    is_host,
    log_quality,
    log_quality_warning,
    log_spectral_flatness,
    quality_logs_enabled,
    run_warm_generation,
    to_uint8_frames,
    weights_dir,
    write_artifacts,
)

NUM_INFERENCE_STEPS = 50
SEED = 0

# The published working points: short edge 768 from 16:9 through 9:16, and ~1 MPix for anything
# wider. `resolve_canvas_size` already implements exactly that (768 short edge, area capped at
# 768*1344, both axes snapped to 32), so the canvas is derived rather than tabulated here --
# 21:9 lands on 672x1536, which is the documented example.
ASPECT_RATIOS = [(21, 9), (16, 9), (4, 3), (1, 1), (3, 4), (9, 16)]
DURATIONS_S = [5, 10, 15]


# calibrated 2026-08-04, fox prompt, seed 0 (single sample; margins are generous)
VBENCH_THRESHOLDS = {
    "subject_consistency": 0.95,
    "background_consistency": 0.95,
    "motion_smoothness": 0.97,
    "dynamic_degree": 1.0,
    "imaging_quality": 0.64,
}
CLIP_THRESHOLD = 33.0  # calibrated 2026-08-04, fox prompt, seed 0

# tier-6 thresholds are calibrated against this exact prompt; swapping it invalidates both bars.
PROMPT = CALIBRATED_FOX_PROMPT

SWEEP = [
    pytest.param(ratio, seconds, id=f"{ratio[0]}x{ratio[1]}_{seconds}s")
    for seconds in DURATIONS_S
    for ratio in ASPECT_RATIOS
]


# Pure (no-device) coverage for the audio T-shard resolver: axis selection, tie-break, 8->4->1 fallback.
@pytest.mark.parametrize(
    ("requested", "mesh_shape", "tp_axis", "sp_axis", "expected"),
    [
        (8, (4, 8), 0, 1, (8, 1)),  # factor 8 -> size-8 SP axis
        (8, (8, 4), 1, 0, (8, 0)),  # reversed 8x4 -> size-8 axis 0
        (4, (4, 8), 0, 1, (4, 0)),  # factor 4 -> size-4 TP axis
        (8, (4, 32), 0, 1, (4, 0)),  # no size-8 axis -> fall back to 4
        (8, (1, 1), 0, 1, (1, None)),  # single device -> unsharded
        (4, (4, 4), 0, 1, (4, 0)),  # equal axes -> prefer TP
        (1, (4, 8), 0, 1, (1, None)),  # explicit unsharded
        (2, (4, 8), 0, 1, (1, None)),  # below the 4/8 chain -> unsharded
        (32, (4, 32), 0, 1, (32, 1)),  # opt-in 32 on a quad -> the 32-wide inter-host axis
        (32, (4, 8), 0, 1, (8, 1)),  # requesting 32 on a single galaxy -> falls to 8
    ],
)
def test_resolve_audio_t_shard(requested, mesh_shape, tp_axis, sp_axis, expected):
    assert _resolve_audio_t_shard(requested, mesh_shape, tp_axis, sp_axis) == expected


def test_requested_audio_t_factor_precedence(monkeypatch, expect_error):
    # explicit kwarg wins over the env var
    monkeypatch.setenv("MINIMAX_H3_AUDIO_T_FACTOR", "32")
    assert _requested_audio_t_factor(4) == (4, False)
    # env var fills the default when the kwarg is None
    assert _requested_audio_t_factor(None) == (32, True)
    # no kwarg, no env -> default 8
    monkeypatch.delenv("MINIMAX_H3_AUDIO_T_FACTOR", raising=False)
    assert _requested_audio_t_factor(None) == (8, False)
    # a non-integer env value is a clear error, not a silent fallback
    monkeypatch.setenv("MINIMAX_H3_AUDIO_T_FACTOR", "thirty-two")
    with expect_error(ValueError, "MINIMAX_H3_AUDIO_T_FACTOR"):
        _requested_audio_t_factor(None)


def test_requested_dynamic_length_precedence(monkeypatch, expect_error):
    monkeypatch.delenv(_DYNAMIC_LENGTH_ENV, raising=False)
    # no kwarg, no env -> the given default
    assert _requested_dynamic_length(None, True) == (True, False)
    assert _requested_dynamic_length(None, False) == (False, False)
    # the env fills a None kwarg either way
    monkeypatch.setenv(_DYNAMIC_LENGTH_ENV, "0")
    assert _requested_dynamic_length(None, True) == (False, True)
    monkeypatch.setenv(_DYNAMIC_LENGTH_ENV, "1")
    assert _requested_dynamic_length(None, False) == (True, True)
    # explicit kwarg wins over the env
    assert _requested_dynamic_length(False, True) == (False, False)
    monkeypatch.setenv(_DYNAMIC_LENGTH_ENV, "0")
    assert _requested_dynamic_length(True, False) == (True, False)
    # anything but '0'/'1' is a clear error
    monkeypatch.setenv(_DYNAMIC_LENGTH_ENV, "yes")
    with expect_error(ValueError, _DYNAMIC_LENGTH_ENV):
        _requested_dynamic_length(None, True)


# Pure (no-device) coverage of which denoise path the constructor kwargs select on each preset.
def _denoise_path(preset, *, coresident=None, **kwargs):
    """(trace_denoise, bucket_denoise, dynamic_length) for the kwargs given; the rest default to None."""
    kwargs = dict(trace_denoise=None, bucket_denoise=None, bucket_ladder=None, dynamic_length=None) | kwargs
    coresident = preset.get("coresident", True) if coresident is None else coresident
    path = _resolve_denoise_path(preset, coresident=coresident, **kwargs)
    return path.trace_denoise, path.bucket_denoise, path.dynamic_length


def test_resolve_denoise_path_bh_4x8(monkeypatch, expect_error):
    monkeypatch.delenv(_DYNAMIC_LENGTH_ENV, raising=False)
    preset = _PRESETS_BH[(4, 8)]
    dynamic, bucketed, traced = (False, False, True), (False, True, False), (True, True, False)
    unbucketed = (False, False, False)
    # the default is dynamic, and any bucketed kwarg keeps the backup path
    assert _denoise_path(preset) == dynamic
    assert _denoise_path(preset, trace_denoise=True) == traced
    assert _denoise_path(preset, bucket_denoise=True) == bucketed
    assert _denoise_path(preset, bucket_ladder=preset["bucket_ladder"]["t2va"]) == bucketed
    assert _denoise_path(preset, trace_denoise=False) == bucketed
    assert _denoise_path(preset, bucket_denoise=False) == unbucketed
    assert _denoise_path(preset, dynamic_length=False) == bucketed
    # an explicit dynamic request: a ladder only supplies the cap, tracing or bucketing contradicts it
    assert _denoise_path(preset, dynamic_length=True, bucket_ladder=preset["bucket_ladder"]["t2va"]) == dynamic
    with expect_error(ValueError, "eager-only"):
        _denoise_path(preset, dynamic_length=True, trace_denoise=True)
    with expect_error(ValueError, "both denoise paths"):
        _denoise_path(preset, dynamic_length=True, bucket_denoise=True)
    # env: below the kwarg, above the preset
    monkeypatch.setenv(_DYNAMIC_LENGTH_ENV, "0")
    assert _denoise_path(preset) == bucketed
    assert _denoise_path(preset, bucket_denoise=True) == bucketed
    assert _denoise_path(preset, dynamic_length=True) == dynamic
    monkeypatch.setenv(_DYNAMIC_LENGTH_ENV, "1")
    assert _denoise_path(preset) == dynamic
    assert _denoise_path(preset, dynamic_length=False) == bucketed
    with expect_error(ValueError, "eager-only"):
        _denoise_path(preset, trace_denoise=True)


def test_resolve_denoise_path_quad_and_wormhole(monkeypatch, expect_error):
    monkeypatch.delenv(_DYNAMIC_LENGTH_ENV, raising=False)
    quad, wormhole = _PRESETS_BH[(4, 32)], _PRESETS_WH[(4, 8)]
    # both stay bucketed; the quad buckets through its trace, which evicting stages drops
    assert _denoise_path(quad) == (True, True, False)
    assert _denoise_path(quad, coresident=False) == (False, False, False)
    assert _denoise_path(wormhole) == (False, True, False)
    assert _denoise_path(quad, dynamic_length=False) == (True, True, False)
    monkeypatch.setenv(_DYNAMIC_LENGTH_ENV, "0")
    assert _denoise_path(quad) == (True, True, False)
    # the switch cannot make the quad dynamic
    monkeypatch.setenv(_DYNAMIC_LENGTH_ENV, "1")
    with expect_error(ValueError, "eager-only"):
        _denoise_path(quad)
    assert _denoise_path(wormhole) == (False, False, True)


def test_resolve_dit_fsdp(monkeypatch, expect_error):
    monkeypatch.delenv(_DIT_FSDP_ENV, raising=False)
    bh, quad, wormhole = _PRESETS_BH[(4, 8)], _PRESETS_BH[(4, 32)], _PRESETS_WH[(4, 8)]
    assert "dit_fsdp" not in bh and "dit_fsdp" not in quad
    env = f"from {_DIT_FSDP_ENV}"
    # dynamic ref2va shards the DiT; bucketed ref2va and dynamic t2va stay unsharded
    assert _resolve_dit_fsdp(bh, dit_fsdp=None, task="ref2va", dynamic_length=True) == (True, "dynamic ref2va")
    assert _resolve_dit_fsdp(bh, dit_fsdp=None, task="ref2va", dynamic_length=False) == (False, "default")
    assert _resolve_dit_fsdp(bh, dit_fsdp=None, task="t2va", dynamic_length=True) == (False, "default")
    assert _resolve_dit_fsdp(quad, dit_fsdp=None, task="ref2va", dynamic_length=False) == (False, "default")
    # the kwarg and a preset key both win over the rule
    assert _resolve_dit_fsdp(bh, dit_fsdp=False, task="ref2va", dynamic_length=True) == (False, "kwarg")
    assert _resolve_dit_fsdp(bh, dit_fsdp=True, task="t2va", dynamic_length=False) == (True, "kwarg")
    assert _resolve_dit_fsdp(wormhole, dit_fsdp=None, task="t2va", dynamic_length=False) == (True, "preset")
    # the env overrides all of them
    monkeypatch.setenv(_DIT_FSDP_ENV, "0")
    assert _resolve_dit_fsdp(bh, dit_fsdp=None, task="ref2va", dynamic_length=True) == (False, env)
    assert _resolve_dit_fsdp(wormhole, dit_fsdp=True, task="ref2va", dynamic_length=False) == (False, env)
    monkeypatch.setenv(_DIT_FSDP_ENV, "1")
    assert _resolve_dit_fsdp(bh, dit_fsdp=False, task="t2va", dynamic_length=False) == (True, env)
    monkeypatch.setenv(_DIT_FSDP_ENV, "yes")
    with expect_error(ValueError, _DIT_FSDP_ENV):
        _resolve_dit_fsdp(bh, dit_fsdp=None, task="t2va", dynamic_length=False)


# Pure (no-device) coverage for the per-mesh bucket ladders.
@pytest.mark.parametrize("task", ["t2va", "ref2va"])
def test_preset_bucket_ladder_is_aligned(task):
    preset = _PRESETS_BH[(4, 8)]
    validate_bucket_ladder(preset["bucket_ladder"][task], 8 * ttnn.TILE_SIZE)
    # dynamic by default; the ladder stays valid as the backup path and supplies the admission cap
    assert preset["dynamic_length"] and preset["bucket_denoise"] and not preset.get("trace_denoise")


@pytest.mark.parametrize("task", ["t2va", "ref2va"])
def test_quad_preset_bucket_ladder_is_aligned(task):
    validate_bucket_ladder(_PRESETS_BH[(4, 32)]["bucket_ladder"][task], 32 * ttnn.TILE_SIZE)


def test_presets_never_trace_a_dynamic_length():
    for presets in (_PRESETS_BH, _PRESETS_WH):
        for shape, preset in presets.items():
            assert not (preset.get("dynamic_length") and preset.get("trace_denoise")), shape
    # dynamic length is measured on the Blackhole 4x8 only
    assert not _PRESETS_BH[(4, 32)].get("dynamic_length") and not _PRESETS_WH[(4, 8)].get("dynamic_length")


def test_denoise_path_name():
    """The construction log's name for each path."""
    assert _DenoisePath(False, False, True, False).name == "dynamic length"
    assert _DenoisePath(True, True, False, False).name == "bucketed, traced"
    assert _DenoisePath(False, True, False, False).name == "bucketed"
    assert _DenoisePath(False, False, False, False).name == "unbucketed"


# Pure (no-device) coverage of the dynamic-length padding and admission rule.
def test_dynamic_padded_length_admission(expect_error):
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.sp_factor, pipeline.dynamic_length, pipeline.max_sequence_length = 8, True, 120320
    pipeline._force_pad_to = None
    assert pipeline._padded_length(37749) == 37888  # 256-aligned, not a ladder rung
    assert pipeline._padded_length(120320) == 120320
    with expect_error(ValueError, "beyond the longest served packed length"):
        pipeline._padded_length(120321)
    pipeline._force_pad_to = 120320  # the warm-up's capacity point
    assert pipeline._padded_length(37749) == 120320
    pipeline._force_pad_to = 18944
    with expect_error(ValueError, "smaller than the packed length"):
        pipeline._padded_length(37749)


# The dynamic-length gate: one program set per length class must serve every length in it. The cheap form runs by
# default on the Blackhole 4x8 (`MINIMAX_H3_DYNAMIC_GATE=0` opts out); `=1` adds the hour-long full-warm form.
DYNAMIC_GATE_ENV = "MINIMAX_H3_DYNAMIC_GATE"
# (quiet warm, measured) durations inside each length class of `length_classes.py` at 16:9.
DYNAMIC_GATE_CLASS_POINTS_S = [(4, 5), (10, 15)]


def _assert_no_denoise_misses(pipeline, warm_s: int, measured_s: int) -> None:
    misses = pipeline.last_program_cache_misses
    assert (
        misses["denoising"] == 0
    ), f"the {measured_s} s request compiled {misses['denoising']} denoise programs after the {warm_s} s warm: {misses}"


@pytest.mark.skipif(os.environ.get(DYNAMIC_GATE_ENV) == "0", reason=f"{DYNAMIC_GATE_ENV}=0 opts out of the gate")
@pytest.mark.timeout(5400)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_t2va_dynamic_length_zero_misses(mesh_device, reset_seeds):
    """Per length class: one quiet request, then a measured one at a new padded length in the same class, which
    must add no denoise program. No init warm-up, so the class's first request is its only compile; audio is
    untraced because the tracer captures on its first call."""
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    height, width = resolve_canvas_size(16, 9)
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        vae_output_type="yuv420",
        dynamic_length=True,
        warmup=False,
        trace_audio=False,
    )
    assert pipeline.dynamic_length and not pipeline.bucket_denoise
    for warm_s, measured_s in DYNAMIC_GATE_CLASS_POINTS_S:
        with pipeline.quiet():
            pipeline(
                PROMPT, num_frames=get_num_frames(warm_s), height=height, width=width, num_inference_steps=3, seed=SEED
            )
        warm_rows = pipeline.last_seq_len.padded // pipeline.sp_factor
        output = pipeline(
            PROMPT, num_frames=get_num_frames(measured_s), height=height, width=width, num_inference_steps=3, seed=SEED
        )
        measured_rows = pipeline.last_seq_len.padded // pipeline.sp_factor
        assert warm_rows != measured_rows, f"{warm_s} s and {measured_s} s both pad to {warm_rows} rows/device"
        assert length_class(warm_rows) == length_class(measured_rows), f"{warm_s} s and {measured_s} s span two classes"
        assert_generation_ok(output, align_num_frames(get_num_frames(measured_s)))
        _assert_no_denoise_misses(pipeline, warm_s, measured_s)
    pipeline.release_traces()


@pytest.mark.skipif(
    os.environ.get(DYNAMIC_GATE_ENV) != "1",
    reason=f"the full-warm gate (init warm-up, then a cross-class measure) runs on request: set {DYNAMIC_GATE_ENV}=1",
)
@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("mesh_device", "device_params"), [MESH_4X8_RING], indirect=["mesh_device", "device_params"])
def test_t2va_dynamic_length_zero_misses_full_warm(mesh_device, reset_seeds):
    """The served configuration: the init warm-up, a quiet 5 s request, then a measured 10 s one across the class
    boundary, which must add no denoise program."""
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    height, width = resolve_canvas_size(16, 9)
    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device, weights_dir=weights, vae_output_type="yuv420", dynamic_length=True
    )
    assert pipeline.dynamic_length and not pipeline.bucket_denoise
    with pipeline.quiet():
        pipeline(PROMPT, num_frames=get_num_frames(5), height=height, width=width, num_inference_steps=3, seed=SEED)
    output = pipeline(
        PROMPT, num_frames=get_num_frames(10), height=height, width=width, num_inference_steps=3, seed=SEED
    )
    assert_generation_ok(output, align_num_frames(get_num_frames(10)))
    _assert_no_denoise_misses(pipeline, 5, 10)
    pipeline.release_traces()


@pytest.mark.timeout(7200)
@pytest.mark.parametrize(("aspect_ratio", "duration_s"), SWEEP)
@pytest.mark.parametrize(("mesh_device", "device_params"), GALAXY_MESHES, indirect=["mesh_device", "device_params"])
def test_t2va_end_to_end(mesh_device, reset_seeds, aspect_ratio, duration_s):
    weights = weights_dir("transformer", "text_encoder", "vae", "audio_vae")
    artifacts = artifact_dir("h3_t2va_artifacts")
    prompt = PROMPT

    HEIGHT, WIDTH = resolve_canvas_size(*aspect_ratio)
    NUM_FRAMES = get_num_frames(duration_s)
    # One artifact per working point, so a sweep does not overwrite itself.
    stem = f"t2va_{aspect_ratio[0]}x{aspect_ratio[1]}_{WIDTH}x{HEIGHT}_{duration_s}s"
    if is_host():
        logger.info(
            f"working point: {aspect_ratio[0]}:{aspect_ratio[1]} -> {WIDTH}x{HEIGHT}, {NUM_FRAMES} frames, {stem}"
        )

    # A missing dependency must report SKIPPED before the long generation, never silently pass as green.
    pytest.importorskip("open_clip", reason="the CLIP gate needs open_clip, which is not installed")

    if is_host() and not os.environ.get("TT_DIT_CACHE_DIR"):
        logger.warning(
            "TT_DIT_CACHE_DIR is unset, so every weight load reads safetensors and the run will take far longer."
        )

    pipeline = MiniMaxH3Pipeline.create_pipeline(
        mesh_device=mesh_device,
        weights_dir=weights,
        vae_output_type="float",
    )

    output = run_warm_generation(
        pipeline,
        prompt,
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )

    expected_frames = align_num_frames(NUM_FRAMES)
    log_quality(
        f"generated {output.num_frames} frames ({output.video_seconds:.3f} s) and "
        f"{output.audio_seconds:.3f} s of audio at {output.sampling_rate} Hz"
    )

    frames = to_uint8_frames(output)

    check_output_sanity(
        frames,
        num_frames=expected_frames,
        height=HEIGHT,
        width=WIDTH,
        log=quality_logs_enabled() and is_host(),
    )
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

    # Boundaries from the VAE's own tile grid; re-deriving them with a wrong overlap gates non-boundaries.
    ratio = pipeline.vae_config.spatial_compression_ratio
    (y_starts, _, _), (x_starts, _, _) = pipeline.vae.decode_tile_grid(HEIGHT // ratio, WIDTH // ratio)
    check_spatial_seams(frames, vertical_boundaries=x_starts[1:], horizontal_boundaries=y_starts[1:])

    # Rank 0 alone writes and scores, and its outcome is held rather than raised: every other rank
    # goes straight to the barrier below, so a gate failure, a VBench `pytest.skip` or an artifact I/O
    # error raised here would leave them blocked and skip the trace release. `pytest.skip.Exception`
    # is named alongside `Exception` because `Skipped` does not derive from it.
    outcome: BaseException | None = None
    is_distributed = ttnn.using_distributed_env()
    if is_host():
        try:
            paths = write_artifacts(frames, output.audio.cpu().numpy(), output.sampling_rate, artifacts, stem=stem)
            check_written_file(paths, expected_frames, height=HEIGHT, width=WIDTH)

            # CLIP_THRESHOLD was measured at 16:9 / 5 s. Applying it across the sweep is an extrapolation,
            # but a generous one: the bar sits well under the calibrated point, and the score is a
            # prompt-alignment number rather than a resolution-dependent one. A prompt change would
            # invalidate it outright -- recalibrate before swapping PROMPT.
            gate_clip(frames, prompt, CLIP_THRESHOLD, stem)
            # RUN_VBENCH=0 drops the VBench gate. It needs its own interpreter (~/vbench_env, pinned to
            # numpy<2 / transformers 4.33), and without one `run_vbench` skips -- which marks the whole
            # test SKIPPED *after* the full generation, hiding the A/V results behind a non-result.
            # Off means "not measured": everything above still gates.
            if os.environ.get("RUN_VBENCH", "1") not in ("0", "false", "False"):
                gate_vbench(paths, prompt, VBENCH_THRESHOLDS, stem)
            else:
                log_quality_warning("RUN_VBENCH=0, so the VBench gate did not run; generative quality is UNMEASURED")

            log_quality(f"artifacts in {artifacts}: {sorted(p.name for p in artifacts.iterdir())}")
        except (Exception, pytest.skip.Exception) as exc:
            outcome = exc
    if is_distributed:
        ttnn.distributed_context_barrier()
    log_quality(
        "REMINDER: read the artifact rubric against these frames -- the seam and flicker scores above "
        "are statistics, and only looking at the output catches what they average away"
    )
    # Every rank, and before the fixture closes the mesh: a captured trace holds device buffers
    # for the request. Only the 4x32 preset traces, so this is a no-op at 4x8.
    pipeline.release_traces()
    # Now that every rank has resynchronised and released its trace, rank 0 reports. mpirun surfaces
    # a single failing rank as a failing job, so the run fails rather than hanging.
    if outcome is not None:
        raise outcome
    assert sync["video_seconds"] > 0
