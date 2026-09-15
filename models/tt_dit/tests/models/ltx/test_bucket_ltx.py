# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the LTX trace bucket ladder: ladder shape, request routing, and rejections.

No device: everything here is integer arithmetic over the served (canvas, fps, duration) grid plus
the host side of the RoPE bucket padding.
"""

import pytest
import torch

from models.tt_dit.pipelines.ltx.pipeline_ltx import _ltx_temporal_chunk_plan, _stitch_ltx_temporal_chunks
from models.tt_dit.pipelines.ltx.pipeline_ltx_distilled import LTXDistilledPipeline
from models.tt_dit.utils.ltx import (
    LTX_AUDIO_N_BUCKET,
    LTX_BUCKET_ALIGN,
    LTX_BUCKET_LADDER,
    LTX_BUCKET_SP_FACTOR,
    LTX_CANVASES,
    LTX_DURATION_VALUES,
    LTX_FAST_AUDIO_N_BUCKET,
    LTX_FAST_1080P_25FPS_6S_LADDER,
    LTX_FPS_VALUES,
    LTX_OUTPUT_CANVASES,
    LTX_SERVED_CANVASES,
    ltx_aligned_num_frames,
    ltx_served_configs,
    ltx_stage_video_n_real,
    route_ltx_config,
    route_ltx_request,
    rungs_for_configs,
    select_bucket,
    validate_bucket_ladder,
)

SP = LTX_BUCKET_SP_FACTOR


def test_named_canvases_decode_aligned_and_export_at_nominal_resolution():
    assert LTX_CANVASES["720p-landscape"] == (768, 1280)
    assert LTX_OUTPUT_CANVASES["720p-landscape"] == (720, 1280)
    assert LTX_CANVASES["1080p-landscape"] == (1088, 1920)
    assert LTX_OUTPUT_CANVASES["1080p-landscape"] == (1080, 1920)


def test_fast_audio_bucket_fits_25fps_6s_and_rejects_25fps_10s():
    hot = route_ltx_config(
        "1080p-landscape",
        25,
        6,
        ladder=LTX_FAST_1080P_25FPS_6S_LADDER,
        audio_n_bucket=LTX_FAST_AUDIO_N_BUCKET,
    )
    assert (hot.audio_n_real, hot.audio_n) == (153, 256)
    with pytest.raises(ValueError, match="audio N=257 exceeds the audio bucket 256"):
        route_ltx_config(
            "1080p-landscape",
            25,
            10,
            ladder=LTX_FAST_1080P_25FPS_6S_LADDER,
            audio_n_bucket=LTX_FAST_AUDIO_N_BUCKET,
        )


def test_ladder_is_aligned_and_monotonic():
    validate_bucket_ladder(LTX_BUCKET_LADDER)
    assert LTX_BUCKET_ALIGN == 32 * SP
    assert all(rung % LTX_BUCKET_ALIGN == 0 for rung in LTX_BUCKET_LADDER)
    assert list(LTX_BUCKET_LADDER) == sorted(set(LTX_BUCKET_LADDER))
    # Six rungs cover a 30x token range, so adjacent rungs are ~2x (bounded pad waste).
    ratios = [b / a for a, b in zip(LTX_BUCKET_LADDER, LTX_BUCKET_LADDER[1:])]
    assert max(ratios) <= 2.01 and min(ratios) >= 1.9, ratios


@pytest.mark.parametrize(
    "bad_ladder, match",
    [
        ((), "must not be empty"),
        ((256, 256), "strictly increasing"),
        ((512, 256), "strictly increasing"),
        ((256, 300), "not a multiple"),
    ],
)
def test_validate_bucket_ladder_rejects(bad_ladder, match):
    with pytest.raises(ValueError, match=match):
        validate_bucket_ladder(bad_ladder)


def test_select_bucket_smallest_fitting_rung():
    assert select_bucket(1) == LTX_BUCKET_LADDER[0]
    assert select_bucket(LTX_BUCKET_LADDER[0]) == LTX_BUCKET_LADDER[0]
    assert select_bucket(LTX_BUCKET_LADDER[0] + 1) == LTX_BUCKET_LADDER[1]
    assert select_bucket(LTX_BUCKET_LADDER[-1]) == LTX_BUCKET_LADDER[-1]
    with pytest.raises(ValueError, match="exceeds the top bucket rung"):
        select_bucket(LTX_BUCKET_LADDER[-1] + 1)
    with pytest.raises(ValueError, match="must be positive"):
        select_bucket(0)


def test_aligned_num_frames_is_8k_plus_1_and_covers_duration():
    for fps in LTX_FPS_VALUES:
        for duration in LTX_DURATION_VALUES:
            n = ltx_aligned_num_frames(fps, duration)
            assert (n - 1) % 8 == 0
            assert n >= fps * duration
            assert n - fps * duration < 8


def test_temporal_vae_chunk_plan_and_stitch_cover_1001_frames():
    latent_frames = 126
    chunk_latents = 64
    overlap_latents = 4
    plan = _ltx_temporal_chunk_plan(latent_frames, chunk_latents, overlap_latents)
    assert plan == ((0, 65), (60, 125), (120, 126))

    # Model the causal boundary convention used by the reference decoder: after the first tile,
    # the final decoded frame is discarded and the next tile begins at the following sample frame.
    decoded = []
    for start, end in plan:
        sample_frames = chunk_latents * 8 + 1
        first_sample = start * 8 + (1 if start else 0)
        values = torch.arange(first_sample, first_sample + sample_frames, dtype=torch.float32)
        decoded.append(values.reshape(1, 1, -1, 1, 1))

    stitched = _stitch_ltx_temporal_chunks(
        decoded,
        num_sample_frames=1001,
        chunk_latents=chunk_latents,
        overlap_latents=overlap_latents,
    )
    assert stitched.shape == (1, 1, 1001, 1, 1)
    torch.testing.assert_close(stitched.flatten(), torch.arange(1001, dtype=torch.float32))


@pytest.mark.parametrize(
    "chunk, overlap, match",
    [(0, 0, "chunk_latents must be positive"), (64, 64, "overlap_latents must be")],
)
def test_temporal_vae_chunk_plan_rejects_invalid_config(chunk, overlap, match):
    with pytest.raises(ValueError, match=match):
        _ltx_temporal_chunk_plan(126, chunk, overlap)


def test_bucket_states_share_only_fixed_shape_deployment_io():
    pipeline = object.__new__(LTXDistilledPipeline)
    pipeline._reset_shared_trace_io()
    first = pipeline._new_trace_state(share_deployment_io=True)
    second = pipeline._new_trace_state(share_deployment_io=True)

    for name in pipeline.SHARED_TRACE_IO_NAMES:
        assert getattr(first, f"_{name}") is getattr(second, f"_{name}")
    for name in (
        "tt_video_lat",
        "tt_video_rope_cos",
        "tt_video_rope_sin",
        "tt_audio_rope_cos",
        "tt_audio_rope_sin",
        "tt_audio_cross_pe_cos",
        "tt_audio_cross_pe_sin",
        "tt_audio_attn_mask",
        "tt_audio_padding_mask",
        "tt_audio_pad_mask",
    ):
        assert getattr(first, f"_{name}") is not getattr(second, f"_{name}")


def test_bound_per_rung_statics_do_not_refresh_when_another_rung_replays():
    pipeline = object.__new__(LTXDistilledPipeline)
    pipeline._reset_shared_trace_io()
    state = pipeline._new_trace_state(share_deployment_io=True)
    state.bound_config = (20, 17, 30, 10240, 10200, 256, 153, 25)

    # Returns before touching pipeline model fields: another rung's dynamic-arena update must not
    # invalidate this rung's already-bound RoPE/cross-PE/mask metadata.
    pipeline._prepare_stage_statics(
        state,
        latent_frames=20,
        latent_h=17,
        latent_w=30,
        video_N=10240,
        video_N_real=10200,
        audio_N=256,
        audio_N_real=153,
        fps=25,
        sp_axis=1,
        traced=True,
    )


def test_served_grid_routes_onto_every_rung():
    configs = ltx_served_configs()
    # 4 served canvases (720p/1080p, landscape+portrait) x 4 fps x 8 durations.
    assert len(configs) == len(LTX_SERVED_CANVASES) * len(LTX_FPS_VALUES) * len(LTX_DURATION_VALUES) == 128
    # Portrait and landscape share token counts (and e.g. 24fps/12s == 48fps/6s in frames), so the
    # distinct token shapes are exactly the distinct (tier pixel count, aligned frames) pairs.
    distinct = {
        tuple(ltx_stage_video_n_real(ltx_aligned_num_frames(fps, d), *LTX_CANVASES[c]).values())
        for c, fps, d in configs
    }
    tiers = {(LTX_CANVASES[c][0] * LTX_CANVASES[c][1], ltx_aligned_num_frames(fps, d)) for c, fps, d in configs}
    assert len(distinct) == len(tiers) == 52
    rungs = rungs_for_configs(configs)
    assert rungs == LTX_BUCKET_LADDER, "the shipped ladder should be exactly the set of rungs the grid needs"
    assert len(rungs) == 6


def test_route_extremes_of_the_served_grid():
    top = route_ltx_config("1080p-landscape", 50, 20)
    assert top.num_frames == 1001 and top.latent_frames == 126
    assert top.stage_video_n_real == {"s1": 64260, "s2": 257040}
    assert top.stage_rung == {"s1": 67840, "s2": 261120}
    assert top.trace_key("s2") == LTX_BUCKET_LADDER[-1]
    assert top.audio_n_real == 500 and top.audio_n == LTX_AUDIO_N_BUCKET

    bottom = route_ltx_config("720p-landscape", 24, 6)
    assert bottom.num_frames == 145
    assert bottom.stage_video_n_real == {"s1": 4560, "s2": 18240}
    assert bottom.stage_rung == {"s1": 8704, "s2": 34560}
    assert bottom.audio_n_real == 151

    portrait = route_ltx_config("720p-portrait", 24, 6)
    assert portrait.stage_rung == bottom.stage_rung


def test_route_padding_never_exceeds_one_rung_and_stays_within_ladder():
    for canvas, fps, duration in ltx_served_configs():
        route = route_ltx_config(canvas, fps, duration)
        for stage in ("s1", "s2"):
            n_real, rung = route.video_n_real(stage), route.video_n(stage)
            assert n_real <= rung
            below = [r for r in LTX_BUCKET_LADDER if r < rung]
            if below:
                assert n_real > below[-1], f"{canvas} {fps}fps {duration}s {stage} should land on {below[-1]}"
        assert route.audio_n_real <= route.audio_n


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(num_frames=145, height=2176, width=3840, fps=24, sp_factor=SP), "not served"),
        (dict(num_frames=145, height=1440, width=2560, fps=24, sp_factor=SP), "not served"),
        (dict(num_frames=144, height=768, width=1280, fps=24, sp_factor=SP), "8k\\+1"),
        (dict(num_frames=145, height=768, width=1280, fps=30, sp_factor=SP), "fps 30 is not served"),
        (dict(num_frames=145, height=768, width=1280, fps=24, sp_factor=4), "SP=8"),
        (dict(num_frames=145, height=768, width=1280, fps=24, sp_factor=SP, mode="video"), "AV mode only"),
        (dict(num_frames=145, height=768, width=1280, fps=24, sp_factor=SP, image_conditioned=True), "T2V only"),
    ],
)
def test_route_rejections(kwargs, match):
    with pytest.raises(ValueError, match=match):
        route_ltx_request(**kwargs)


def test_route_any_canvas_when_unrestricted():
    route = route_ltx_request(num_frames=121, height=512, width=768, fps=24, sp_factor=SP, served_canvases=None)
    assert route.canvas == "512x768"
    assert route.stage_video_n_real == {"s1": 16 * 8 * 12, "s2": 16 * 16 * 24}
    assert route.stage_rung == {"s1": 8704, "s2": 8704}
    with pytest.raises(ValueError, match="multiples of 64"):
        route_ltx_request(num_frames=121, height=500, width=768, fps=24, sp_factor=SP, served_canvases=None)


def test_route_rejects_above_ladder_top():
    with pytest.raises(ValueError, match="exceeds the top bucket rung"):
        route_ltx_request(num_frames=1001, height=1088, width=1920, fps=50, sp_factor=SP, ladder=LTX_BUCKET_LADDER[:-1])


def test_pad_video_rope_sp_explicit_bucket():
    ttnn = pytest.importorskip("ttnn")
    from models.tt_dit.models.transformers.ltx.rope_ltx import pad_video_rope_sp

    H, n_real, d_half = 2, 300, 8
    cos = torch.rand(1, H, n_real, d_half)
    sin = torch.rand(1, H, n_real, d_half)
    align = ttnn.TILE_SIZE * SP

    # Default: SP boundary.
    c, s = pad_video_rope_sp(cos, sin, SP)
    assert c.shape[2] == s.shape[2] == 512
    # Explicit rung: padded slots are the identity rotation.
    c, s = pad_video_rope_sp(cos, sin, SP, video_N=8704)
    assert c.shape[2] == s.shape[2] == 8704
    assert torch.equal(c[:, :, :n_real], cos) and torch.equal(s[:, :, :n_real], sin)
    assert torch.all(c[:, :, n_real:] == 1.0) and torch.all(s[:, :, n_real:] == 0.0)
    # Exact fit is a no-op.
    c, s = pad_video_rope_sp(cos[:, :, :align], sin[:, :, :align], SP, video_N=align)
    assert c.shape[2] == align
    with pytest.raises(ValueError, match="smaller than logical"):
        pad_video_rope_sp(cos, sin, SP, video_N=256)
    with pytest.raises(ValueError, match="divisible by TILE_SIZE"):
        pad_video_rope_sp(cos, sin, SP, video_N=8704 + 32)


@pytest.mark.parametrize("shape,fps", [((19, 17, 30), 24), ((20, 34, 60), 25)])
def test_compact_video_rope_expands_to_full_reference(shape, fps):
    from models.tt_dit.models.transformers.ltx.rope_ltx import (
        LTXRopeType,
        expand_compact_video_rope,
        pad_video_rope_sp,
        precompute_freqs_cis,
        prepare_compact_video_rope,
        reshape_interleaved_to_bhnd,
    )
    from models.tt_dit.utils.patchifiers import VideoLatentShape, get_pixel_coords, video_get_patch_grid_bounds

    F, H, W = shape
    positions = get_pixel_coords(
        video_get_patch_grid_bounds(VideoLatentShape(1, 128, F, H, W)),
        scale_factors=(8, 32, 32),
        causal_fix=True,
    ).float()
    positions[:, 0] /= fps
    kwargs = dict(
        out_dtype=torch.float32,
        theta=10000.0,
        use_middle_indices_grid=True,
        num_attention_heads=32,
        rope_type=LTXRopeType.INTERLEAVED,
    )
    self_cos, self_sin = precompute_freqs_cis(positions, dim=4096, max_pos=[20, 2048, 2048], **kwargs)
    cross_cos, cross_sin = precompute_freqs_cis(positions[:, 0:1], dim=2048, max_pos=[20], **kwargs)
    reference = tuple(reshape_interleaved_to_bhnd(value, 32) for value in (self_cos, self_sin, cross_cos, cross_sin))

    video_N = ((F * H * W + 255) // 256) * 256
    reference = (
        *pad_video_rope_sp(reference[0], reference[1], SP, video_N=video_N),
        *pad_video_rope_sp(reference[2], reference[3], SP, video_N=video_N),
    )
    compact = prepare_compact_video_rope(
        F,
        H,
        W,
        inner_dim=4096,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        fps=fps,
        axis_capacity=64,
    )
    actual = expand_compact_video_rope(compact, inner_dim=4096, num_attention_heads=32, video_N=video_N)

    # Vectorized CPU cos/sin can differ by about one ulp depending on
    # whether equal phases are evaluated compactly or after token expansion.
    for index, (got, expected) in enumerate(zip(actual, reference)):
        torch.testing.assert_close(got, expected, rtol=0, atol=1e-3)
        identity = 1 if index in (0, 2) else 0
        assert torch.all(got[:, :, F * H * W :] == identity)


def test_compact_video_rope_capacity_and_fps_contract():
    from models.tt_dit.models.transformers.ltx.rope_ltx import prepare_compact_video_rope

    kwargs = dict(inner_dim=4096, theta=10000.0, max_pos=[20, 2048, 2048], axis_capacity=64)
    at_24 = prepare_compact_video_rope(20, 34, 60, fps=24, **kwargs)
    at_25 = prepare_compact_video_rope(20, 34, 60, fps=25, **kwargs)
    assert at_24.self_cos.shape == (3, 64, 682)
    assert at_24.cross_cos.shape == (64, 1024)
    assert not torch.equal(at_24.self_cos[0, :20], at_25.self_cos[0, :20])
    assert torch.equal(at_24.self_cos[1:], at_25.self_cos[1:])
    assert torch.all(at_24.self_cos[0, 20:] == 1) and torch.all(at_24.self_sin[0, 20:] == 0)
    with pytest.raises(ValueError, match="axis_capacity"):
        prepare_compact_video_rope(20, 34, 60, fps=24, **(kwargs | {"axis_capacity": 59}))
