# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for the LTX trace bucket ladder: ladder shape, request routing, and rejections.

No device: everything here is integer arithmetic over the served (canvas, fps, duration) grid plus
the host side of the RoPE bucket padding.
"""

import pytest
import torch

from models.tt_dit.utils.ltx import (
    LTX_AUDIO_N_BUCKET,
    LTX_BUCKET_ALIGN,
    LTX_BUCKET_LADDER,
    LTX_BUCKET_SP_FACTOR,
    LTX_CANVASES,
    LTX_DURATION_VALUES,
    LTX_FPS_VALUES,
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


def test_ladder_is_aligned_and_monotonic():
    validate_bucket_ladder(LTX_BUCKET_LADDER)
    assert LTX_BUCKET_ALIGN == 32 * SP
    assert all(rung % LTX_BUCKET_ALIGN == 0 for rung in LTX_BUCKET_LADDER)
    assert list(LTX_BUCKET_LADDER) == sorted(set(LTX_BUCKET_LADDER))
    # ~1.4x geometric spacing: no rung more than 1.5x its predecessor (bounded pad waste).
    ratios = [b / a for a, b in zip(LTX_BUCKET_LADDER, LTX_BUCKET_LADDER[1:])]
    assert max(ratios) <= 1.5 and min(ratios) >= 1.3, ratios


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
    assert len(rungs) == 11


def test_route_extremes_of_the_served_grid():
    top = route_ltx_config("1080p-landscape", 50, 20)
    assert top.num_frames == 1001 and top.latent_frames == 126
    assert top.stage_video_n_real == {"s1": 64260, "s2": 257040}
    assert top.stage_rung == {"s1": 67840, "s2": 261120}
    assert top.trace_key("s2") == LTX_BUCKET_LADDER[-1]
    assert top.audio_n_real == 500 and top.audio_n == LTX_AUDIO_N_BUCKET

    bottom = route_ltx_config("720p-landscape", 24, 6)
    assert bottom.num_frames == 145
    assert bottom.stage_video_n_real == {"s1": 4180, "s2": 16720}
    assert bottom.stage_rung == {"s1": 8704, "s2": 17408}
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
        (dict(num_frames=144, height=704, width=1280, fps=24, sp_factor=SP), "8k\\+1"),
        (dict(num_frames=145, height=704, width=1280, fps=30, sp_factor=SP), "fps 30 is not served"),
        (dict(num_frames=145, height=704, width=1280, fps=24, sp_factor=4), "SP=8"),
        (dict(num_frames=145, height=704, width=1280, fps=24, sp_factor=SP, mode="video"), "AV mode only"),
        (dict(num_frames=145, height=704, width=1280, fps=24, sp_factor=SP, image_conditioned=True), "T2V only"),
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
