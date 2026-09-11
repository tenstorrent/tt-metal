# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from models.tt_dit.models.transformers.ltx.rope_ltx import pad_video_rope_sp
from models.tt_dit.pipelines.ltx.pipeline_ltx import LTXTransformerState
from models.tt_dit.utils.ltx import (
    LTX_FAST_CANVASES,
    LTX_FAST_LOWEST_BUCKET_VIDEO_N,
    LTX_FAST_SP_FACTOR,
    ltx_fast_aligned_num_frames,
    ltx_fast_lowest_bucket_configs,
    ltx_fast_lowest_bucket_n,
    ltx_fast_nominal_configs,
    route_ltx_fast_lowest_bucket,
)


def _route(canvas: str, fps: int, duration: int):
    height, width = LTX_FAST_CANVASES[canvas]
    return route_ltx_fast_lowest_bucket(
        num_frames=ltx_fast_aligned_num_frames(fps, duration),
        height=height,
        width=width,
        fps=fps,
        sp_factor=LTX_FAST_SP_FACTOR,
    )


def test_lowest_bucket_routes_exactly_20_of_256_product_configs():
    nominal = ltx_fast_nominal_configs()
    supported = set(ltx_fast_lowest_bucket_configs())
    routed = set()

    assert len(nominal) == 256
    assert len(supported) == 20
    for config in nominal:
        try:
            _route(*config)
        except ValueError:
            continue
        routed.add(config)

    assert routed == supported


@pytest.mark.parametrize("canvas", ["720p-landscape", "720p-portrait"])
def test_lowest_bucket_sequence_formula_and_trace_keys(canvas):
    expected_by_timing = {
        (24, 6): (4180, 16720),
        (24, 8): (5500, 22000),
        (24, 10): (6820, 27280),
        (24, 12): (8140, 32560),
        (25, 6): (4400, 17600),
        (25, 8): (5720, 22880),
        (25, 10): (7260, 29040),
        (25, 12): (8580, 34320),
        (48, 6): (8140, 32560),
        (50, 6): (8580, 34320),
    }
    for timing, expected_real in expected_by_timing.items():
        route = _route(canvas, *timing)
        assert (route.stage_video_n_real["s1"], route.stage_video_n_real["s2"]) == expected_real
        assert route.stage_video_n == LTX_FAST_LOWEST_BUCKET_VIDEO_N
        assert route.trace_key("s1") == ("s1", "lowest")
        assert route.trace_key("s2") == ("s2", "lowest")


@pytest.mark.parametrize("stage,n_bucket", LTX_FAST_LOWEST_BUCKET_VIDEO_N.items())
def test_lowest_bucket_boundary_is_inclusive(stage, n_bucket, expect_error):
    assert ltx_fast_lowest_bucket_n(stage, n_bucket) == n_bucket
    with expect_error(ValueError, "exceeds"):
        ltx_fast_lowest_bucket_n(stage, n_bucket + 1)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"sp_factor": 4}, "SP=8"),
        ({"mode": "video"}, "AV mode"),
        ({"image_conditioned": True}, "T2V only"),
    ],
)
def test_lowest_bucket_rejects_other_structural_classes(kwargs, match, expect_error):
    height, width = LTX_FAST_CANVASES["720p-landscape"]
    request = {
        "num_frames": ltx_fast_aligned_num_frames(24, 6),
        "height": height,
        "width": width,
        "fps": 24,
        "sp_factor": 8,
    }
    request.update(kwargs)
    with expect_error(ValueError, match):
        route_ltx_fast_lowest_bucket(**request)


def test_video_rope_padding_uses_explicit_bucket_shape():
    cos = torch.randn(1, 2, 257, 4)
    sin = torch.randn_like(cos)
    padded_cos, padded_sin = pad_video_rope_sp(cos, sin, sp_factor=8, video_N=8704)

    assert padded_cos.shape == (1, 2, 8704, 4)
    assert padded_sin.shape == (1, 2, 8704, 4)
    torch.testing.assert_close(padded_cos[:, :, :257], cos)
    torch.testing.assert_close(padded_sin[:, :, :257], sin)
    assert torch.all(padded_cos[:, :, 257:] == 1)
    assert torch.all(padded_sin[:, :, 257:] == 0)


def test_video_rope_explicit_bucket_validates_shape(expect_error):
    cos = torch.zeros(1, 2, 257, 4)
    sin = torch.zeros_like(cos)
    with expect_error(ValueError, "smaller"):
        pad_video_rope_sp(cos, sin, sp_factor=8, video_N=256)
    with expect_error(ValueError, "divisible"):
        pad_video_rope_sp(cos, sin, sp_factor=8, video_N=8705)


def test_transformer_state_attribute_returns_state_tensor_value():
    state = LTXTransformerState()
    sentinel = object()
    state._tt_video_lat._data = sentinel
    assert state.tt_video_lat is sentinel
