# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for JSON-object prompt handling (`cosmos3_prompt`).

Covers the NVIDIA-faithful spec injection (`format_json_prompt_with_template`),
object detection (`parse_json_object_prompt`), aspect-ratio derivation, and the
`tokenize_prompt` wrapper contract: free text and action-mode pass through
untouched; a JSON positive is reformatted with the flat templates suppressed
while the negative keeps its inverse templates.
"""

from __future__ import annotations

import pytest

from models.tt_dit.experimental.cosmos3_i2v.pipelines.cosmos3_prompt import (
    _derive_aspect_ratio,
    format_json_prompt_with_template,
    install_json_prompt_parsing,
    parse_json_object_prompt,
)

# --- object detection -------------------------------------------------------


def test_parse_returns_dict_for_json_object():
    assert parse_json_object_prompt('{"a": 1}') == {"a": 1}


@pytest.mark.parametrize("text", ["a cat walks", "[1, 2]", "42", '"string"', "null", "{bad", ""])
def test_parse_returns_none_for_non_objects(text):
    assert parse_json_object_prompt(text) is None


# --- spec injection (NVIDIA _format_json_prompt_with_template parity) --------


def test_format_video_injects_and_overwrites():
    obj = {"subjects": ["a car"], "resolution": {"W": 100, "H": 50}, "fps": 12, "duration": "3s"}
    out = format_json_prompt_with_template(obj, fps=24, num_frames=189, aspect_ratio="16,9", height=720, width=1280)
    # Existing keys overwritten in place (specs are source of truth), aspect_ratio
    # appended; default json.dumps separators; H before W; fps float; duration int"s".
    assert out == (
        '{"subjects": ["a car"], "resolution": {"H": 720, "W": 1280}, '
        '"fps": 24.0, "duration": "7s", "aspect_ratio": "16,9"}'
    )


def test_format_image_drops_temporal_metadata():
    obj = {"subjects": ["a car"], "resolution": {"W": 100, "H": 50}, "fps": 12, "duration": "3s"}
    out = format_json_prompt_with_template(obj, fps=24, num_frames=1, aspect_ratio="1,1", height=1024, width=1024)
    assert out == '{"subjects": ["a car"], "resolution": {"H": 1024, "W": 1024}, "aspect_ratio": "1,1"}'


def test_format_omits_aspect_ratio_when_none():
    out = format_json_prompt_with_template(
        {"subjects": ["x"]}, fps=24, num_frames=1, aspect_ratio=None, height=512, width=512
    )
    assert "aspect_ratio" not in out


def test_include_temporal_metadata_defaults_to_video():
    video = format_json_prompt_with_template({"s": 1}, fps=24, num_frames=189, aspect_ratio=None, height=8, width=8)
    image = format_json_prompt_with_template({"s": 1}, fps=24, num_frames=1, aspect_ratio=None, height=8, width=8)
    assert '"duration"' in video and '"fps"' in video
    assert '"duration"' not in image and '"fps"' not in image


@pytest.mark.parametrize(
    "width, height, expected",
    [(1280, 720, "16,9"), (720, 1280, "9,16"), (1024, 1024, "1,1"), (1280, 960, "4,3"), (960, 1280, "3,4")],
)
def test_derive_aspect_ratio(width, height, expected):
    assert _derive_aspect_ratio(width, height) == expected


# --- tokenize_prompt wrapper contract ---------------------------------------

# Vendored inverse template strings (see reference pipeline_cosmos3_omni.py).
_INV_DUR = "The video is not {duration:.1f} seconds long and is not of {fps:.0f} FPS."
_INV_IMG_RES = "This image is not of {height}x{width} resolution."
_INV_VID_RES = "This video is not of {height}x{width} resolution."


class _FakePipe:
    """Minimal stand-in exposing the attrs the wrapper reads and a capturing tokenize_prompt."""

    inverse_duration_template = _INV_DUR
    inverse_image_resolution_template = _INV_IMG_RES
    inverse_video_resolution_template = _INV_VID_RES

    def __init__(self):
        self.received = None

    def tokenize_prompt(self, prompt, negative_prompt=None, **kw):
        self.received = {"prompt": prompt, "negative_prompt": negative_prompt, **kw}
        return ("cond", "uncond")


_VIDEO_KW = dict(
    num_frames=189, height=720, width=1280, fps=24, add_duration_template=True, add_resolution_template=True
)


def test_wrapper_passthrough_free_text():
    pipe = install_json_prompt_parsing(_FakePipe())
    pipe.tokenize_prompt("a cat walks", "ugly", **_VIDEO_KW)
    r = pipe.received
    # Free text and flags untouched -> reference tokenizes identically to before.
    assert r["prompt"] == "a cat walks"
    assert r["negative_prompt"] == "ugly"
    assert r["add_duration_template"] is True and r["add_resolution_template"] is True


def test_wrapper_passthrough_action_mode():
    pipe = install_json_prompt_parsing(_FakePipe())
    pipe.tokenize_prompt('{"subjects": ["x"]}', None, action_mode="some_mode", **_VIDEO_KW)
    # Action mode owns its own JSON caption downstream; the wrapper must not touch it.
    assert pipe.received["prompt"] == '{"subjects": ["x"]}'
    assert pipe.received["add_resolution_template"] is True


def test_wrapper_reformats_json_positive_and_suppresses_flat_templates():
    pipe = install_json_prompt_parsing(_FakePipe())
    pipe.tokenize_prompt('{"subjects": ["a car"]}', "ugly", **_VIDEO_KW)
    r = pipe.received
    assert r["prompt"] == (
        '{"subjects": ["a car"], "duration": "7s", "fps": 24.0, '
        '"resolution": {"H": 720, "W": 1280}, "aspect_ratio": "16,9"}'
    )
    assert r["add_duration_template"] is False and r["add_resolution_template"] is False


def test_wrapper_preserves_negative_inverse_templates_on_json_positive():
    pipe = install_json_prompt_parsing(_FakePipe())
    pipe.tokenize_prompt('{"subjects": ["a car"]}', "ugly", **_VIDEO_KW)
    # Flat flags get forced off for the JSON positive; the negative's inverse
    # templates are re-applied so it is unchanged vs the free-text path.
    expected_neg = (
        "ugly. " + _INV_DUR.format(duration=189 / 24, fps=24) + " " + _INV_VID_RES.format(height=720, width=1280)
    )
    assert pipe.received["negative_prompt"] == expected_neg


def test_wrapper_negative_none_becomes_empty_then_templated():
    pipe = install_json_prompt_parsing(_FakePipe())
    pipe.tokenize_prompt('{"subjects": ["a car"]}', None, **_VIDEO_KW)
    expected_neg = _INV_DUR.format(duration=189 / 24, fps=24) + " " + _INV_VID_RES.format(height=720, width=1280)
    assert pipe.received["negative_prompt"] == expected_neg


def test_wrapper_install_is_idempotent():
    pipe = _FakePipe()
    once = install_json_prompt_parsing(pipe).tokenize_prompt
    install_json_prompt_parsing(pipe)
    # Second install must not double-wrap.
    assert pipe.tokenize_prompt is once
