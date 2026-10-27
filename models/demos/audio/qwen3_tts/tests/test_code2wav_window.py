# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""The Code2Wav stage's decode window, from the connector settings vLLM-Omni chunks by.

Host only. Every chunk the settings allow has to fit one decode, or the stage refuses it at
run time; the window is sized at startup instead.

Run:
    pytest -svv models/demos/audio/qwen3_tts/tests/test_code2wav_window.py
"""

import pytest

pytest.importorskip("vllm_omni")

from models.demos.audio.qwen3_tts.tt.code2wav_vllm import stream_window
from models.demos.audio.qwen3_tts.tt.ttnn_qwen3_codec import STREAM_WINDOW


def test_the_deployed_settings_fit_the_default_window():
    extra = {"codec_chunk_frames": 25, "codec_left_context_frames": 72, "codec_chunk_ramp": [2, 3, 25]}
    assert stream_window({"extra": extra}) == STREAM_WINDOW


def test_a_longer_reference_prefix_widens_the_window_to_a_bucket():
    extra = {"codec_chunk_frames": 25, "codec_left_context_frames": 72, "ref_code_context_frames": 125}
    assert stream_window({"extra": extra}) == 160  # 125 + 25, up to a multiple of 32


def test_the_largest_ramp_entry_counts_even_above_the_steady_size():
    extra = {"codec_chunk_frames": 25, "codec_left_context_frames": 100, "codec_chunk_ramp": [2, 40, 25]}
    assert stream_window({"extra": extra}) == 160  # 100 + 40


def test_a_ramp_written_as_a_string_counts_too():
    extra = {"codec_chunk_frames": 25, "codec_left_context_frames": 100, "codec_chunk_ramp": "2,40,25"}
    assert stream_window({"extra": extra}) == 160


def test_an_unbounded_reference_prefix_is_refused(expect_error):
    with expect_error(ValueError, "no bounded length"):
        stream_window({"extra": {"codec_left_context_frames": 0}})


def test_no_connector_settings_use_vllm_omnis_defaults():
    assert stream_window(None) == STREAM_WINDOW  # 25 + 25
