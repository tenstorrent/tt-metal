# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the Whisper temperature-fallback decision.

The decision must match openai/whisper (transcribe.py decode_with_fallback) and HF
WhisperGenerationMixin._need_fallback: retry at a higher temperature when the compression ratio is
above its threshold or the average log-probability is below its threshold, except that a segment with
no_speech_prob above its threshold and avg_logprob below the log-probability threshold is accepted as
silence (no retry) and produces no text.
"""

import pytest

from models.demos.audio.whisper.tt.whisper_generator import WhisperGenerator

check = WhisperGenerator._check_generation_quality


@pytest.mark.parametrize(
    "avg_logprob, no_speech_prob, compression_ratio, expected",
    [
        (-0.3, 0.1, 1.5, (True, "quality_good")),
        (-1.5, 0.1, 1.5, (False, "low_avg_logprob")),  # low log-prob alone triggers the fallback
        (-0.3, 0.1, 3.0, (False, "high_compression_ratio")),
        (-1.5, 0.1, 3.0, (False, "high_compression_ratio")),
        (-1.5, 0.9, 1.5, (True, "silence_detected")),  # silence is accepted, not retried
        (-1.5, 0.9, 3.0, (True, "silence_detected")),  # silence overrides a high compression ratio
        (-0.3, 0.9, 1.5, (True, "quality_good")),  # confident text is kept even with a high no-speech prob
        (-1.5, 0.5, 1.5, (False, "low_avg_logprob")),  # no-speech prob below its threshold: normal fallback
    ],
)
def test_fallback_decision_matches_openai(avg_logprob, no_speech_prob, compression_ratio, expected):
    assert check("text", avg_logprob, no_speech_prob, compression_ratio, -1.0, 2.4, 0.6) == expected


def test_disabled_thresholds_never_fall_back():
    assert check("text", -9.0, 0.99, 9.0, None, None, None) == (True, "quality_good")
    # logprob_threshold unset: neither the log-probability rule nor the silence rule applies
    assert check("text", -9.0, 0.99, 1.0, None, 2.4, 0.6) == (True, "quality_good")
    # no_speech_threshold unset: the log-probability rule still applies
    assert check("text", -9.0, 0.99, 1.0, -1.0, 2.4, None) == (False, "low_avg_logprob")


def test_blank_silent_items_returns_no_text():
    data = ["a", "b", [{"text": "c", "start": 0.0, "end": 1.0}]]
    WhisperGenerator._blank_silent_items(data, [0, 2], [False, False, True])
    assert data == ["", "b", []]
