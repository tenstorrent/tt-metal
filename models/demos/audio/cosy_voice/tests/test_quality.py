from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.demos.audio.cosy_voice.demo.common import load_cases
from models.demos.audio.cosy_voice.demo.validate_tt import validate_quality_metrics
from models.demos.audio.cosy_voice.tt.reference import (
    cosine_similarity_percent,
    normalize_transcript_text,
    transcript_tokens,
    word_error_rate_percent,
)

DEMO_ROOT = Path(__file__).resolve().parents[1] / "demo"


def test_quality_manifest_contains_public_mode_coverage():
    cases = load_cases(DEMO_ROOT / "quality_cases.json")
    assert {case.mode for case in cases} == {"sft", "zero_shot", "cross_lingual", "instruct"}


@pytest.mark.parametrize(
    ("text", "language", "expected"),
    [
        ("<|en|>Hello, World!", "en", "hello world"),
        ("<|yue|>今日我哋会用广东话试下呢个语音模型。", "yue", "今日我哋会用广东话试下呢个语音模型"),
    ],
)
def test_normalize_transcript_text(text, language, expected):
    assert normalize_transcript_text(text, language) == expected


def test_transcript_tokens_use_character_granularity_for_cjk():
    assert transcript_tokens("你好 世界", "zh") == list("你好世界")


def test_word_error_rate_percent():
    assert word_error_rate_percent("hello world", "hello brave world", "en") == 50.0


def test_cosine_similarity_percent():
    similarity = cosine_similarity_percent(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))
    assert similarity == pytest.approx(100.0)


def test_validate_quality_metrics_accepts_public_gate():
    validate_quality_metrics(2.5, 75.0)


@pytest.mark.parametrize(
    ("wer_pct", "speaker_similarity_pct"),
    [
        (3.1, 75.0),
        (2.5, 59.9),
    ],
)
def test_validate_quality_metrics_rejects_gate_failures(wer_pct, speaker_similarity_pct):
    with pytest.raises(AssertionError):
        validate_quality_metrics(wer_pct, speaker_similarity_pct)
