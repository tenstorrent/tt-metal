from __future__ import annotations

from pathlib import Path

import pytest

from models.demos.audio.cosy_voice.demo.common import load_cases
from models.demos.audio.cosy_voice.demo.validate_tt import validate_performance_metrics
from models.demos.audio.cosy_voice.tt.vocoder import audio_seconds

DEMO_ROOT = Path(__file__).resolve().parents[1] / "demo"


def test_performance_manifest_contains_single_public_case():
    cases = load_cases(DEMO_ROOT / "performance_cases.json")
    assert len(cases) == 1
    assert cases[0].mode == "zero_shot"
    assert cases[0].target_tokens_per_second == 30.0
    assert cases[0].target_rtf == 0.5


def test_validate_performance_metrics_accepts_public_gate():
    validate_performance_metrics(30.0, 0.49)


@pytest.mark.parametrize(
    ("tokens_per_second", "rtf"),
    [
        (29.99, 0.49),
        (30.0, 0.5),
    ],
)
def test_validate_performance_metrics_rejects_gate_failures(tokens_per_second, rtf):
    with pytest.raises(AssertionError):
        validate_performance_metrics(tokens_per_second, rtf)


def test_audio_seconds_helper():
    assert audio_seconds(22050, 22050) == 1.0
