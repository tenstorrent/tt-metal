# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Per-block performance gate for qwen3_tts on Blackhole P150.

Runs the full TTS demo (warmup + trace capture + inference) and asserts
two timings from the inference pass within ``MARGIN`` of a hardcoded
golden value:

  prefill_ms          (Talker prefill: ICL -> first decode token)
  steady_ms_per_frame (mean AR step time, excluding the first decode step)

Warmup and trace-capture cost are excluded — only the inference numbers
are checked. Reference goldens were measured on Blackhole P150 at HEAD.

Run:
    pytest -s -v models/demos/qwen3_tts/tests/test_qwen3_tts_perf_device.py
"""
from pathlib import Path

import pytest

# Steady AR-step ms is checked against a tight bidirectional bound (3% margin).
# Prefill ms varies a lot run-to-run with trace-cache state and bucket warmth
# (observed 15-25 ms swing on otherwise-identical runs), so we only enforce an
# upper bound — catches regressions, ignores favorable variance.
EXPECTED_STEADY_MS_PER_FRAME = 43.3
STEADY_MARGIN = 0.05
PREFILL_MS_UPPER_BOUND = 20.0

REPO_ROOT = Path(__file__).resolve().parents[4]
REF_AUDIO = str(REPO_ROOT / "models" / "demos" / "qwen3_tts" / "demo" / "jim_reference.wav")
REF_TEXT = "Jason, can we take a look at the review slides"
TARGET_TEXT = (
    "Good morning. Today is a beautiful day for a walk in the park, with bright sun "
    "and a gentle breeze through the trees."
)
OUTPUT_WAV = "/tmp/qwen3_tts_perf_device.wav"


@pytest.fixture(scope="module")
def demo_run():
    """Run run_full_ttnn_tts once and return the timing dict."""
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import run_full_ttnn_tts

    result = run_full_ttnn_tts(
        text=TARGET_TEXT,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        output_path=OUTPUT_WAV,
        seed=42,
        pre_measurement_warmup=True,
    )
    assert isinstance(result, dict), f"run_full_ttnn_tts must return a dict; got {type(result)}"
    return result


def test_prefill_ms(demo_run):
    measured = demo_run["prefill_ms"]
    print(f"[prefill_ms] measured = {measured:.2f} ms  upper bound = {PREFILL_MS_UPPER_BOUND:.2f} ms")
    assert (
        measured < PREFILL_MS_UPPER_BOUND
    ), f"[prefill_ms] {measured:.2f} ms >= {PREFILL_MS_UPPER_BOUND:.2f} ms upper bound"


def test_steady_ms_per_frame(demo_run):
    measured = demo_run["steady_ms_per_frame"]
    expected = EXPECTED_STEADY_MS_PER_FRAME
    lower = expected * (1 - STEADY_MARGIN)
    upper = expected * (1 + STEADY_MARGIN)
    print(
        f"[steady_ms_per_frame] measured = {measured:.2f} ms  expected = {expected:.2f} ms  "
        f"bounds = [{lower:.2f}, {upper:.2f}]"
    )
    assert lower <= measured <= upper, (
        f"[steady_ms_per_frame] {measured:.2f} ms outside [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected:.2f}, margin {STEADY_MARGIN:.0%})"
    )
