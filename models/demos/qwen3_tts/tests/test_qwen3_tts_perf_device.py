# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""End-to-end performance gate for qwen3_tts on Wormhole.

Runs the full TTS demo (warmup + trace capture + inference) once per module and
asserts two timings from the inference pass:

  prefill_ms          (Talker prefill: ICL -> first decode token)  upper bound only
  steady_ms_per_frame (mean AR step time, excluding the first decode step)  two-sided

Warmup and trace-capture cost are excluded — only the inference numbers are checked.

The steady frame is goldened PER SKU, because N150 and N300 are not the same speed
and one band cannot hold both: N300 runs the frame faster (its matmuls are half the
size) but noisier (per-layer CCLs), while N150 is slower and almost perfectly
repeatable. See STEADY_GOLDEN_MS. The SKU test that does not match MESH_DEVICE
skips, so exactly one steady gate runs per invocation.

Run:
    MESH_DEVICE=N300 pytest -s -v models/demos/qwen3_tts/tests/test_qwen3_tts_perf_device.py
    MESH_DEVICE=N150 pytest -s -v models/demos/qwen3_tts/tests/test_qwen3_tts_perf_device.py
"""
import os
from pathlib import Path

import pytest

# Prefill ms varies a lot run-to-run with trace-cache state and bucket warmth, so we
# only enforce an upper bound — catches regressions, ignores favorable variance. One
# bound covers both SKUs (N150 prefills slightly faster than N300).
PREFILL_MS_UPPER_BOUND = 22.0

# Which SKU this run measures. MESH_DEVICE is the same knob the demo uses to pick its
# mesh shape (N150=(1,1), N300=(1,2), T3K=(1,8)); unset selects the legacy single-chip
# ttnn.open_device path, which takes the same is_n150() fast paths as a 1x1 mesh and is
# therefore goldened as N150.
SKU = os.environ.get("MESH_DEVICE") or "N150"

# Steady AR-step ms per SKU: (expected ms/frame, fractional margin). The band is
# BIDIRECTIONAL — a faster model breaks it too, so re-golden rather than widen.
#
# One shared golden used to serve both SKUs and could not: the two are ~5 % apart and
# N300 alone swings more than that. They differ in kind, not just degree —
#
#   N150  slower per frame, almost perfectly repeatable (single chip, no CCLs in the
#         layer), so it takes a tight margin.
#   N300  faster per frame (each chip's matmuls are half the size) but noisy, because
#         every layer carries collectives whose small-payload timings swing; hence the
#         looser margin. Take medians, not single runs, when re-goldening this one.
STEADY_GOLDEN_MS = {
    "N150": (39.8, 0.03),
    "N300": (37.9, 0.05),
}

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
    )
    assert isinstance(result, dict), f"run_full_ttnn_tts must return a dict; got {type(result)}"
    return result


def test_prefill_ms(demo_run):
    measured = demo_run["prefill_ms"]
    print(f"[prefill_ms] measured = {measured:.2f} ms  upper bound = {PREFILL_MS_UPPER_BOUND:.2f} ms")
    assert (
        measured < PREFILL_MS_UPPER_BOUND
    ), f"[prefill_ms] {measured:.2f} ms >= {PREFILL_MS_UPPER_BOUND:.2f} ms upper bound"


def _assert_steady_ms(demo_run, sku: str) -> None:
    """Two-sided check of the steady AR frame against that SKU's golden."""
    expected, margin = STEADY_GOLDEN_MS[sku]
    measured = demo_run["steady_ms_per_frame"]
    lower = expected * (1 - margin)
    upper = expected * (1 + margin)
    print(
        f"[steady_ms_per_frame:{sku}] measured = {measured:.2f} ms  expected = {expected:.2f} ms  "
        f"bounds = [{lower:.2f}, {upper:.2f}]"
    )
    assert lower <= measured <= upper, (
        f"[steady_ms_per_frame:{sku}] {measured:.2f} ms outside [{lower:.2f}, {upper:.2f}] "
        f"(expected {expected:.2f}, margin {margin:.0%}). The band is bidirectional: if the "
        f'model got FASTER, re-golden STEADY_GOLDEN_MS["{sku}"] instead of widening the margin.'
    )


@pytest.mark.skipif(SKU != "N150", reason=f"N150 steady golden; this run is MESH_DEVICE={SKU}")
def test_steady_ms_per_frame_n150(demo_run):
    _assert_steady_ms(demo_run, "N150")


@pytest.mark.skipif(SKU != "N300", reason=f"N300 steady golden; this run is MESH_DEVICE={SKU}")
def test_steady_ms_per_frame_n300(demo_run):
    _assert_steady_ms(demo_run, "N300")
