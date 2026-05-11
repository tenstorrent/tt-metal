# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""AR-decode perf + accuracy gate for qwen3_tts.

Runs the canonical TTS demo (jim reference, fixed text, seed=42, --use-2cq,
TT_QWEN3_CP_FP32=1) and asserts:
  1. Steady decode ms/frame < 55.0
  2. ECAPA cos(generated, reference) >= 0.97
  3. Whisper exact-match of transcribed text against the target text.

Designed as the gate for the CP fp32-layer optimization (see
docs/superpowers/specs/2026-05-11-qwen3-tts-ar-decode-opti-design.md).
"""
import os
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

TARGET_TEXT = (
    "Hello, this is a test of the Qwen3 TTS speech system running on "
    "Tenstorrent hardware. The autoregressive decoder loop is what we want "
    "to profile here today."
)
REF_TEXT = "Jason, can we take a look at the review slides"
REF_AUDIO = str(REPO_ROOT / "models" / "demos" / "qwen3_tts" / "demo" / "jim_reference.wav")
OUTPUT_WAV = "/tmp/test_ar_perf_gate.wav"

MS_PER_FRAME_MAX = 55.0
ECAPA_COS_MIN = 0.97


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@pytest.fixture(scope="module")
def gate_run():
    os.environ["TT_QWEN3_CP_FP32"] = "1"
    from models.demos.qwen3_tts.demo.demo_full_ttnn_tts import run_full_ttnn_tts

    result = run_full_ttnn_tts(
        text=TARGET_TEXT,
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
        output_path=OUTPUT_WAV,
        seed=42,
        use_2cq=True,
    )
    assert isinstance(result, dict), f"run_full_ttnn_tts must return a dict with timing info; got {type(result)}"
    assert (
        "steady_ms_per_frame" in result
    ), f"run_full_ttnn_tts result missing 'steady_ms_per_frame': keys={list(result)}"
    return result


def test_steady_ms_per_frame_under_55(gate_run):
    steady = gate_run["steady_ms_per_frame"]
    assert steady < MS_PER_FRAME_MAX, f"Steady AR decode {steady:.2f} ms/frame >= {MS_PER_FRAME_MAX} ms/frame target"


def test_ecapa_cos_at_least_0_97(gate_run):
    from models.demos.qwen3_tts.tests.audio_diff import speaker_similarity_via_reference

    sims = speaker_similarity_via_reference(REF_AUDIO, OUTPUT_WAV)
    assert sims is not None, "speaker_similarity_via_reference failed to load (ECAPA disabled)"
    assert len(sims) == 1
    cos = sims[0]
    assert cos >= ECAPA_COS_MIN, f"ECAPA cos(generated, reference) = {cos:.4f} < {ECAPA_COS_MIN} threshold"


def test_whisper_exact_match(gate_run):
    from models.demos.qwen3_tts.tests.test_ttnn_audio_quality import _transcribe

    transcript = _transcribe(OUTPUT_WAV)
    assert _norm(transcript) == _norm(
        TARGET_TEXT
    ), f"Whisper transcript mismatch:\n  got: {_norm(transcript)!r}\n  exp: {_norm(TARGET_TEXT)!r}"
