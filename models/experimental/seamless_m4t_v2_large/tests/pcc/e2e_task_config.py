# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared E2E task names, languages, and reference paths (no script/helper imports)."""

from __future__ import annotations

from pathlib import Path

import pytest

TEXT_OUTPUT_TASKS = ("t2tt", "s2tt", "asr")
TASK_TGT_LANG = {
    "t2tt": "hin",
    "t2st": "hin",
    "s2tt": "eng",
    "s2st": "spa",
    "asr": "eng",
}
TEXT_INPUT_TASKS = frozenset({"t2tt", "t2st"})
SPEECH_INPUT_TASKS = frozenset({"s2tt", "s2st", "asr"})

# Mel frames at or below this are too short for stable speech-input E2E gates (all meshes).
# ~50 Hz mel → 32 frames ≈ 0.64 s, 64 ≈ 1.3 s. HF often EOS after 3–9 tokens; WER refs can be
# a single word. See README "Short speech inputs (mel ≤ 64)".
SHORT_SPEECH_E2E_MAX_MEL = 64

DEFAULT_MAX_DECODE_STEPS = 128


def maybe_skip_short_speech_input(task: str, seq_len: int) -> None:
    """Skip speech-input E2E points with too few mel frames for a meaningful gate.

    Applies to S2TT / S2ST / ASR on every mesh (P150 1×1 and BH-QB 1×4). Text-input tasks
    (T2TT / T2ST) are unaffected.
    """
    if task not in SPEECH_INPUT_TASKS or seq_len > SHORT_SPEECH_E2E_MAX_MEL:
        return
    pytest.skip(
        f"{task.upper()} E2E sweep len={seq_len}: speech input has too few mel frames "
        f"(≤{SHORT_SPEECH_E2E_MAX_MEL}); HF hits EOS early and WER/token metrics are unstable "
        f"— skipped on all meshes"
    )


_REF_DIR = Path(__file__).resolve().parent.parent / "reference_outputs"
_REFPT_NAMES = {
    "t2tt": "seamless_m4t_v2_t2tt_eng_hin.refpt",
    "t2st": "seamless_m4t_v2_t2st.refpt",
    "s2tt": "seamless_m4t_v2_s2tt.refpt",
    "s2st": "seamless_m4t_v2_s2st.refpt",
    "asr": "seamless_m4t_v2_asr.refpt",
}
_SWEEP_REF_DIR = Path(__file__).resolve().parent.parent / "teacher_forced_sweep_outputs" / "references"


def default_refpt_path(task: str) -> Path:
    return _REF_DIR / _REFPT_NAMES[task]


def sweep_refpt_path(task: str, seq_len: int, max_decode_steps: int = DEFAULT_MAX_DECODE_STEPS) -> Path:
    return _SWEEP_REF_DIR / f"seamless_m4t_v2_{task}_len{seq_len}_eval{max_decode_steps}.refpt"
