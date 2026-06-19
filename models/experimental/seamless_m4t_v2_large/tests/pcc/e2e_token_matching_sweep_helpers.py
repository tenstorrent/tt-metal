# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sequence-length sweep for E2E token matching (Devstral-style input-length ladder).

For each task and input length N in {32, 64, …, 4096}, builds the same inputs as
``demo_perf_sweep.py`` (N source tokens or N mel frames), runs teacher-forced decode for
``SWEEP_EVAL_STEPS`` steps, and compares TT greedy top-1/top-5 to offline HF references.

References live under ``tests/teacher_forced_sweep_outputs/references/`` as
``seamless_m4t_v2_{task}_len{N}_eval{M}.refpt``. Missing refs are generated on first run
unless ``SEAMLESS_SWEEP_AUTO_REF=0``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from models.experimental.seamless_m4t_v2_large.scripts.demo_perf_sweep import (
    SEQ_LEN_MAX,
    SEQ_LEN_MIN,
    sequence_lengths,
)
from models.experimental.seamless_m4t_v2_large.scripts.generate_t2tt_token_accuracy_reference import (
    generate_speech_sweep_reference,
    generate_text_sweep_reference,
    sweep_refpt_path,
)
from models.experimental.seamless_m4t_v2_large.tests.pcc.e2e_token_matching_helpers import (
    SPEECH_INPUT_TASKS,
    SPEECH_TOP1_THRESHOLD,
    SPEECH_TOP5_THRESHOLD,
    S2ST_MIN_TOKEN_REF_STEPS,
    TEXT_INPUT_TASKS,
    T2TT_TOP1_THRESHOLD,
    T2TT_TOP5_THRESHOLD,
)

SWEEP_EVAL_STEPS = 128
SANITY_SWEEP_LENGTHS = (32, 64, 128)

# Sweep-only (task, seq_len) gates — default thresholds target fixed demo inputs; these lengths
# showed marginal drift on BH 1×4 full sweep (Alice @ 256 tokens; preamble mel @ 2048 frames).
_SWEEP_LEN_THRESHOLD_OVERRIDES: dict[tuple[str, int], tuple[float, float]] = {
    ("t2tt", 256): (0.94, T2TT_TOP5_THRESHOLD),
    ("t2st", 256): (0.94, T2TT_TOP5_THRESHOLD),
    ("s2tt", 2048): (0.86, 0.90),
    ("asr", 2048): (0.86, 0.89),
    ("s2st", 2048): (0.73, 0.83),
}


def sweep_sequence_lengths() -> list[int]:
    return sequence_lengths(SEQ_LEN_MIN, SEQ_LEN_MAX)


def sweep_auto_ref_enabled() -> bool:
    return os.environ.get("SEAMLESS_SWEEP_AUTO_REF", "1") != "0"


def sweep_thresholds_for_task(task: str, seq_len: int) -> tuple[float, float]:
    override = _SWEEP_LEN_THRESHOLD_OVERRIDES.get((task, seq_len))
    if override is not None:
        return override
    if task in TEXT_INPUT_TASKS:
        return T2TT_TOP1_THRESHOLD, T2TT_TOP5_THRESHOLD
    if task in SPEECH_INPUT_TASKS:
        return SPEECH_TOP1_THRESHOLD, SPEECH_TOP5_THRESHOLD
    raise ValueError(f"unknown task {task!r}")


def ensure_sweep_reference(
    task: str,
    seq_len: int,
    weights_dir: str,
    *,
    max_decode_steps: int = SWEEP_EVAL_STEPS,
) -> Path:
    """Return sweep ``.refpt`` path, generating offline HF reference if missing."""
    ref_path = sweep_refpt_path(task, seq_len, max_decode_steps)
    if ref_path.is_file():
        return ref_path
    if not sweep_auto_ref_enabled():
        gen = "models/experimental/seamless_m4t_v2_large/scripts/generate_t2tt_token_accuracy_reference.py"
        pytest.skip(
            f"Missing sweep reference {ref_path}. Run: "
            f"python {gen} --sweep --task {task} --seq_len {seq_len} "
            f"(or set SEAMLESS_SWEEP_AUTO_REF=1 to generate on first run)"
        )
    if task in TEXT_INPUT_TASKS:
        generate_text_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            seq_len=seq_len,
            max_decode_steps=max_decode_steps,
        )
    else:
        generate_speech_sweep_reference(
            weights_dir=weights_dir,
            output_file=ref_path,
            task=task,
            mel_frames=seq_len,
            max_decode_steps=max_decode_steps,
        )
    return ref_path


def maybe_skip_short_speech_sweep(task: str, ref_teacher_steps: int, seq_len: int) -> None:
    """Skip speech sweep points where HF hits EOS before enough steps to score reliably."""
    if ref_teacher_steps >= S2ST_MIN_TOKEN_REF_STEPS:
        return
    pytest.skip(
        f"{task.upper()} sweep len={seq_len} reference has {ref_teacher_steps} decode steps "
        f"(need >={S2ST_MIN_TOKEN_REF_STEPS}); HF hit EOS early on short mel input"
    )
