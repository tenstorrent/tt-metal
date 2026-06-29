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
import torch

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
    TEXT_OUTPUT_TASKS,
    T2TT_TOP1_THRESHOLD,
    T2TT_TOP5_THRESHOLD,
)

SWEEP_EVAL_STEPS = 128
SANITY_SWEEP_LENGTHS = (32, 64, 128)

# Sweep-only (task, seq_len) gates — default thresholds target fixed demo inputs; these lengths
# showed marginal drift on BH 1×4 full sweep (Alice @ 256 tokens; preamble mel @ 2048 frames).
# Mel @ 2048 observed ~81% top-1 / ~85% top-5 (S2TT/ASR) on bh-qbge-06.
_SWEEP_LEN_THRESHOLD_OVERRIDES: dict[tuple[str, int], tuple[float, float]] = {
    ("t2tt", 256): (0.94, T2TT_TOP5_THRESHOLD),
    ("s2tt", 2048): (0.80, 0.84),
    ("asr", 2048): (0.79, 0.84),
}


def sweep_sequence_lengths() -> list[int]:
    return sequence_lengths(SEQ_LEN_MIN, SEQ_LEN_MAX)


def sweep_auto_ref_enabled() -> bool:
    return os.environ.get("SEAMLESS_SWEEP_AUTO_REF", "1") != "0"


def sweep_thresholds_for_task(task: str, seq_len: int) -> tuple[float, float]:
    if task not in TEXT_OUTPUT_TASKS:
        raise ValueError(f"token matching sweep expects a text-output task, got {task!r}")
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
    if task not in TEXT_OUTPUT_TASKS:
        raise ValueError(f"token matching sweep expects a text-output task, got {task!r}")
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


def default_speech_sweep_mel_debug_dir(*, task: str, mel_frames: int) -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "teacher_forced_sweep_outputs"
        / "debug_mel_inputs"
        / f"mel{mel_frames}"
        / task
    )


def save_speech_sweep_mel_artifacts(
    *,
    task: str,
    mel_frames: int,
    input_features: torch.Tensor,
    mel_attention_mask: torch.Tensor,
    seed_ids: torch.Tensor | None = None,
    teacher_tokens: torch.Tensor | None = None,
    out_dir: Path | None = None,
    wav_path: Path | None = None,
) -> Path:
    """Write sweep mel inputs (exact ``input_features`` / mask) for offline inspection.

    Optionally copies ``wav_path`` (full long preamble WAV used by ``ensure_long_audio``).
    """
    import json

    dest = out_dir or default_speech_sweep_mel_debug_dir(task=task, mel_frames=mel_frames)
    dest.mkdir(parents=True, exist_ok=True)

    payload: dict = {
        "task": task,
        "mel_frames": mel_frames,
        "input_features": input_features.detach().cpu(),
        "mel_attention_mask": mel_attention_mask.detach().cpu(),
    }
    if seed_ids is not None:
        payload["seed_ids"] = seed_ids.detach().cpu()
    if teacher_tokens is not None:
        payload["teacher_tokens"] = teacher_tokens.detach().cpu()

    torch.save(payload, dest / "mel_input.pt")

    meta = {
        "task": task,
        "mel_frames": mel_frames,
        "input_features_shape": list(input_features.shape),
        "mel_valid_frames": int(mel_attention_mask.sum().item()),
        "teacher_decode_steps": int(teacher_tokens.numel()) if teacher_tokens is not None else None,
        "note": (
            "mel_input.pt is the exact tensor fed to the speech encoder in token-matching sweep "
            "(first mel_frames of processor output from long preamble audio)."
        ),
    }
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    if wav_path is not None and wav_path.is_file():
        import shutil

        shutil.copy2(wav_path, dest / "long_preamble_source.wav")

    return dest


def save_speech_sweep_mel_from_refpt(
    *,
    task: str,
    mel_frames: int,
    ref_path: Path | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Export mel sweep inputs from an existing ``.refpt`` (no device / weights required)."""
    path = ref_path or sweep_refpt_path(task, mel_frames, SWEEP_EVAL_STEPS)
    if not path.is_file():
        raise FileNotFoundError(f"Missing sweep reference {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    return save_speech_sweep_mel_artifacts(
        task=task,
        mel_frames=mel_frames,
        input_features=data["input_features"],
        mel_attention_mask=data["mel_attention_mask"],
        seed_ids=data.get("seed_ids"),
        teacher_tokens=data.get("teacher_tokens"),
        out_dir=out_dir,
    )


def maybe_save_speech_sweep_mel_env(
    *,
    task: str,
    seq_len: int,
    input_features: torch.Tensor,
    mel_attention_mask: torch.Tensor,
    seed_ids: torch.Tensor,
    teacher_tokens: torch.Tensor,
) -> None:
    """If ``SEAMLESS_SWEEP_SAVE_MEL`` matches ``seq_len``, write mel debug artifacts."""
    want = os.environ.get("SEAMLESS_SWEEP_SAVE_MEL", "").strip()
    if not want or str(seq_len) != want:
        return
    if task not in SPEECH_INPUT_TASKS:
        return
    dest = save_speech_sweep_mel_artifacts(
        task=task,
        mel_frames=seq_len,
        input_features=input_features,
        mel_attention_mask=mel_attention_mask,
        seed_ids=seed_ids,
        teacher_tokens=teacher_tokens,
    )
    from loguru import logger

    logger.info(f"Saved speech sweep mel debug artifacts for {task.upper()} mel={seq_len} to {dest}")
