# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Full prompt-to-wav smoke on BH_QB (Blackhole Demo CI e2e)."""

from __future__ import annotations

import os
import sys
import wave
from pathlib import Path

import pytest


def _checkpoint_root() -> Path:
    for key in ("ACE_STEP_CHECKPOINT_DIR", "ACESTEP_CHECKPOINTS_DIR"):
        val = os.environ.get(key)
        if val:
            return Path(val).expanduser().resolve()
    return Path.home() / ".cache" / "huggingface" / "hub" / "ACE-Step-1.5-checkpoints"


def _checkpoints_available() -> bool:
    turbo = _checkpoint_root() / "acestep-v15-turbo"
    if (turbo / "model.safetensors").is_file():
        return True
    return bool(list(turbo.glob("model-*.safetensors")))


@pytest.mark.parametrize("duration_label,duration_sec", [("15s", 15)])
def test_prompt_to_wav_bh_demo(tmp_path, monkeypatch, duration_label: str, duration_sec: int):
    """Turbo mesh smoke: LM preprocess → DiT denoise → VAE decode → WAV."""
    if not _checkpoints_available():
        pytest.skip("ACE-Step v1.5 checkpoints not found; set ACE_STEP_CHECKPOINT_DIR.")

    out = tmp_path / f"ace_step_bh_demo_{duration_label}.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_prompt_to_wav",
            "--mesh-device",
            "BH_QB",
            "--variant",
            "acestep-v15-turbo",
            "--lm_variant",
            "acestep-5Hz-lm-1.7B",
            "--duration_sec",
            str(duration_sec),
            "--infer_steps",
            "8",
            "--guidance_scale",
            "1",
            "--no-use-cot-caption",
            "--seed",
            "0",
            "--prompt",
            "Electronic dance track with deep bass",
            "--out",
            str(out),
        ],
    )

    from models.experimental.ace_step_v1_5.demo.run_prompt_to_wav import main

    main()

    assert out.is_file(), f"Expected wav at {out}"
    with wave.open(str(out), "rb") as wf:
        assert wf.getnframes() > 0, "WAV contains no audio frames"
        assert wf.getframerate() == 48000
