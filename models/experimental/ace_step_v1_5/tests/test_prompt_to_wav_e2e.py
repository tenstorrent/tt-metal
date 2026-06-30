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

# BH QuietBox 2 shared MLPerf volume; downloaded on first test run if absent.
_CI_CHECKPOINT_ROOT = Path("/mnt/MLPerf/huggingface/hub/ACE-Step-1.5-checkpoints")


def _running_in_ci() -> bool:
    return os.environ.get("CI", "").lower() == "true"


def _require_ci_checkpoints() -> Path:
    turbo = _CI_CHECKPOINT_ROOT / "acestep-v15-turbo"
    has_weights = (turbo / "model.safetensors").is_file() or bool(list(turbo.glob("model-*.safetensors")))
    if has_weights:
        return _CI_CHECKPOINT_ROOT

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        pytest.fail(f"huggingface_hub is required to download ACE-Step checkpoints: {exc}")

    _CI_CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    offline = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        print(f"[ace_step_v1_5] Downloading ACE-Step/Ace-Step1.5 -> {_CI_CHECKPOINT_ROOT} ...", flush=True)
        snapshot_download(
            repo_id="ACE-Step/Ace-Step1.5",
            local_dir=str(_CI_CHECKPOINT_ROOT),
            token=os.environ.get("HF_TOKEN") or None,
            resume_download=True,
            local_files_only=False,
        )
    except OSError as exc:
        pytest.fail(
            f"Failed to download ACE-Step v1.5 checkpoints to {_CI_CHECKPOINT_ROOT}: {exc}. "
            "Ensure the MLPerf volume is writable (mlperf-read-only disabled) and HF_TOKEN is set."
        )
    except Exception as exc:
        pytest.fail(f"Failed to download ACE-Step v1.5 checkpoints to {_CI_CHECKPOINT_ROOT}: {exc}")
    finally:
        if offline is not None:
            os.environ["HF_HUB_OFFLINE"] = offline

    if not (turbo / "model.safetensors").is_file() and not list(turbo.glob("model-*.safetensors")):
        pytest.fail(f"ACE-Step v1.5 checkpoints still missing at {turbo} after download.")
    return _CI_CHECKPOINT_ROOT


@pytest.mark.parametrize("duration_label,duration_sec", [("15s", 15)])
def test_prompt_to_wav_bh_demo(tmp_path, monkeypatch, duration_label: str, duration_sec: int):
    """Turbo mesh smoke: LM preprocess → DiT denoise → VAE decode → WAV."""
    if not _running_in_ci():
        pytest.skip("ACE-Step BH_QB e2e smoke runs in CI only (requires bh_quietbox_2 hardware).")

    ckpt_root = _require_ci_checkpoints()
    monkeypatch.setenv("ACE_STEP_CHECKPOINT_DIR", str(ckpt_root))

    demo_script = Path(__file__).resolve().parent.parent / "demo" / "run_prompt_to_wav.py"
    out = tmp_path / f"ace_step_bh_demo_{duration_label}.wav"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(demo_script),
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
