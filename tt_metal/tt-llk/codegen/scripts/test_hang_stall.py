# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""run_test.sh hang threshold: the default log-stall seconds per mode."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

HELPER = Path(__file__).resolve().parent / "hang_stall.sh"


def _default_stall(mode: str, **env: str) -> int:
    clean = {
        k: v
        for k, v in os.environ.items()
        if k not in ("HANG_STALL", "ENABLE_WAVEFORM")
    }
    clean.update(env)
    out = subprocess.run(
        ["bash", "-c", 'source "$1"; llk_default_stall "$2"', "_", str(HELPER), mode],
        env=clean,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return int(out.strip())


def test_emulator_default_is_unchanged():
    assert _default_stall("simulator") == 180


def test_silicon_default_is_unchanged():
    assert _default_stall("hardware") == 300


@pytest.mark.parametrize("waves", ["1", "true"])
def test_waves_on_emulator_get_a_long_threshold(waves):
    # A waveform dump slows the emulator enough that a healthy run prints nothing for minutes.
    assert _default_stall("simulator", ENABLE_WAVEFORM=waves) == 1800


def test_waves_off_value_keeps_emulator_default():
    assert _default_stall("simulator", ENABLE_WAVEFORM="0") == 180


def test_waves_do_not_change_silicon():
    assert _default_stall("hardware", ENABLE_WAVEFORM="1") == 300


def test_hang_stall_env_wins_over_waves():
    assert _default_stall("simulator", ENABLE_WAVEFORM="1", HANG_STALL="240") == 240


def test_run_test_uses_the_helper():
    script = (
        Path(__file__).resolve().parents[2] / ".claude" / "scripts" / "run_test.sh"
    ).read_text()
    assert "hang_stall.sh" in script
    assert 'STALL="$(llk_default_stall "$MODE")"' in script
