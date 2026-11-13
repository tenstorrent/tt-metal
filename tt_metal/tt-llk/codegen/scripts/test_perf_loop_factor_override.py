# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK_PERF_LOOP_FACTOR shortens a perf run without changing its test id.

A Quasar emulator wave dump holds about 3.4 ms of design time (a fixed index of
3963 capture intervals per FPGA), so perf-capture runs slow kernels with a smaller
loop factor. The override lives in the LOOP_FACTOR runtime parameter: the kernel and
the perf CSV's loop_factor column both see the value used.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PY_TESTS = Path(__file__).resolve().parents[2] / "tests" / "python_tests"


def _loop_factor(passed: int, **env: str) -> str:
    full = {k: v for k, v in os.environ.items() if k != "LLK_PERF_LOOP_FACTOR"}
    full.update(env)
    code = (
        "from helpers.test_variant_parameters import LOOP_FACTOR\n"
        f"p = LOOP_FACTOR({passed})\n"
        "print(p.loop_factor, p.convert_to_cpp())\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PY_TESTS,
        env=full,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 and "ModuleNotFoundError" in done.stderr:
        pytest.skip(
            f"tt-llk test helpers not importable here: {done.stderr.strip().splitlines()[-1]}"
        )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip()


def test_default_keeps_the_passed_loop_factor():
    assert _loop_factor(32) == "32 constexpr std::uint32_t LOOP_FACTOR = 32;"


def test_env_overrides_the_loop_factor():
    assert (
        _loop_factor(32, LLK_PERF_LOOP_FACTOR="4")
        == "4 constexpr std::uint32_t LOOP_FACTOR = 4;"
    )


def test_empty_env_is_ignored():
    assert _loop_factor(32, LLK_PERF_LOOP_FACTOR="").startswith("32 ")
