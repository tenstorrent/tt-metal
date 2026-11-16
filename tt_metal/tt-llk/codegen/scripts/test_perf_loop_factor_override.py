# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK_PERF_LOOP_FACTOR shortens a perf run without changing its test id.

A Quasar emulator wave dump holds about 3.4 ms of design time (a fixed index of
3963 capture intervals per FPGA), so perf-capture runs slow kernels with a smaller
loop factor. The override lives in the perf harness (helpers/perf/core.py), so only
perf runs see it: the kernel and the perf CSV's loop_factor column both get the value
used, and functional tests keep their LOOP_FACTOR.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PY_TESTS = Path(__file__).resolve().parents[2] / "tests" / "python_tests"

_CODE = """
from helpers.perf.core import _loop_factor_override
from helpers.test_variant_parameters import LOOP_FACTOR, TILE_COUNT
params = _loop_factor_override([TILE_COUNT(4), LOOP_FACTOR({passed})])
assert params[0] == TILE_COUNT(4)
print(params[1].loop_factor, LOOP_FACTOR({passed}).loop_factor)
"""


def _run(passed: int, **env: str) -> subprocess.CompletedProcess:
    full = {k: v for k, v in os.environ.items() if k != "LLK_PERF_LOOP_FACTOR"}
    full.setdefault(
        "CHIP_ARCH", "quasar"
    )  # import helpers without probing for a device
    full.update(env)
    done = subprocess.run(
        [sys.executable, "-c", _CODE.format(passed=passed)],
        cwd=PY_TESTS,
        env=full,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 and "ModuleNotFoundError" in done.stdout + done.stderr:
        pytest.skip("tt-llk test helpers not importable with this interpreter")
    return done


def _loop_factors(passed: int, **env: str) -> list[str]:
    done = _run(passed, **env)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout.split()


def test_default_keeps_the_passed_loop_factor():
    assert _loop_factors(32) == ["32", "32"]


def test_env_overrides_the_perf_loop_factor_only():
    # the harness gets 4; a LOOP_FACTOR built outside it (functional tests) keeps 32
    assert _loop_factors(32, LLK_PERF_LOOP_FACTOR="4") == ["4", "32"]


def test_empty_env_is_ignored():
    assert _loop_factors(32, LLK_PERF_LOOP_FACTOR="") == ["32", "32"]


@pytest.mark.parametrize("bad", ["four", "0", "-2"])
def test_a_bad_value_is_a_clear_error(bad):
    done = _run(32, LLK_PERF_LOOP_FACTOR=bad)
    assert done.returncode != 0
    # this venv prints tracebacks to stdout
    assert (
        "LLK_PERF_LOOP_FACTOR must be a positive integer" in done.stdout + done.stderr
    )
