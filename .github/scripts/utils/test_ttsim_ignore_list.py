#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parser checks for tests/run_ttsim_regression.sh reading ttsim-skip-list.yaml."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tt_metal/tt-llk/tests/run_ttsim_regression.sh"
LLK_PREFIX = "tt_metal/tt-llk/tests/python_tests/"
VECTOR_ADD = "quasar/test_vector_add_l1_quasar.py::test_vector_add_l1_quasar"


def _run(*args, ignore_file=None):
    cmd = ["bash", str(SCRIPT), *args, "--list-ignore"]
    if ignore_file is not None:
        cmd[2:2] = ["--ignore-file", str(ignore_file)]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_default_list_selects_quasar_vector_add_only():
    quasar = _run("-a", "quasar")
    blackhole = _run("-a", "blackhole")
    wormhole = _run("-a", "wormhole")

    assert quasar.returncode == 0, quasar.stderr
    assert quasar.stdout.splitlines() == [VECTOR_ADD]
    assert blackhole.returncode == 0, blackhole.stderr
    assert blackhole.stdout == ""
    assert wormhole.returncode == 0, wormhole.stderr
    assert wormhole.stdout == ""


def test_no_ignore_prints_nothing():
    result = _run("-a", "quasar", "--no-ignore")
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_llk_paths_are_stripped_and_other_suites_are_left_alone(tmp_path):
    ignore_file = tmp_path / "ttsim-skip-list.yaml"
    ignore_file.write_text(
        "\n".join(
            [
                "wormhole_b0:",
                "  - tests/ttnn/unit_tests/test_conv.py  # ttnn, not collected here",
                f"  - {LLK_PREFIX}test_wh.py::test_wh  # llk",
                "blackhole:",
                f"  - {LLK_PREFIX}test_bh.py::test_bh",
                "quasar:",
                f"  - {LLK_PREFIX}quasar/test_other.py::test_other",
                "  - tests/ttnn/unit_tests/test_quasar_ttnn.py",
            ]
        )
        + "\n"
    )

    wormhole = _run("-a", "wormhole", ignore_file=ignore_file)
    blackhole = _run("-a", "blackhole", ignore_file=ignore_file)
    quasar = _run("-a", "quasar", ignore_file=ignore_file)

    assert wormhole.returncode == 0, wormhole.stderr
    assert wormhole.stdout.splitlines() == ["test_wh.py::test_wh"]
    assert blackhole.returncode == 0, blackhole.stderr
    assert blackhole.stdout.splitlines() == ["test_bh.py::test_bh"]
    assert quasar.returncode == 0, quasar.stderr
    assert quasar.stdout.splitlines() == ["quasar/test_other.py::test_other"]


def test_loader_rejects_an_entry_before_any_key(tmp_path):
    ignore_file = tmp_path / "ttsim-skip-list.yaml"
    ignore_file.write_text(f"- {LLK_PREFIX}quasar/foo.py::test_x\n")
    result = _run("-a", "quasar", ignore_file=ignore_file)
    assert result.returncode != 0
    assert "skip entry before any architecture key" in result.stderr


def test_missing_ignore_file(tmp_path):
    result = _run("-a", "quasar", ignore_file=tmp_path / "missing.yaml")
    assert result.returncode != 0
    assert "ttsim skip list not found" in result.stderr


def test_ignore_file_requires_a_path():
    result = subprocess.run(["bash", str(SCRIPT), "--ignore-file"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "requires a path" in result.stderr

    result = subprocess.run(
        ["bash", str(SCRIPT), "--ignore-file", "--no-ignore"], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "requires a path" in result.stderr
