#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Parser checks for tests/run_ttsim_regression.sh --list-ignore."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tt_metal/tt-llk/tests/run_ttsim_regression.sh"
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


def test_trailing_comment_and_all_section(tmp_path):
    ignore_file = tmp_path / "ignore.txt"
    ignore_file.write_text(
        "\n".join(
            [
                "# header",
                "",
                "[all]  # every arch",
                "common/test_shared.py::test_shared  # reason",
                "[quasar]",
                "quasar/test_other.py::test_other   # some reason",
                "[blackhole]",
                "bh_only.py::test_bh",
                "[wormhole] # wh",
            ]
        )
        + "\n"
    )

    quasar = _run("-a", "quasar", ignore_file=ignore_file)
    blackhole = _run("-a", "blackhole", ignore_file=ignore_file)

    assert quasar.returncode == 0, quasar.stderr
    assert quasar.stdout.splitlines() == [
        "common/test_shared.py::test_shared",
        "quasar/test_other.py::test_other",
    ]
    assert blackhole.returncode == 0, blackhole.stderr
    assert blackhole.stdout.splitlines() == [
        "common/test_shared.py::test_shared",
        "bh_only.py::test_bh",
    ]


@pytest.mark.parametrize(
    "body, message",
    [
        ("quasar/foo.py::test_x\n", "ignore entry before any [section]"),
        ("[riscv]\nfoo.py::test_x\n", "unknown section [riscv]"),
    ],
)
def test_loader_rejects_bad_lists(tmp_path, body, message):
    ignore_file = tmp_path / "ignore.txt"
    ignore_file.write_text(body)
    result = _run("-a", "quasar", ignore_file=ignore_file)
    assert result.returncode != 0
    assert message in result.stderr


def test_missing_ignore_file(tmp_path):
    result = _run("-a", "quasar", ignore_file=tmp_path / "missing.txt")
    assert result.returncode != 0
    assert "ignore list not found" in result.stderr


def test_ignore_file_requires_a_path():
    result = subprocess.run(["bash", str(SCRIPT), "--ignore-file"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "requires a path" in result.stderr

    result = subprocess.run(
        ["bash", str(SCRIPT), "--ignore-file", "--no-ignore"], capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "requires a path" in result.stderr
