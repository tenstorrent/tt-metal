# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "perf"))

import sum_split_durations as summing  # noqa: E402


def _legs(root: Path, legs: dict, inner: str = ".perf_split_durations") -> Path:
    for name, durations in legs.items():
        leg = root / name
        leg.mkdir(parents=True)
        (leg / inner).write_text(json.dumps(durations))
    return root


def test_a_test_that_ran_in_two_legs_adds_up(tmp_path):
    download = _legs(
        tmp_path / "download",
        {
            "test-durations-blackhole-1": {"a.py::t": 1.0},
            "test-durations-blackhole-2": {"a.py::t": 0.5, "b.py::t": 2.0},
            "test-durations-quasar-1": {"q.py::t": 4.0},
        },
    )
    assert summing.sum_durations(download, "blackhole") == {
        "a.py::t": 1.5,
        "b.py::t": 2.0,
    }
    assert summing.sum_durations(download, "wormhole") == {}


def test_both_the_smoke_and_the_perf_file_are_read(tmp_path):
    # The smoke legs store pytest-split's .test_durations, the perf legs their own.
    download = _legs(
        tmp_path / "download", {"test-durations-wormhole-1": {"a.py::t": 1.0}}
    )
    _legs(download, {"test-durations-wormhole-2": {"b.py::t": 2.0}}, ".test_durations")
    assert summing.sum_durations(download, "wormhole") == {
        "a.py::t": 1.0,
        "b.py::t": 2.0,
    }


def test_one_table_is_written_per_architecture_that_has_durations(
    tmp_path, monkeypatch
):
    download = _legs(
        tmp_path / "download", {"test-durations-wormhole-3": {"b.py::t": 2.0}}
    )
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert summing.main([str(download), str(tmp_path / "out")]) == 0
    assert json.loads((tmp_path / "out" / "wormhole.json").read_text()) == {
        "b.py::t": 2.0
    }
    assert not (tmp_path / "out" / "blackhole.json").exists()
    assert output.read_text().strip() == "found=true"


def test_nothing_found_is_reported(tmp_path, monkeypatch):
    (tmp_path / "download").mkdir()
    output = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(output))
    assert summing.main([str(tmp_path / "download"), str(tmp_path / "out")]) == 0
    assert output.read_text().strip() == "found=false"
