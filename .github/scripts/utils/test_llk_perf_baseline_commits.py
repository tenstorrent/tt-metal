#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for llk-perf-inputs-changed.sh and llk-perf-baseline-commits.sh."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent
CLASSIFY = SCRIPTS / "llk-perf-inputs-changed.sh"
COMMITS = SCRIPTS / "llk-perf-baseline-commits.sh"


def _classify(*paths):
    out = subprocess.run(
        ["bash", str(CLASSIFY)], input="".join(f"{p}\n" for p in paths), capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


@pytest.mark.parametrize(
    "path, changed",
    [
        ("tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_math_matmul.h", "true"),
        ("tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_square.h", "true"),
        ("tt_metal/tt-llk/common/inc/ckernel.h", "true"),
        ("tt_metal/tt-llk/tests/python_tests/perf_eltwise_unary_sfpu.py", "true"),
        ("tt_metal/tt-llk/tests/python_tests/helpers/perf/core.py", "true"),
        ("tt_metal/sfpi-version", "true"),
        ("tests/pipeline_reorg/llk_perf_merge_gate_tests.yaml", "true"),
        (".github/workflows/llk-perf-impl.yaml", "true"),
        (".github/scripts/llk-perf-inputs-changed.sh", "true"),
        ("tt_metal/tt-llk/tt_llk_quasar/llk_lib/llk_math_matmul.h", "false"),
        ("tt_metal/tt-llk/tests/python_tests/quasar/test_matmul_quasar.py", "false"),
        ("tt_metal/hw/ckernels/quasar/llk_api/llk_math_api.h", "false"),
        ("ttnn/cpp/ttnn/operations/eltwise/unary/unary.cpp", "false"),
        (".github/workflows/merge-gate.yaml", "false"),
    ],
)
def test_classify_one_path(path, changed):
    assert _classify(path) == changed


def test_classify_any_changed_path_counts():
    assert _classify("ttnn/a.cpp", "tt_metal/tt-llk/common/inc/b.h", "docs/c.md") == "true"


def test_classify_no_paths_is_false():
    assert _classify() == "false"


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """root, llk, other, quasar-only, llk; returns the shas oldest first."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    shas = []
    for name in (
        "README.md",
        "tt_metal/tt-llk/common/inc/a.h",
        "ttnn/b.cpp",
        "tt_metal/tt-llk/tt_llk_quasar/c.h",
        "tt_metal/hw/ckernels/wormhole_b0/d.h",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
        _git(tmp_path, "add", name)
        _git(tmp_path, "commit", "-q", "-m", name)
        shas.append(_git(tmp_path, "rev-parse", "HEAD"))
    return tmp_path, shas


def _commits(repo, *args):
    return subprocess.run(["bash", str(COMMITS), *args], cwd=repo, capture_output=True, text=True)


def test_commits_lists_llk_commits_newest_first(repo):
    path, shas = repo
    out = _commits(path, "HEAD")
    assert out.returncode == 0
    assert out.stdout.split() == [shas[4], shas[1]]


def test_commits_start_at_the_merge_base_with_the_ref(repo):
    path, shas = repo
    _git(path, "checkout", "-q", "-b", "entry", shas[2])
    assert _commits(path, "main").stdout.split() == [shas[1]]


def test_commits_stop_after_max_commits(repo):
    path, shas = repo
    assert _commits(path, "HEAD", "2").stdout.split() == [shas[4]]


def test_commits_fail_when_no_llk_commit_is_in_the_window(repo):
    path, shas = repo
    _git(path, "checkout", "-q", shas[0])
    out = _commits(path, "HEAD")
    assert out.returncode == 1 and out.stdout == ""
    assert "No LLK perf change" in out.stderr
