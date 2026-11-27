#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Routing tests for find-changed-files.sh, run against a scratch repository.

Run: python3 -m pytest --noconftest .github/scripts/utils/test_find_changed_files.py -q
"""

import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent / "find-changed-files.sh"
HELPERS = "tt_metal/tt-llk/tests/python_tests/helpers"
WH_TABLE = f"{HELPERS}/sfpu_accuracy_budget_wh.yaml"
BH_TABLE = f"{HELPERS}/sfpu_accuracy_budget_bh.yaml"
# The flags that start llk-smoke-tests in pr-gate.yaml, which carries the SFPU ULP PR check.
LLK_SMOKE_FLAGS = (
    "llk-wormhole-changed",
    "llk-blackhole-changed",
    "llk-common-changed",
    "llk-sfpi-changed",
    "llk-tests-changed",
    "llk-ci-changed",
    "llk-perf-changed",
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository whose origin/main holds both budget tables and one unrelated file."""
    for name in (WH_TABLE, BH_TABLE, f"{HELPERS}/other.yaml", "notes/unrelated.txt"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("content\n")
    # The script lists the submodule paths.
    (tmp_path / ".gitmodules").write_text(
        '[submodule "sub"]\n\tpath = third_party/sub\n\turl = https://example.com/sub\n'
    )
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "--no-verify", "-m", "base")
    _git(tmp_path, "update-ref", "refs/remotes/origin/main", "HEAD")
    return tmp_path


def _flags(repo: Path) -> dict[str, bool]:
    _git(repo, "commit", "-q", "--no-verify", "-a", "-m", "change")
    out = subprocess.run(
        ["bash", str(SCRIPT)], cwd=repo, check=True, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}
    )
    return {name: value == "true" for name, value in (line.split("=", 1) for line in out.stdout.splitlines())}


def _starts_llk_smoke(flags: dict[str, bool]) -> bool:
    return any(flags[name] for name in LLK_SMOKE_FLAGS)


@pytest.mark.parametrize("deleted", [(WH_TABLE,), (BH_TABLE,), (WH_TABLE, BH_TABLE)])
def test_deleting_only_a_budget_table_starts_the_llk_smoke_job(repo, deleted):
    for name in deleted:
        (repo / name).unlink()
    flags = _flags(repo)
    # llk-tests-changed also gives the job every SKU, so both cards' checks run.
    assert flags["llk-tests-changed"]
    assert _starts_llk_smoke(flags)


def test_editing_a_budget_table_starts_the_llk_smoke_job(repo):
    (repo / WH_TABLE).write_text("edited\n")
    assert _flags(repo)["llk-tests-changed"]


@pytest.mark.parametrize("deleted", ["notes/unrelated.txt", f"{HELPERS}/other.yaml"])
def test_other_deletions_stay_unrouted(repo, deleted):
    (repo / deleted).unlink()
    flags = _flags(repo)
    assert not any(flags.values()), [name for name, on in flags.items() if on]
