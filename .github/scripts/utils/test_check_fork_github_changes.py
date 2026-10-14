#!/usr/bin/env python3
"""Tests for check_fork_github_changes.py -- fork PRs must not change .github/."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_fork_github_changes as check  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "check_fork_github_changes.py"

# A git hook exports GIT_DIR/GIT_INDEX_FILE; without this the fixture git calls hit the outer repo.
CLEAN_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
CLEAN_ENV.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=CLEAN_ENV
    ).stdout.strip()


def _commit(repo: Path, files: dict[str, str | None], msg: str) -> None:
    for name, content in files.items():
        path = repo / name
        if content is None:
            _git(repo, "rm", "-q", name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", msg)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repo with main and a PR branch 'pr' cut from it; HEAD is on 'pr'."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _commit(r, {".github/workflows/ci.yaml": "on: push\n", "src/a.py": "a = 1\n"}, "base")
    _git(r, "checkout", "-q", "-b", "pr")
    return r


def _event(tmp_path: Path, fork: bool | None, base_sha: str | None = None, head_sha: str | None = None) -> Path:
    if fork is None:
        payload: dict = {"ref": "refs/heads/main"}
    else:
        payload = {
            "pull_request": {
                "head": {"repo": {"fork": fork}, "sha": head_sha},
                "base": {"sha": base_sha, "ref": "main", "repo": {"full_name": "tenstorrent/tt-metal"}},
            }
        }
    path = tmp_path / "event.json"
    path.write_text(json.dumps(payload))
    return path


def _run(repo: Path, event: Path | None, extra_env: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(CLEAN_ENV)
    for key in ("GITHUB_EVENT_PATH", "GITHUB_ACTIONS", "PRE_COMMIT_FROM_REF", "PRE_COMMIT_TO_REF"):
        env.pop(key, None)
    if event is not None:
        env["GITHUB_EVENT_PATH"] = str(event)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=repo, env=env, capture_output=True, text=True)


def _refs() -> dict:
    return {"PRE_COMMIT_FROM_REF": "main", "PRE_COMMIT_TO_REF": "HEAD"}


@pytest.mark.parametrize(
    "files",
    [
        {".github/workflows/ci.yaml": "on: pull_request\n"},
        {".github/time_budget.yaml": "x: 1\n"},
        {".github/workflows/ci.yaml": None},
    ],
    ids=["modify", "add", "delete"],
)
def test_fork_pr_with_github_change_fails(repo: Path, tmp_path: Path, files: dict) -> None:
    _commit(repo, files, "change .github")
    result = _run(repo, _event(tmp_path, fork=True), _refs())
    assert result.returncode == 1
    for name in files:
        assert f"  {name}" in result.stdout
    assert "git fetch https://github.com/tenstorrent/tt-metal.git main" in result.stdout
    assert '--source="$(git merge-base FETCH_HEAD HEAD)"' in result.stdout


def test_fork_pr_without_github_change_passes(repo: Path, tmp_path: Path) -> None:
    _commit(repo, {"src/a.py": "a = 2\n"}, "code only")
    assert _run(repo, _event(tmp_path, fork=True), _refs()).returncode == 0


def test_internal_pr_with_github_change_passes(repo: Path, tmp_path: Path) -> None:
    _commit(repo, {".github/workflows/ci.yaml": "on: pull_request\n"}, "change .github")
    assert _run(repo, _event(tmp_path, fork=False), _refs()).returncode == 0


def test_no_event_passes(repo: Path) -> None:
    _commit(repo, {".github/workflows/ci.yaml": "on: pull_request\n"}, "change .github")
    assert _run(repo, None, _refs()).returncode == 0


def test_non_pr_event_passes(repo: Path, tmp_path: Path) -> None:
    _commit(repo, {".github/workflows/ci.yaml": "on: pull_request\n"}, "change .github")
    assert _run(repo, _event(tmp_path, fork=None), _refs()).returncode == 0


def test_change_only_on_main_is_ignored(repo: Path, tmp_path: Path) -> None:
    # Three-dot diff: a .github change that landed on main after the branch point is not the PR's.
    _commit(repo, {"src/a.py": "a = 2\n"}, "code only")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, {".github/workflows/ci.yaml": "on: workflow_dispatch\n"}, "main moves")
    _git(repo, "checkout", "-q", "pr")
    assert _run(repo, _event(tmp_path, fork=True), _refs()).returncode == 0


def test_diffs_fork_head_not_merge_commit(repo: Path, tmp_path: Path) -> None:
    # main lands the same .github change; the merge commit then shows no .github diff, but the
    # runner hook diffs the fork head and still refuses the job, so this check must fail too.
    change = {".github/workflows/ci.yaml": "on: pull_request\n"}
    _commit(repo, change, "change .github")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "main")
    _commit(repo, change, "maintainer lands it")
    _git(repo, "checkout", "-q", "--detach", "main")
    _git(repo, "merge", "-q", "--no-ff", "--no-edit", head)  # GitHub's refs/pull/N/merge
    assert _git(repo, "diff", "--name-only", "main...HEAD", "--", ".github") == ""
    result = _run(repo, _event(tmp_path, fork=True, head_sha=head), _refs())
    assert result.returncode == 1
    assert ".github/workflows/ci.yaml" in result.stdout


def test_falls_back_to_event_base_sha(repo: Path, tmp_path: Path) -> None:
    base = _git(repo, "rev-parse", "main")
    _commit(repo, {".github/workflows/ci.yaml": "on: pull_request\n"}, "change .github")
    result = _run(repo, _event(tmp_path, fork=True, base_sha=base))
    assert result.returncode == 1
    assert ".github/workflows/ci.yaml" in result.stdout


def test_git_failure_fails_open(repo: Path, tmp_path: Path) -> None:
    _commit(repo, {".github/workflows/ci.yaml": "on: pull_request\n"}, "change .github")
    result = _run(repo, _event(tmp_path, fork=True, base_sha="0" * 40))
    assert result.returncode == 0
    assert "git diff failed, skipping" in result.stderr


def test_annotation_only_on_github_actions(repo: Path, tmp_path: Path) -> None:
    _commit(repo, {".github/workflows/ci.yaml": "on: pull_request\n"}, "change .github")
    event = _event(tmp_path, fork=True)
    assert "::error::" not in _run(repo, event, _refs()).stdout
    assert "::error::" in _run(repo, event, {**_refs(), "GITHUB_ACTIONS": "true"}).stdout


def test_restore_command_removes_all_github_changes(repo: Path, tmp_path: Path) -> None:
    # The restore the message prints must undo adds, edits and deletes alike (FETCH_HEAD = main here).
    _commit(
        repo,
        {".github/workflows/ci.yaml": None, ".github/new.yaml": "x: 1\n", "src/a.py": "a = 2\n"},
        "mixed change",
    )
    _git(repo, "fetch", "-q", ".", "main")
    subprocess.run(
        'git restore --staged --worktree --source="$(git merge-base FETCH_HEAD HEAD)" -- .github',
        shell=True,
        cwd=repo,
        check=True,
        env=CLEAN_ENV,
    )
    _git(repo, "commit", "-q", "-m", "Remove .github changes")
    assert _run(repo, _event(tmp_path, fork=True), _refs()).returncode == 0
    assert _git(repo, "diff", "--name-only", "main...HEAD") == "src/a.py"


@pytest.mark.parametrize(
    "payload",
    [{}, {"pull_request": None}, {"pull_request": {"head": {"repo": None}}}, []],
    ids=["empty", "null-pr", "deleted-head-repo", "not-a-dict"],
)
def test_malformed_events_are_not_forks(payload, tmp_path: Path) -> None:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(payload))
    assert check.is_fork_pull_request(check.load_event(str(path))) is False


def test_unreadable_event_is_empty(tmp_path: Path) -> None:
    assert check.load_event(str(tmp_path / "missing.json")) == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert check.load_event(str(bad)) == {}
