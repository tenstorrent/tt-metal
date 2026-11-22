# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""F191: the run checks at launch that a gate's commit can go through (DeepSeek-V4.1 8x4 CK.1: a passed gate's commit
was refused for a missing git identity, found only after the gate)."""

import subprocess

from models.demos.common.bringup.orchestrator import commit_preflight


def _repo(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))  # no global identity
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for v in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL", "EMAIL"):
        monkeypatch.delenv(v, raising=False)
    repo = tmp_path / "r"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.useConfigOnly", "true"], cwd=repo, check=True)
    return repo


def test_a_repo_without_a_git_identity_fails_the_preflight(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    bad = commit_preflight(repo)
    assert bad and all("identity" in b for b in bad)
    subprocess.run(["git", "config", "user.name", "A"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    assert commit_preflight(repo) == []


def test_a_stale_index_lock_or_a_rebase_fails_the_preflight(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    subprocess.run(["git", "config", "user.name", "A"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "a@b.c"], cwd=repo, check=True)
    (repo / ".git" / "index.lock").write_text("")
    (repo / ".git" / "rebase-merge").mkdir()
    bad = commit_preflight(repo)
    assert any("index is locked" in b for b in bad) and any("rebase" in b for b in bad)
