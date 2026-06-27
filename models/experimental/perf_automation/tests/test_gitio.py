"""gitio helpers — real git against a throwaway temp repo (no key, no hardware)."""

import subprocess

import pytest

from agent import gitio


def _init_repo(d):
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=d, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=d, check=True)
    (d / "model.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "."], cwd=d, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=d, check=True)
    return d


def test_repo_root_walks_up_from_subdir(tmp_path):
    repo = _init_repo(tmp_path)
    sub = tmp_path / "models" / "demos"
    sub.mkdir(parents=True)
    assert gitio.repo_root(sub).samefile(repo)


def test_head_sha_is_40_hex(tmp_path):
    repo = _init_repo(tmp_path)
    sha = gitio.head_sha(repo)
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_is_clean_tracks_edits(tmp_path):
    repo = _init_repo(tmp_path)
    assert gitio.is_clean(repo) is True
    (repo / "model.py").write_text("x = 2\n")
    assert gitio.is_clean(repo) is False


def test_reset_hard_restores(tmp_path):
    repo = _init_repo(tmp_path)
    sha = gitio.head_sha(repo)
    (repo / "model.py").write_text("x = 999\n")
    gitio.reset_hard(repo, sha)
    assert (repo / "model.py").read_text() == "x = 1\n"
    assert gitio.is_clean(repo)


def test_repo_root_raises_outside_repo(tmp_path):
    with pytest.raises(gitio.GitError):
        gitio.repo_root(tmp_path)


def test_changed_files_lists_worktree_edits(tmp_path):
    repo = _init_repo(tmp_path)
    sha = gitio.head_sha(repo)
    assert gitio.changed_files(repo, sha) == []
    (repo / "model.py").write_text("x = 2\n")
    assert gitio.changed_files(repo, sha) == ["model.py"]


def test_changed_files_pathspec_scopes_the_diff(tmp_path):
    repo = _init_repo(tmp_path)
    sub = repo / "models"
    sub.mkdir()
    (sub / "m.py").write_text("a = 1\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "add sub"], check=True)
    sha = gitio.head_sha(repo)
    (repo / "model.py").write_text("x = 2\n")  # unrelated change at root
    (sub / "m.py").write_text("a = 2\n")  # change in the scoped dir
    assert gitio.changed_files(repo, sha) == sorted(["model.py", "models/m.py"])
    assert gitio.changed_files(repo, sha, pathspec="models") == ["models/m.py"]  # scoped
