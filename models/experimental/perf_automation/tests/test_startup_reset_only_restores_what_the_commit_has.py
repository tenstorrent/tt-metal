"""Pin: the startup reset must ask the commit what it has before restoring.

`git checkout <sha> -- <paths>` fails the WHOLE invocation when any single path is unknown to that
commit, and prints one "did not match any file(s) known to git" line per path. A model directory
that is untracked on the checked-out branch -- it lives on its own branch, or is staged but never
committed -- therefore turned Step 2 into ~100 error lines that restored nothing, and then reported
itself as "skipped", which reads as deliberate.

Observed 2026-09-24: optimize launched from the tool branch, where the model is untracked, emitted
97 such lines and still announced "model reset to published <sha>".
"""

from __future__ import annotations

import inspect
import subprocess

import pytest

from models.experimental.perf_automation.agent import gitio


def _run(args, cwd):
    subprocess.run(args, cwd=str(cwd), check=True, capture_output=True)


@pytest.fixture()
def repo(tmp_path):
    _run(["git", "init", "-q"], tmp_path)
    _run(["git", "config", "user.email", "t@t"], tmp_path)
    _run(["git", "config", "user.name", "t"], tmp_path)
    (tmp_path / "committed.py").write_text("x = 1\n")
    _run(["git", "add", "committed.py"], tmp_path)
    _run(["git", "commit", "-qm", "base"], tmp_path)
    return tmp_path


def test_present_at_returns_only_paths_the_commit_has(repo):
    (repo / "staged_only.py").write_text("y = 2\n")
    _run(["git", "add", "staged_only.py"], repo)
    head = gitio.head_sha(repo)

    got = gitio.present_at(repo, head, ["committed.py", "staged_only.py"])
    assert got == ["committed.py"], got


def test_present_at_is_empty_when_the_commit_has_none_of_them(repo):
    """The real case: every model file is staged-but-uncommitted, so there is nothing to restore."""
    head = gitio.head_sha(repo)
    assert gitio.present_at(repo, head, ["a.py", "b/c.py"]) == []


def test_present_at_handles_no_paths_without_calling_git(repo):
    head = gitio.head_sha(repo)
    assert gitio.present_at(repo, head, []) == []
    assert gitio.present_at(repo, head, None) == []


def test_present_at_never_raises_on_a_bad_sha(repo):
    """The reset must degrade to 'restore nothing', never take the run down."""
    assert gitio.present_at(repo, "0" * 40, ["committed.py"]) == []


def test_checkout_would_still_fail_on_a_mixed_pathspec(repo):
    """WHY the filter is needed: git rejects the whole invocation, not just the unknown path."""
    (repo / "staged_only.py").write_text("y = 2\n")
    _run(["git", "add", "staged_only.py"], repo)
    head = gitio.head_sha(repo)
    with pytest.raises(gitio.GitError):  # allow-pytest.raises: no expect_error fixture
        gitio.checkout(repo, head, pathspec=["committed.py", "staged_only.py"])


def test_startup_reset_filters_before_it_checks_out():
    from models.experimental.perf_automation.agent import before_loop

    src = inspect.getsource(before_loop)
    at_filter = src.find("present_at(repo, head, code_dirty)")
    at_checkout = src.find("gitio.checkout(repo, head, pathspec=restorable)")
    assert at_filter >= 0, "the startup reset must ask the commit what it has"
    assert at_checkout > at_filter, "the filter must run BEFORE the checkout"
    assert "pathspec=code_dirty" not in src, "the unfiltered list must never reach checkout"


def test_startup_reset_does_not_claim_a_reset_it_did_not_do():
    """'skipped' read as deliberate. When nothing is restorable, say why."""
    from models.experimental.perf_automation.agent import before_loop

    src = inspect.getsource(before_loop)
    assert "no reset:" in src, "the nothing-to-restore case must name itself, not report success"
    assert "untracked on this branch" in src
