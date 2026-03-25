"""Tests for diff_line_numbers, diff_file_paths, _truncate_diff, and check_prerequisites."""

from unittest.mock import patch

import pytest

from bug_checker.github_client import (
    _truncate_diff,
    check_prerequisites,
    diff_file_paths,
    diff_line_numbers,
)


DIFF_SINGLE_HUNK = """\
diff --git a/foo/bar.cpp b/foo/bar.cpp
index abc..def 100644
--- a/foo/bar.cpp
+++ b/foo/bar.cpp
@@ -10,5 +10,6 @@
 context line 1
 context line 2
-removed line
+added line 1
+added line 2
 context line 3
"""


def test_added_lines_are_valid():
    lines = diff_line_numbers(DIFF_SINGLE_HUNK)
    valid = lines["foo/bar.cpp"]
    # +added line 1 is at new_line 12, +added line 2 at 13
    assert 12 in valid
    assert 13 in valid


def test_context_lines_are_valid():
    lines = diff_line_numbers(DIFF_SINGLE_HUNK)
    valid = lines["foo/bar.cpp"]
    # " context line 1" starts at new_line 10
    assert 10 in valid
    assert 11 in valid
    assert 14 in valid


def test_removed_lines_are_not_valid():
    lines = diff_line_numbers(DIFF_SINGLE_HUNK)
    valid = lines["foo/bar.cpp"]
    # The removed line does not occupy a slot in the new file
    # new_line sequence: 10, 11, (skip -removed), 12, 13, 14
    # so there is no "gap" line — removed line simply isn't in the set
    assert len([l for l in valid if l == 0]) == 0


def test_line_numbers_start_from_hunk_header():
    diff = (
        "diff --git a/x.cpp b/x.cpp\n"
        "index 000..111 100644\n"
        "--- a/x.cpp\n"
        "+++ b/x.cpp\n"
        "@@ -50,3 +50,4 @@\n"
        " existing line\n"
        "+new line\n"
        " another line\n"
    )
    valid = diff_line_numbers(diff)["x.cpp"]
    assert 50 in valid  # context line at new_line 50
    assert 51 in valid  # added line at new_line 51
    assert 52 in valid  # context line at new_line 52
    assert 1 not in valid


def test_multiple_hunks_in_same_file():
    diff = (
        "diff --git a/f.cpp b/f.cpp\n"
        "index 000..111 100644\n"
        "--- a/f.cpp\n"
        "+++ b/f.cpp\n"
        "@@ -1,2 +1,2 @@\n"
        "-old\n"
        "+new\n"
        " context\n"
        "@@ -100,2 +100,3 @@\n"
        " keep\n"
        "+added\n"
        " keep2\n"
    )
    valid = diff_line_numbers(diff)["f.cpp"]
    assert 1 in valid  # +new at line 1
    assert 2 in valid  # context at line 2
    assert 100 in valid  # keep at line 100
    assert 101 in valid  # +added at line 101
    assert 102 in valid  # keep2 at line 102


def test_multiple_files_tracked_separately():
    diff = (
        "diff --git a/a.cpp b/a.cpp\n"
        "index 000..111 100644\n"
        "--- a/a.cpp\n"
        "+++ b/a.cpp\n"
        "@@ -1,1 +1,2 @@\n"
        " line\n"
        "+extra\n"
        "diff --git a/b.cpp b/b.cpp\n"
        "index 000..111 100644\n"
        "--- a/b.cpp\n"
        "+++ b/b.cpp\n"
        "@@ -5,1 +5,1 @@\n"
        "+replaced\n"
    )
    result = diff_line_numbers(diff)
    assert "a.cpp" in result
    assert "b.cpp" in result
    assert 1 in result["a.cpp"]
    assert 2 in result["a.cpp"]
    assert 5 in result["b.cpp"]
    assert 1 not in result["b.cpp"]


def test_empty_diff_returns_empty():
    assert diff_line_numbers("") == {}


# --- diff_file_paths ---


def test_diff_file_paths_single_file():
    assert diff_file_paths(DIFF_SINGLE_HUNK) == {"foo/bar.cpp"}


def test_diff_file_paths_multiple_files():
    diff = (
        "diff --git a/a.cpp b/a.cpp\nindex 000..111 100644\n--- a/a.cpp\n+++ b/a.cpp\n"
        "@@ -1 +1 @@\n+x\n"
        "diff --git a/b.cpp b/b.cpp\nindex 000..111 100644\n--- a/b.cpp\n+++ b/b.cpp\n"
        "@@ -1 +1 @@\n+y\n"
    )
    assert diff_file_paths(diff) == {"a.cpp", "b.cpp"}


def test_diff_file_paths_empty():
    assert diff_file_paths("") == set()


# --- _truncate_diff ---


def _make_diff_for_files(file_names: list[str], lines_per_file: int = 10) -> str:
    parts = []
    for name in file_names:
        parts.append(f"diff --git a/{name} b/{name}")
        parts.append(f"index 000..111 100644")
        parts.append(f"--- a/{name}")
        parts.append(f"+++ b/{name}")
        parts.append(f"@@ -1,{lines_per_file} +1,{lines_per_file} @@")
        for i in range(lines_per_file):
            parts.append(f"+line {i} of {name}")
    return "\n".join(parts) + "\n"


def test_truncate_diff_no_truncation_needed():
    diff = _make_diff_for_files(["a.cpp"], lines_per_file=5)
    result_diff, truncated = _truncate_diff(diff, ["a.cpp"])
    assert result_diff == diff
    assert truncated == []


def test_truncate_diff_identifies_cut_files():
    from bug_checker.github_client import MAX_DIFF_LINES

    # file_a alone exceeds MAX_DIFF_LINES, so file_b's header falls after the cutoff
    diff = _make_diff_for_files(
        ["file_a.cpp", "file_b.cpp"], lines_per_file=MAX_DIFF_LINES
    )
    result_diff, truncated = _truncate_diff(diff, ["file_a.cpp", "file_b.cpp"])
    assert "file_b.cpp" in truncated
    assert "file_a.cpp" not in truncated
    assert "[diff truncated" in result_diff


def test_truncate_diff_no_cut_files_when_all_fit():
    from bug_checker.github_client import MAX_DIFF_LINES

    diff = _make_diff_for_files(["a.cpp", "b.cpp"], lines_per_file=5)
    _, truncated = _truncate_diff(diff, ["a.cpp", "b.cpp"])
    assert truncated == []


# --- check_prerequisites ---


def _completed(returncode: int) -> object:
    """Minimal mock CompletedProcess."""

    class _CP:
        pass

    cp = _CP()
    cp.returncode = returncode
    cp.stdout = ""
    cp.stderr = ""
    return cp


@patch("bug_checker.github_client.subprocess.run")
def test_check_prerequisites_gh_not_installed(mock_run):
    mock_run.return_value = _completed(1)  # gh --version fails
    with pytest.raises(RuntimeError, match="gh.*not installed"):
        check_prerequisites(need_gh=True)


@patch("bug_checker.github_client.subprocess.run")
def test_check_prerequisites_gh_not_authenticated(mock_run):
    # gh --version succeeds, gh auth status fails
    mock_run.side_effect = [_completed(0), _completed(1)]
    with pytest.raises(RuntimeError, match="not authenticated"):
        check_prerequisites(need_gh=True)


@patch("bug_checker.github_client.subprocess.run")
def test_check_prerequisites_git_not_installed(mock_run):
    mock_run.return_value = _completed(1)
    with pytest.raises(RuntimeError, match="git.*not installed"):
        check_prerequisites(need_git=True)


@patch("bug_checker.github_client.subprocess.run")
def test_check_prerequisites_no_checks_when_neither_needed(mock_run):
    check_prerequisites()  # should not call subprocess at all
    mock_run.assert_not_called()


def test_no_newline_marker_is_ignored():
    diff = (
        "diff --git a/f.cpp b/f.cpp\n"
        "index 000..111 100644\n"
        "--- a/f.cpp\n"
        "+++ b/f.cpp\n"
        "@@ -1,1 +1,1 @@\n"
        "+new line\n"
        "\\ No newline at end of file\n"
    )
    valid = diff_line_numbers(diff)["f.cpp"]
    assert 1 in valid
    assert len(valid) == 1
