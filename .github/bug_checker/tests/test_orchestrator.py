"""Tests for orchestrator internals — diff filtering."""

from bug_checker.orchestrator import _filter_diff_for_rule
from bug_checker.rules import Rule


def _rule(paths: list[str]) -> Rule:
    return Rule(
        id="test-rule",
        file="test.md",
        severity="warning",
        suggest_fix=False,
        model=None,
        group=None,
        paths=paths,
        labels=[],
    )


DIFF_TWO_FILES = """\
diff --git a/foo/bar.cpp b/foo/bar.cpp
index abc..def 100644
--- a/foo/bar.cpp
+++ b/foo/bar.cpp
@@ -1,3 +1,4 @@
+// added line
 existing line
diff --git a/baz/qux.cpp b/baz/qux.cpp
index 111..222 100644
--- a/baz/qux.cpp
+++ b/baz/qux.cpp
@@ -5,2 +5,3 @@
+// another added line
 other line
"""


def test_returns_empty_when_no_files_match():
    rule = _rule(["ttnn/cpp/**"])
    result = _filter_diff_for_rule(DIFF_TWO_FILES, ["foo/bar.cpp", "baz/qux.cpp"], rule)
    assert result == ""


def test_returns_empty_when_changed_files_list_is_empty():
    rule = _rule(["foo/**"])
    result = _filter_diff_for_rule(DIFF_TWO_FILES, [], rule)
    assert result == ""


def test_keeps_only_matching_section():
    rule = _rule(["foo/**"])
    result = _filter_diff_for_rule(DIFF_TWO_FILES, ["foo/bar.cpp", "baz/qux.cpp"], rule)
    assert "foo/bar.cpp" in result
    assert "baz/qux.cpp" not in result


def test_keeps_multiple_matching_sections():
    rule = _rule(["foo/**", "baz/**"])
    result = _filter_diff_for_rule(DIFF_TWO_FILES, ["foo/bar.cpp", "baz/qux.cpp"], rule)
    assert "foo/bar.cpp" in result
    assert "baz/qux.cpp" in result


def test_full_section_content_is_preserved():
    rule = _rule(["foo/**"])
    result = _filter_diff_for_rule(DIFF_TWO_FILES, ["foo/bar.cpp"], rule)
    assert "// added line" in result
    assert "@@ -1,3 +1,4 @@" in result


def test_last_section_is_flushed():
    # The last file in the diff is the one that needs the explicit end-of-loop flush
    rule = _rule(["baz/**"])
    result = _filter_diff_for_rule(DIFF_TWO_FILES, ["foo/bar.cpp", "baz/qux.cpp"], rule)
    assert "baz/qux.cpp" in result
    assert "// another added line" in result


def test_single_file_diff():
    diff = (
        "diff --git a/only/file.cpp b/only/file.cpp\n"
        "index 000..111 100644\n"
        "--- a/only/file.cpp\n"
        "+++ b/only/file.cpp\n"
        "@@ -1 +1,2 @@\n"
        "+// new\n"
        " old\n"
    )
    rule = _rule(["only/**"])
    result = _filter_diff_for_rule(diff, ["only/file.cpp"], rule)
    assert "only/file.cpp" in result
    assert "// new" in result


def test_empty_diff_returns_empty():
    rule = _rule(["foo/**"])
    assert _filter_diff_for_rule("", ["foo/bar.cpp"], rule) == ""
