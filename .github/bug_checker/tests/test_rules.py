"""Tests for rule loading and targeting."""

from bug_checker.rules import Rule, group_rules, load_manifest, load_rules, select_rules


def _rule(paths=None, labels=None) -> Rule:
    return Rule(
        id="test",
        file="test.md",
        severity="warning",
        suggest_fix=False,
        model=None,
        group=None,
        paths=paths or [],
        labels=labels or [],
    )


def test_load_manifest():
    manifest = load_manifest()
    assert "rules" in manifest
    assert "ccl-ring-buffer-mismatch" in manifest["rules"]
    assert "reshape-dim-check" in manifest["rules"]


def test_load_rules():
    rules = load_rules()
    assert len(rules) >= 2
    ids = {r.id for r in rules}
    assert "ccl-ring-buffer-mismatch" in ids
    assert "reshape-dim-check" in ids
    for rule in rules:
        assert rule.content, f"Rule {rule.id} has no content"


def test_rule_matches_by_path():
    rule = _rule(paths=["ttnn/cpp/ttnn/operations/ccl/**"])
    assert rule.matches_pr(["ttnn/cpp/ttnn/operations/ccl/all_gather/foo.cpp"], [])
    assert not rule.matches_pr(["ttnn/cpp/ttnn/operations/data_movement/bar.cpp"], [])


def test_rule_matches_by_label():
    rule = _rule(labels=["area:ccl"])
    assert rule.matches_pr([], ["area:ccl"])
    assert not rule.matches_pr([], ["area:ops"])


# --- match_reason ---


def test_match_reason_returns_none_when_no_match():
    rule = _rule(paths=["foo/**"], labels=["area:foo"])
    assert rule.match_reason(["bar/x.cpp"], ["area:bar"]) is None


def test_match_reason_identifies_path_and_pattern():
    rule = _rule(paths=["ttnn/cpp/ttnn/operations/ccl/**"])
    reason = rule.match_reason(["ttnn/cpp/ttnn/operations/ccl/foo.cpp"], [])
    assert reason is not None
    assert "ttnn/cpp/ttnn/operations/ccl/foo.cpp" in reason
    assert "ttnn/cpp/ttnn/operations/ccl/**" in reason


def test_match_reason_identifies_label():
    rule = _rule(labels=["area:ccl"])
    reason = rule.match_reason([], ["area:ccl"])
    assert reason is not None
    assert "area:ccl" in reason


def test_match_reason_prefers_path_over_label():
    rule = _rule(paths=["foo/**"], labels=["area:foo"])
    # Both match — path should appear in the reason (checked first)
    reason = rule.match_reason(["foo/bar.cpp"], ["area:foo"])
    assert "foo/bar.cpp" in reason


def test_matches_pr_delegates_to_match_reason():
    rule = _rule(paths=["foo/**"])
    assert rule.matches_pr(["foo/bar.cpp"], []) is True
    assert rule.matches_pr(["baz/bar.cpp"], []) is False


# --- orphan rule ---


def test_orphan_rule_never_matches():
    """A rule with no paths and no labels can never be selected."""
    rule = _rule()
    assert rule.match_reason(["any/file.cpp"], ["any:label"]) is None
    assert not rule.matches_pr(["any/file.cpp"], ["any:label"])


def test_select_rules():
    rules = load_rules()
    selected = select_rules(
        rules,
        changed_files=["ttnn/cpp/ttnn/operations/ccl/something.cpp"],
        pr_labels=[],
    )
    assert any(r.id == "ccl-ring-buffer-mismatch" for r in selected)
    assert not any(r.id == "reshape-dim-check" for r in selected)


def test_select_rules_by_label():
    rules = load_rules()
    selected = select_rules(rules, changed_files=[], pr_labels=["area:ops"])
    assert any(r.id == "reshape-dim-check" for r in selected)


def test_group_rules_isolated():
    rules = [
        Rule(
            id="a",
            file="a.md",
            severity="warning",
            suggest_fix=False,
            model=None,
            group=None,
        ),
        Rule(
            id="b",
            file="b.md",
            severity="warning",
            suggest_fix=False,
            model=None,
            group=None,
        ),
    ]
    groups = group_rules(rules)
    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)


def test_group_rules_grouped():
    rules = [
        Rule(
            id="a",
            file="a.md",
            severity="warning",
            suggest_fix=False,
            model=None,
            group="grp",
        ),
        Rule(
            id="b",
            file="b.md",
            severity="warning",
            suggest_fix=False,
            model=None,
            group="grp",
        ),
        Rule(
            id="c",
            file="c.md",
            severity="warning",
            suggest_fix=False,
            model=None,
            group=None,
        ),
    ]
    groups = group_rules(rules)
    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]
