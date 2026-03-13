"""Tests for rule loading and targeting."""

from bug_checker.rules import Rule, group_rules, load_manifest, load_rules, select_rules


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
    rule = Rule(
        id="test",
        file="test.md",
        severity="warning",
        suggest_fix=False,
        model=None,
        group=None,
        paths=["ttnn/cpp/ttnn/operations/ccl/**"],
        labels=[],
    )
    assert rule.matches_pr(["ttnn/cpp/ttnn/operations/ccl/all_gather/foo.cpp"], [])
    assert not rule.matches_pr(["ttnn/cpp/ttnn/operations/data_movement/bar.cpp"], [])


def test_rule_matches_by_label():
    rule = Rule(
        id="test",
        file="test.md",
        severity="warning",
        suggest_fix=False,
        model=None,
        group=None,
        paths=[],
        labels=["area:ccl"],
    )
    assert rule.matches_pr([], ["area:ccl"])
    assert not rule.matches_pr([], ["area:ops"])


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
        Rule(id="a", file="a.md", severity="warning", suggest_fix=False, model=None, group=None),
        Rule(id="b", file="b.md", severity="warning", suggest_fix=False, model=None, group=None),
    ]
    groups = group_rules(rules)
    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)


def test_group_rules_grouped():
    rules = [
        Rule(id="a", file="a.md", severity="warning", suggest_fix=False, model=None, group="grp"),
        Rule(id="b", file="b.md", severity="warning", suggest_fix=False, model=None, group="grp"),
        Rule(id="c", file="c.md", severity="warning", suggest_fix=False, model=None, group=None),
    ]
    groups = group_rules(rules)
    assert len(groups) == 2
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2]
