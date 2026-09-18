#!/usr/bin/env python3
"""Tests for the generic CI-failure Jira adapter (MINFRA-1611)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from file_ci_failure_jira import (  # noqa: E402
    dedup_key,
    description_for,
    is_actionable,
    load_payload,
    route_for,
    summary_for,
)

ROUTING = json.loads((SCRIPTS_DIR / "ci_failure_routing.json").read_text())


def failure(**over):
    base = {
        "repository": "tenstorrent/tt-metal",
        "workflow": "All post-commit tests",
        "job": "ttnn unit tests",
        "test": "tests/ttnn/unit_tests/test_matmul.py::test_bcast",
        "run_url": "https://github.com/x/runs/1",
        "error_message": "assert 0.97 > 0.99",
        "category": "model",
        "error_layer": "kernel",
        "confidence": 0.9,
    }
    base.update(over)
    return base


def test_payload_accepts_object_list_or_wrapper(tmp_path):
    one, many = tmp_path / "a.json", tmp_path / "b.json"
    wrap = tmp_path / "c.json"
    one.write_text(json.dumps(failure()))
    many.write_text(json.dumps([failure(), failure(test="t2")]))
    wrap.write_text(json.dumps({"failures": [failure()]}))
    assert len(load_payload(one)) == 1
    assert len(load_payload(many)) == 2
    assert len(load_payload(wrap)) == 1


def test_low_confidence_is_not_filed():
    assert not is_actionable(failure(confidence=0.1), 0.6)
    assert is_actionable(failure(confidence=0.9), 0.6)


def test_explicit_actionable_overrides_confidence():
    assert not is_actionable(failure(confidence=0.99, actionable=False), 0.6)
    assert is_actionable(failure(confidence=0.01, actionable=True), 0.6)


def test_missing_confidence_is_filed():
    """The AI declining to score is not evidence the failure is spurious."""
    f = failure()
    del f["confidence"]
    assert is_actionable(f, 0.6)
    assert is_actionable(failure(confidence="n/a"), 0.6)


def test_route_matches_most_specific_first():
    infra = route_for(failure(error_layer="infra"), ROUTING)
    assert "infra" in infra["labels"]
    model = route_for(failure(error_layer="kernel", category="model"), ROUTING)
    assert "models" in model["labels"]


def test_unmatched_failure_falls_back_to_default():
    r = route_for(failure(category="something-new", error_layer="unknown"), ROUTING)
    assert r["project"] == ROUTING["default"]["project"]
    assert r["issue_type"] == ROUTING["default"]["issue_type"]


def test_route_never_lands_on_the_ai_ip_board():
    """RELEASE carries AI/IP commitments only; generic CI noise must not reach it."""
    projects = {ROUTING["default"]["project"]} | {r.get("project") for r in ROUTING["routes"]}
    assert "RELEASE" not in projects


def test_dedup_key_is_stable_across_runs_and_commits():
    a = dedup_key(failure(run_url="https://x/1", commit="aaa"))
    b = dedup_key(failure(run_url="https://x/2", commit="bbb"))
    assert a == b, "the same test failing again must comment, not open a duplicate"


def test_dedup_key_separates_different_tests():
    assert dedup_key(failure()) != dedup_key(failure(test="tests/other.py::test_x"))
    assert dedup_key(failure()) != dedup_key(failure(category="infra"))


def test_dedup_key_is_a_valid_jira_label():
    key = dedup_key(failure(test="tests/ttnn/unit_tests/test_matmul.py::test_bcast[8-16]"))
    assert " " not in key and len(key) <= 100


def test_description_quotes_the_error_and_links_back():
    d = description_for(failure(root_cause="pcc drift", suggested_action="retune"))
    for expected in ["assert 0.97 > 0.99", "pcc drift", "retune", "https://github.com/x/runs/1"]:
        assert expected in d


def test_description_omits_absent_fields():
    d = description_for({"job": "j", "error_message": "boom"})
    assert "Test:" not in d and "Root cause" not in d


def test_summary_is_bounded_and_identifies_the_failure():
    s = summary_for(failure())
    assert "test_bcast" in s and len(s) <= 250
    assert summary_for({"job": "build"}) == "CI: build failed"
