#!/usr/bin/env python3
"""Parity tests for `.github/scripts/select_owners_to_ping.py`.

These lock the behaviour that was ported out of the inline bash step
`Select owners for notification` in codeowners-group-analysis.yaml, and guard
the selector against regressions from the upcoming OOO-filter change.

Run: python3 -m pytest .github/scripts/utils/test_select_owners_to_ping.py
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

# Load the sibling script (it lives in .github/scripts, not a package).
_SCRIPT = Path(__file__).resolve().parents[1] / "select_owners_to_ping.py"
_spec = importlib.util.spec_from_file_location("select_owners_to_ping", _SCRIPT)
soi = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(soi)


def run(monkeypatch, env, members_file_content=None, tmp_path=None):
    """Invoke Selector().select() with a controlled environment."""
    for key in (
        "TEAMS",
        "INDIVIDUALS",
        "APPROVED_REVIEWERS",
        "MOREH_TEAM_MEMBERS",
        "PR_AUTHOR_LOGIN",
        "TEAM_MEMBERS",
        "TEAM_MEMBERS_FILE",
    ):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    if members_file_content is not None:
        assert tmp_path is not None
        mf = tmp_path / "team_members.txt"
        mf.write_text(members_file_content, encoding="utf-8")
        monkeypatch.setenv("TEAM_MEMBERS_FILE", str(mf))
    owners, groups, no_owners = soi.Selector().select()
    return owners, groups, no_owners


# --- unit-level: match semantics --------------------------------------------
def test_approved_is_substring():
    assert soi.approved("foo", "foobar,baz") is True  # original grep -q behaviour


def test_approved_exact_requires_full_token():
    # exact per-file check must NOT treat 'foobar' as approval for 'foo'
    assert soi.approved_exact("foo", "foobar,baz") is False
    assert soi.approved_exact("foo", "foo,baz") is True


# --- individual patterns -----------------------------------------------------
def test_individual_pattern_picks_two(monkeypatch):
    owners, groups, no_owners = run(
        monkeypatch,
        {"INDIVIDUALS": "src/*.cpp:alice|,bob|,carol|:src/a.cpp"},
    )
    assert len(owners) == 2
    assert set(owners) <= {"alice", "bob", "carol"}
    assert groups == []
    assert no_owners is False


def test_individual_excludes_author_and_moreh(monkeypatch):
    owners, _, _ = run(
        monkeypatch,
        {
            "INDIVIDUALS": "src/*.cpp:alice|,bob|,carol|:src/a.cpp",
            "PR_AUTHOR_LOGIN": "alice",
            "MOREH_TEAM_MEMBERS": "bob",
        },
    )
    assert owners == ["carol"]


def test_individual_already_approved_skipped(monkeypatch):
    owners, groups, no_owners = run(
        monkeypatch,
        {"INDIVIDUALS": "src/*.cpp:alice|,bob|:src/a.cpp", "APPROVED_REVIEWERS": "alice"},
    )
    assert owners == []
    assert groups == []
    assert no_owners is True


# --- teams -------------------------------------------------------------------
def test_team_with_slack_group_pings_group(monkeypatch, tmp_path):
    owners, groups, _ = run(
        monkeypatch,
        {"TEAMS": "@tenstorrent/metalium-developers-infra:src/x.cpp"},
        members_file_content="@tenstorrent/metalium-developers-infra:m1,m2,m3",
        tmp_path=tmp_path,
    )
    assert groups == ["S0985AN7TC5"]
    assert owners == []


def test_team_without_slack_group_picks_two(monkeypatch, tmp_path):
    owners, groups, _ = run(
        monkeypatch,
        {"TEAMS": "@tenstorrent/some-random-team:src/x.cpp"},
        members_file_content="@tenstorrent/some-random-team:u1,u2,u3,u4",
        tmp_path=tmp_path,
    )
    assert len(owners) == 2
    assert set(owners) <= {"u1", "u2", "u3", "u4"}
    assert groups == []


def test_api_owners_always_includes_required_reviewer(monkeypatch, tmp_path):
    owners, _, _ = run(
        monkeypatch,
        {"TEAMS": "@tenstorrent/metalium-api-owners:tt_metal/api/foo.h"},
        members_file_content="@tenstorrent/metalium-api-owners:akerteszTT,p1,p2,p3",
        tmp_path=tmp_path,
    )
    # akerteszTT is added in ADDITION to the (up to) 2 random picks.
    assert "akerteszTT" in owners
    assert len(owners) == 3


def test_api_owners_non_api_files_no_special_case(monkeypatch, tmp_path):
    owners, _, _ = run(
        monkeypatch,
        {"TEAMS": "@tenstorrent/metalium-api-owners:programming_examples/foo.cpp"},
        members_file_content="@tenstorrent/metalium-api-owners:akerteszTT,p1,p2,p3",
        tmp_path=tmp_path,
    )
    assert len(owners) == 2  # standard pick-two, no forced akerteszTT


def test_eltwise_key_member_approved_skips_group(monkeypatch, tmp_path):
    owners, groups, no_owners = run(
        monkeypatch,
        {
            "TEAMS": "@tenstorrent/metalium-developers-eltwise:src/e.cpp",
            "APPROVED_REVIEWERS": "dchenTT",
        },
        members_file_content="@tenstorrent/metalium-developers-eltwise:dchenTT,x1",
        tmp_path=tmp_path,
    )
    assert groups == []
    assert no_owners is True


def test_eltwise_no_key_member_pings_group(monkeypatch, tmp_path):
    _, groups, _ = run(
        monkeypatch,
        {"TEAMS": "@tenstorrent/metalium-developers-eltwise:src/e.cpp"},
        members_file_content="@tenstorrent/metalium-developers-eltwise:x1,x2",
        tmp_path=tmp_path,
    )
    assert groups == ["S0ABKSS1D3R"]


def test_team_member_sentinel_yields_no_owners(monkeypatch, tmp_path):
    owners, groups, no_owners = run(
        monkeypatch,
        {"TEAMS": "@tenstorrent/some-random-team:src/x.cpp"},
        members_file_content="@tenstorrent/some-random-team:team-not-found",
        tmp_path=tmp_path,
    )
    assert owners == []
    assert groups == []
    assert no_owners is True


def test_per_file_approval_uses_exact_match(monkeypatch, tmp_path):
    # The only "approval" (foobar) is a substring of member `foo` but not an
    # exact match. The per-file team-skip check must use EXACT matching, so the
    # team is NOT considered covered and is still selected for a ping (old
    # substring behaviour would have skipped it, yielding no owners at all).
    #
    # Within that selection, the unapproved-member collection uses substring
    # matching (faithful to the original `grep -q`), so `foo` is dropped as
    # "approved" and only `baz` remains. The point of this test is the team is
    # selected at all — i.e. no_owners is False.
    owners, _, no_owners = run(
        monkeypatch,
        {
            "TEAMS": "@tenstorrent/some-random-team:src/x.cpp",
            "APPROVED_REVIEWERS": "foobar",
        },
        members_file_content="@tenstorrent/some-random-team:foo,baz",
        tmp_path=tmp_path,
    )
    assert owners == ["baz"]
    assert no_owners is False


def test_empty_inputs_report_no_owners(monkeypatch):
    owners, groups, no_owners = run(monkeypatch, {})
    assert owners == []
    assert groups == []
    assert no_owners is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
