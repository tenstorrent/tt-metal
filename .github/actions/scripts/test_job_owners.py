#!/usr/bin/env python3
"""Tests for the pipeline_reorg-backed job -> owner lookup."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_owners import _load, lookup  # noqa: E402

FIXTURE = """\
# file comment
- name: fabric infra unit tests
  cmd: |
    ./run --name not-a-field
    pytest -k "owner_id: trap"
  owner_id: U07D5N7QL4U # Joseph Chu
  team: scaleout

- id: bh-mllama
  name: Llama 3.1-8B e2e tests
  skus:
    bh_p150:
      timeout: 30
  owner_id: U03PUAKE719 # Miguel Tairum Cruz
  team: models

- name: orphaned test
  owner_id: U000UNNAMED
  team: models

- name: teamless test
  owner_id: U111 # Someone
"""


def fixture_dir(tmp_path):
    (tmp_path / "a_tests.yaml").write_text(FIXTURE)
    return str(tmp_path)


def test_resolves_by_name_with_runner_tag_stripped(tmp_path):
    root = fixture_dir(tmp_path)
    assert lookup("Llama 3.1-8B e2e tests [bh_p150]", root=root) == {
        "owner": "Miguel Tairum Cruz",
        "team": "models",
    }
    assert lookup("fabric infra unit tests", root=root)["owner"] == "Joseph Chu"


def test_entry_fields_are_found_regardless_of_field_order(tmp_path):
    """Entries may open with "- id:" (name on the next line) or "- name:"."""
    table = _load(fixture_dir(tmp_path))
    assert "Llama 3.1-8B e2e tests" in table
    assert "fabric infra unit tests" in table


def test_cmd_block_content_never_leaks_into_fields(tmp_path):
    table = _load(fixture_dir(tmp_path))
    assert table["fabric infra unit tests"]["owner"] == "Joseph Chu"
    assert "not-a-field" not in str(table)


def test_unknown_job_is_none(tmp_path):
    assert lookup("no such job [r]", root=fixture_dir(tmp_path)) is None


def test_owner_without_a_name_comment_is_unresolvable(tmp_path):
    """A ticket must never show a raw Slack id."""
    assert lookup("orphaned test", root=fixture_dir(tmp_path)) is None


def test_entry_without_a_team_is_unresolvable(tmp_path):
    assert lookup("teamless test", root=fixture_dir(tmp_path)) is None


def test_duplicate_names_keep_the_first_hit(tmp_path):
    (tmp_path / "a.yaml").write_text("- name: dup\n  owner_id: U1 # First\n  team: models\n")
    (tmp_path / "b.yaml").write_text("- name: dup\n  owner_id: U2 # Second\n  team: ttnn\n")
    assert _load(str(tmp_path))["dup"]["owner"] == "First"


def test_real_tree_resolves_a_release_demo_job():
    """Shape check against the live registry; owners change, teams rarely do."""
    hit = lookup("Llama 3.1-8B e2e tests [bh_p150]")
    assert hit and hit["owner"] and hit["team"] == "models"


def test_real_tree_has_broad_coverage():
    from job_owners import _load as load

    assert len(load()) > 400
