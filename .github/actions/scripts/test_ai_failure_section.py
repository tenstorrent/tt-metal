#!/usr/bin/env python3
"""Tests for the AI-Summary-to-Jira-section rendering."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ai_failure_section import section  # noqa: E402


def row(**over):
    base = {
        "job_name": "Gemma-4-31B e2e tests [bh_p150]",
        "job_url": "https://x.test/job/1",
        "status": "FAILURE",
        "category": "tt-metal:pcc",
        "error_message": "assert 0.97 > 0.99",
        "root_cause": "PCC drift after SFPI bump",
        "log_complete": True,
    }
    base.update(over)
    return base


def test_no_summary_renders_nothing():
    assert section(None) == ""
    assert section({"failed": [], "infra_failure": []}) == ""


def test_rows_render_as_the_other_information_section():
    out = section({"failed": [row()], "infra_failure": [row(job_name="build", root_cause="runner OOM")]})
    assert out.startswith("### Other information\n")
    assert "- Gemma-4-31B e2e tests [bh_p150]: PCC drift after SFPI bump [tt-metal:pcc]" in out
    assert "- build: runner OOM" in out


def test_error_message_backfills_a_missing_root_cause():
    out = section({"failed": [row(root_cause=None)]})
    assert "assert 0.97 > 0.99" in out
    assert "— error:" not in out


def test_the_verbatim_error_rides_along_with_the_root_cause():
    out = section({"failed": [row()]})
    assert "PCC drift after SFPI bump [tt-metal:pcc] — error: assert 0.97 > 0.99" in out


def test_subcategory_extends_the_category_tag():
    out = section({"failed": [row(subcategory="pcc_drop")]})
    assert "[tt-metal:pcc/pcc_drop]" in out


def test_subcategory_alone_still_tags():
    out = section({"failed": [row(category=None, subcategory="pcc_drop")]})
    assert "[pcc_drop]" in out


def test_a_row_with_no_cause_at_all_is_dropped():
    assert section({"failed": [row(root_cause=None, error_message=None)]}) == ""


def test_multiline_causes_collapse_to_one_bullet_line():
    out = section({"failed": [row(root_cause="line one\n  line two")]})
    assert "line one line two" in out
    assert out.count("\n- ") == 1


def test_incomplete_logs_are_flagged():
    out = section({"failed": [row(log_complete=False)]})
    assert "(log incomplete)" in out


def test_very_long_causes_are_bounded():
    out = section({"failed": [row(root_cause="x" * 1000)]})
    line = [l for l in out.splitlines() if l.startswith("- ")][0]
    assert len(line) < 400 and "..." in line
