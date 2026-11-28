"""RUN_REPORT.md is the SINGLE source of truth: bring-up, emit-e2e, and optimize each own a
marker-delimited section in ONE file. Writing one section must replace only that section (in place)
and preserve the others, regardless of write order — so the three phases (and the optimize phase's
live per-attempt rewrites) accumulate into one report."""

from __future__ import annotations

from scripts.tt_hw_planner.run_report import upsert_report_section


def _read(d):
    return (d / "RUN_REPORT.md").read_text()


def test_single_file_accumulates_three_phase_sections(tmp_path):
    upsert_report_section(tmp_path, "bringup", "# Bring-up\n\nplacement")
    upsert_report_section(tmp_path, "emit-e2e", "# E2E\n\nverdict PASS")
    upsert_report_section(tmp_path, "optimize", "# Optimize\n\nattempt 1")
    txt = _read(tmp_path)
    assert "# Bring-up" in txt and "# E2E" in txt and "# Optimize" in txt
    assert txt.index("Bring-up") < txt.index("E2E") < txt.index("Optimize")  # write order preserved
    for k in ("bringup", "emit-e2e", "optimize"):
        assert f"<!-- BEGIN {k} -->" in txt and f"<!-- END {k} -->" in txt


def test_optimize_live_update_replaces_in_place_and_preserves_others(tmp_path):
    upsert_report_section(tmp_path, "bringup", "# Bring-up\n\nB1")
    upsert_report_section(tmp_path, "optimize", "# Optimize\n\nattempt 1")
    # proactive live rewrite: optimize section grows each attempt — replace in place, never duplicate
    upsert_report_section(tmp_path, "optimize", "# Optimize\n\nattempt 1\nattempt 2")
    txt = _read(tmp_path)
    assert txt.count("<!-- BEGIN optimize -->") == 1  # replaced, not appended
    assert "attempt 2" in txt
    assert "B1" in txt  # bring-up section untouched by the optimize rewrite
    assert txt.index("Bring-up") < txt.index("Optimize")  # order stable across live updates


def test_idempotent_rewrite_same_content(tmp_path):
    upsert_report_section(tmp_path, "optimize", "# Optimize\n\nx")
    first = _read(tmp_path)
    upsert_report_section(tmp_path, "optimize", "# Optimize\n\nx")
    assert _read(tmp_path) == first
